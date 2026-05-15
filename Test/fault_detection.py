"""
fault_detection.py — Detección de fallas trifásicas en tiempo real
Hardware : Jetson Nano + Waveshare High-Precision AD HAT (ADS1263)
Sistema  : 60 Hz, ventana deslizante de 2 ciclos, VA = 0° referencia

Features del modelo (orden estricto, 12 en total):
  VA_mag, VB_mag, VC_mag,
  VA_ang, VB_ang, VC_ang,
  IA_mag, IB_mag, IC_mag,
  IA_ang, IB_ang, IC_ang

Canales ADC:
  IN0=VA  IN1=VB  IN2=VC  IN5=IA  IN6=IB  IN7=IC

Uso:
  sudo python3 fault_detection.py
"""

import sys
import os
import time
import threading
import datetime
import numpy as np
from collections import deque

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
REGISTER    = os.path.join(RESULTS_DIR, "registers.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Modelos (ajusta rutas si es necesario)
MODEL_PATHS = {
    "ieee5": {
        "model":  os.path.join(BASE_DIR, "Models", "model_det_ieee5.pkl"),
        "scaler": os.path.join(BASE_DIR, "Models", "scaler_det_ieee5.pkl"),
    },
    "ieee13": {
        "model":  os.path.join(BASE_DIR, "Models", "model_det_ieee13.pkl"),
        "scaler": os.path.join(BASE_DIR, "Models", "scaler_det_ieee13.pkl"),
    },
}

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/santiago/PF_Smart-Relay/SPI/High-Pricision_AD_HAT/python')
import ADS1263
import joblib
import Jetson.GPIO as GPIO

# ── Configuración ─────────────────────────────────────────────────────────────
REF            = 5.08
FREQ           = 60.0
RATE_STR       = 'ADS1263_38400SPS'
KV             = 50.0          # 1 V_relay → 50 V reales
KI             = 30.0          # 1 V_relay → 30 A reales
CYCLES_WINDOW  = 2
CHANNELS       = [0, 1, 2, 5, 6, 7]
IS_VOLTAGE     = {0: True, 1: True, 2: True, 5: False, 6: False, 7: False}
FS_PER_CHANNEL = 38400 / len(CHANNELS)
WINDOW_SIZE    = int(round(FS_PER_CHANNEL * CYCLES_WINDOW / FREQ))

GPIO_PIN       = 40            # Pin físico número 40

# ── GPIO setup ────────────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BOARD)
GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)

# ── ADC setup ─────────────────────────────────────────────────────────────────
ADC = ADS1263.ADS1263()
if ADC.ADS1263_init_ADC1(RATE_STR) == -1:
    print("[ERROR] No se pudo inicializar el ADS1263.")
    GPIO.cleanup()
    sys.exit(1)
ADC.ADS1263_SetMode(0)

# ── Selección del sistema IEEE ────────────────────────────────────────────────
def seleccionar_sistema():
    while True:
        print("\n¿Qué sistema IEEE desea usar?")
        print("  [1] IEEE 5")
        print("  [2] IEEE 13")
        op = input("Opción: ").strip()
        if op == "1":
            return "ieee5"
        elif op == "2":
            return "ieee13"
        print("  Opción inválida. Ingrese 1 o 2.")

sistema   = seleccionar_sistema()
paths     = MODEL_PATHS[sistema]
model     = joblib.load(paths["model"])
scaler    = joblib.load(paths["scaler"])
print(f"\n[OK] Modelos IEEE {sistema[-2:].upper()} cargados.")

# ── Utilidades ────────────────────────────────────────────────────────────────
def raw_to_voltage(raw: int) -> float:
    if raw >> 31:
        return -(REF * 2 - raw / 2_147_483_648.0 * REF)
    return raw / 2_147_483_647.0 * REF

def compute_rms_phasor(samples: np.ndarray, fs: float):
    n   = len(samples)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    k = np.argmin(np.abs(freqs - FREQ))
    dft = np.fft.rfft(samples)
    mag_rms = (2 * np.abs(dft[k]) / n) / np.sqrt(2)
    angle   = np.degrees(np.angle(dft[k]))
    return mag_rms, angle

def registrar_falla(fecha, hora, V, I, angV, angI):
    """Append de la falla en registers.txt."""
    linea = (
        f"{fecha} | {hora} | "
        f"VA={V[0]:.2f}V∠{angV[0]:+.1f}° "
        f"VB={V[1]:.2f}V∠{angV[1]:+.1f}° "
        f"VC={V[2]:.2f}V∠{angV[2]:+.1f}° | "
        f"IA={I[0]:.2f}A∠{angI[0]:+.1f}° "
        f"IB={I[1]:.2f}A∠{angI[1]:+.1f}° "
        f"IC={I[2]:.2f}A∠{angI[2]:+.1f}°\n"
    )
    with open(REGISTER, "a") as f:
        f.write(linea)

def fmt_ang(a):
    return f"{a:+.1f}°"

# ── Display ───────────────────────────────────────────────────────────────────
PHASES   = ['A', 'B', 'C']
N_LINES  = 11

def print_display(V_mag, V_ang, I_mag, I_ang, elapsed, ready, fault=False):
    status = "🔴 FALLA DETECTADA" if fault else "🟢 Normal"
    lines  = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append(f"║  ADS1263 · IEEE {sistema[-2:].upper()} · 60 Hz · Estado: {status:<22}║")
    lines.append("╠═══════════╦═══════════════════════╦═════════════════════════╣")
    lines.append("║  Fase     ║  Tensión               ║  Corriente              ║")
    lines.append("╠═══════════╬═══════════════════════╬═════════════════════════╣")
    for i, ph in enumerate(PHASES):
        if ready:
            v_str = f"{V_mag[i]:>7.2f} V  ∠{fmt_ang(V_ang[i]):>8}"
            i_str = f"{I_mag[i]:>7.2f} A  ∠{fmt_ang(I_ang[i]):>8}"
        else:
            v_str = "   acumulando...      "
            i_str = "   acumulando...      "
        lines.append(f"║  Fase {ph}   ║  {v_str}  ║  {i_str}  ║")
    lines.append("╚═══════════╩═══════════════════════╩═════════════════════════╝")
    lines.append(f"  t={elapsed:>6.1f}s | ventana={CYCLES_WINDOW} ciclos | {WINDOW_SIZE} muestras/canal")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"\033[{len(lines)}A")
    sys.stdout.flush()

# ── Hilo que escucha "Falla despejada" ───────────────────────────────────────
fault_cleared = threading.Event()

def listen_for_clear():
    while True:
        cmd = input()
        if cmd.strip().lower() == "falla despejada":
            fault_cleared.set()

listener = threading.Thread(target=listen_for_clear, daemon=True)
listener.start()

# ── Bucle principal ───────────────────────────────────────────────────────────
buffers    = {ch: deque(maxlen=WINDOW_SIZE) for ch in CHANNELS}
V_mag      = [0.0, 0.0, 0.0]
V_ang      = [0.0, 0.0, 0.0]
I_mag      = [0.0, 0.0, 0.0]
I_ang      = [0.0, 0.0, 0.0]

print(f"\nVentana: {WINDOW_SIZE} muestras/canal ({CYCLES_WINDOW} ciclos @ ~{FS_PER_CHANNEL:.0f} Sa/s)")
print("Escriba  'Falla despejada'  para reiniciar tras una falla.\n")
time.sleep(0.3)

t_start    = time.time()
total_samp = 0
in_fault   = False

try:
    while True:
        # ── Adquisición ───────────────────────────────────────────────────────
        for ch in CHANNELS:
            raw = ADC.ADS1263_GetChannalValue(ch)
            buffers[ch].append(raw_to_voltage(raw))
        total_samp += 1
        ready = (total_samp >= WINDOW_SIZE)

        # ── Cálculo de fasores ────────────────────────────────────────────────
        if ready:
            fs_ch   = (total_samp / (time.time() - t_start)) / len(CHANNELS)
            rms_ang = {}
            for ch in CHANNELS:
                arr          = np.array(buffers[ch])
                r, a         = compute_rms_phasor(arr, fs_ch)
                scale        = KV if IS_VOLTAGE[ch] else KI
                rms_ang[ch]  = (r * scale, a)

            ref_ang = rms_ang[0][1]   # VA = 0°

            V_mag = [rms_ang[ch][0] for ch in [0, 1, 2]]
            V_ang = [rms_ang[ch][1] - ref_ang for ch in [0, 1, 2]]
            I_mag = [rms_ang[ch][0] for ch in [5, 6, 7]]
            I_ang = [rms_ang[ch][1] - ref_ang for ch in [5, 6, 7]]

            # ── Inferencia ────────────────────────────────────────────────────
            features = np.array([[
                V_mag[0], V_mag[1], V_mag[2],
                V_ang[0], V_ang[1], V_ang[2],
                I_mag[0], I_mag[1], I_mag[2],
                I_ang[0], I_ang[1], I_ang[2],
            ]])
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]

            if pred == 1 and not in_fault:
                # ── FALLA DETECTADA ───────────────────────────────────────────
                in_fault = True
                GPIO.output(GPIO_PIN, GPIO.HIGH)

                now   = datetime.datetime.now()
                fecha = now.strftime("%Y-%m-%d")
                hora  = now.strftime("%H:%M:%S")
                registrar_falla(fecha, hora, V_mag, I_mag, V_ang, I_ang)

                # Mostrar alerta y bloquear hasta que se despeje
                sys.stdout.write(f"\033[{N_LINES + 2}B\n")
                sys.stdout.flush()
                print("\n" + "="*60)
                print("  ⚠️  ¡FALLA DETECTADA!")
                print(f"  Fecha : {fecha}  Hora : {hora}")
                print(f"  VA={V_mag[0]:.2f}V∠{V_ang[0]:+.1f}°  "
                      f"VB={V_mag[1]:.2f}V∠{V_ang[1]:+.1f}°  "
                      f"VC={V_mag[2]:.2f}V∠{V_ang[2]:+.1f}°")
                print(f"  IA={I_mag[0]:.2f}A∠{I_ang[0]:+.1f}°  "
                      f"IB={I_mag[1]:.2f}A∠{I_ang[1]:+.1f}°  "
                      f"IC={I_mag[2]:.2f}A∠{I_ang[2]:+.1f}°")
                print("  Registro guardado en Results/registers.txt")
                print("  Escriba 'Falla despejada' para reanudar la medición.")
                print("="*60)

                # Esperar confirmación del operador
                fault_cleared.wait()
                fault_cleared.clear()

                # ── FALLA DESPEJADA ───────────────────────────────────────────
                GPIO.output(GPIO_PIN, GPIO.LOW)
                in_fault   = False
                total_samp = 0
                t_start    = time.time()
                for ch in CHANNELS:
                    buffers[ch].clear()
                print("\n[OK] Medición reiniciada.\n")
                time.sleep(0.5)
                continue

        elapsed = time.time() - t_start
        print_display(V_mag, V_ang, I_mag, I_ang, elapsed, ready, fault=in_fault)

except KeyboardInterrupt:
    sys.stdout.write(f"\033[{N_LINES + 2}B\n")
    sys.stdout.flush()
    print("\n[INFO] Detenido por el usuario.")

finally:
    try:
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup()
    except:
        pass
    try:
        ADC.ADS1263_Exit()
    except:
        pass
