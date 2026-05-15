"""
read_adc_phasors.py — Medición trifásica con RMS y fasores
Hardware : Jetson Nano + Waveshare High-Precision AD HAT (ADS1263)
Sistema  : 60 Hz, referencia VA = 0°

Relaciones de transformación:
  Tensión  → 1 V_relay  = 50 V_real
  Corriente→ 1 V_relay  = 30 A_real

Canales ADC:
  IN0=VA  IN1=VB  IN2=VC  IN5=IA  IN6=IB  IN7=IC

Uso:
  sudo python3 read_adc_phasors.py
"""

import sys
import time
import numpy as np
from collections import deque

# ── Importar librería ADS1263 ─────────────────────────────────────────────────
sys.path.insert(0, '/home/santiago/PF_Smart-Relay/SPI/High-Pricision_AD_HAT/python')
import ADS1263

# ── Configuración del sistema ─────────────────────────────────────────────────
REF        = 5.08          # Tensión de referencia ADC (medir entre AVDD-AVSS)
FREQ       = 60.0          # Frecuencia del sistema eléctrico [Hz]
RATE_STR   = 'ADS1263_38400SPS'

# Ganancia de los transformadores (factor real / V_relay)
KV         = 50.0          # 1 V_relay → 50 V reales
KI         = 30.0          # 1 V_relay → 30 A reales

# Canales y etiquetas
CHANNELS   = [0, 1, 2, 5, 6, 7]
LABELS     = {0: "VA", 1: "VB", 2: "VC", 5: "IA", 6: "IB", 7: "IC"}
IS_VOLTAGE = {0: True,  1: True,  2: True,
              5: False, 6: False, 7: False}

# Muestras por ventana de cálculo
# A ~185 Sa/s por canal (38400 / 6 canales) → ~107 muestras/ciclo
# Usamos 2 ciclos para tener ~214 muestras por canal → buena resolución espectral
CYCLES_WINDOW  = 2
FS_PER_CHANNEL = 38400 / len(CHANNELS)          # muestras/s estimadas por canal
WINDOW_SIZE    = int(round(FS_PER_CHANNEL * CYCLES_WINDOW / FREQ))

# ── Inicialización del ADC ─────────────────────────────────────────────────────
ADC = ADS1263.ADS1263()
if ADC.ADS1263_init_ADC1(RATE_STR) == -1:
    print("[ERROR] No se pudo inicializar el ADS1263.")
    sys.exit(1)
ADC.ADS1263_SetMode(0)   # Single-ended

# ── Buffers circulares ────────────────────────────────────────────────────────
buffers = {ch: deque(maxlen=WINDOW_SIZE) for ch in CHANNELS}

# ── Conversión raw → voltios ──────────────────────────────────────────────────
def raw_to_voltage(raw: int) -> float:
    if raw >> 31:
        return -(REF * 2 - raw / 2_147_483_648.0 * REF)
    return raw / 2_147_483_647.0 * REF

# ── Cálculo de RMS y fasor por DFT en la fundamental ─────────────────────────
def compute_rms_phasor(samples: np.ndarray, fs: float, freq: float):
    """
    Devuelve (rms, angle_deg) calculados sobre la componente fundamental.
    - rms      : valor eficaz real (ya aplicado KV o KI externamente)
    - angle_deg: ángulo de la fundamental en grados
    """
    n   = len(samples)
    # Frecuencia fundamental más cercana en la DFT
    freqs = np.fft.rfftfreq(n, d=1/fs)
    k = np.argmin(np.abs(freqs - freq))
    dft = np.fft.rfft(samples)
    # Magnitud pico → RMS = magnitud_pico / sqrt(2)
    mag_rms = (2 * np.abs(dft[k]) / n) / np.sqrt(2)
    angle   = np.degrees(np.angle(dft[k]))
    return mag_rms, angle

# ── Display ───────────────────────────────────────────────────────────────────
PHASES  = ['A', 'B', 'C']
V_CHNLS = [0, 1, 2]
I_CHNLS = [5, 6, 7]

def fmt_ang(a: float) -> str:
    return f"{a:+.1f}°"

def print_display(results: dict, elapsed: float, n_samples: int, ready: bool):
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║     ADS1263 — Sistema Trifásico · 60 Hz · 🟢 Monitoreo      ║")
    lines.append("╠═══════════╦═══════════════════════╦═════════════════════════╣")
    lines.append("║  Fase     ║  Tensión               ║  Corriente              ║")
    lines.append("╠═══════════╬═══════════════════════╬═════════════════════════╣")

    for i, ph in enumerate(PHASES):
        vch = V_CHNLS[i]
        ich = I_CHNLS[i]
        if ready:
            v_rms, v_ang = results[vch]
            i_rms, i_ang = results[ich]
            v_str = f"{v_rms:>7.2f} V  ∠{fmt_ang(v_ang):>8}"
            i_str = f"{i_rms:>7.2f} A  ∠{fmt_ang(i_ang):>8}"
        else:
            v_str = "   acumulando...      "
            i_str = "   acumulando...      "
        lines.append(f"║  Fase {ph}   ║  {v_str}  ║  {i_str}  ║")

    lines.append("╚═══════════╩═══════════════════════╩═════════════════════════╝")
    lines.append(f"  t={elapsed:>6.1f}s | ventana={CYCLES_WINDOW} ciclos | {n_samples} muestras/canal")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"\033[{len(lines)}A")
    sys.stdout.flush()

# ── Bucle principal ───────────────────────────────────────────────────────────
print(f"\nVentana de cálculo: {WINDOW_SIZE} muestras/canal  ({CYCLES_WINDOW} ciclos @ ~{FS_PER_CHANNEL:.0f} Sa/s)\n")
time.sleep(0.3)

t_start    = time.time()
total_samp = 0
results    = {ch: (0.0, 0.0) for ch in CHANNELS}
N_LINES    = 12

try:
    while True:
        # Adquirir una muestra de cada canal
        for ch in CHANNELS:
            raw  = ADC.ADS1263_GetChannalValue(ch)
            v    = raw_to_voltage(raw)
            buffers[ch].append(v)

        total_samp += 1
        ready       = (total_samp >= WINDOW_SIZE)

        # Calcular fasores cuando el buffer está lleno
        if ready:
            fs_est  = total_samp / (time.time() - t_start)   # FS real medido
            fs_ch   = fs_est / len(CHANNELS)

            raw_results = {}
            for ch in CHANNELS:
                arr          = np.array(buffers[ch])
                rms, ang     = compute_rms_phasor(arr, fs_ch, FREQ)
                scale        = KV if IS_VOLTAGE[ch] else KI
                raw_results[ch] = (rms * scale, ang)

            # Referenciar todos los ángulos a VA = 0°
            ref_ang = raw_results[0][1]
            for ch in CHANNELS:
                rms, ang         = raw_results[ch]
                results[ch]      = (rms, ang - ref_ang)

        elapsed = time.time() - t_start
        print_display(results, elapsed, len(buffers[CHANNELS[0]]), ready)

except KeyboardInterrupt:
    sys.stdout.write(f"\033[{N_LINES + 2}B\n")
    sys.stdout.flush()
    print("\n[INFO] Detenido.")

finally:
    try:
        ADC.ADS1263_Exit()
    except AttributeError:
        pass
