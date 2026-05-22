"""
fault_detection.py — Detección de fallas con ADS1263 + modelo IA (solo magnitudes)

Función:
  - Lee VA, VB, VC, IA, IB, IC desde ADS1263.
  - Calcula magnitudes reales usando RMS de componente AC.
  - Construye el vector de entrada para el modelo de detección:
      VA_mag, VB_mag, VC_mag,
      IA_mag, IB_mag, IC_mag
  - Carga el modelo y scaler desde ../Pruebas.
  - Si detecta falla:
      1. Detiene la medición.
      2. Muestra alerta.
      3. Activa GPIO físico 40 en alto.
      4. Registra el evento en ../Pruebas/fallas.txt.
      5. Espera a que se escriba: falla despejada.
      6. Apaga GPIO 40.
      7. Termina el proceso.

Uso:
  sudo python3 fault_detection.py
"""

import sys
import os
import time
import datetime
import numpy as np
from collections import deque

import joblib
import Jetson.GPIO as GPIO

# ── Importar librería ADS1263 ─────────────────────────────────────────────────
sys.path.insert(0, "/home/santiago/PF_Smart-Relay/SPI/High-Pricision_AD_HAT/python")
import ADS1263


# =============================================================================
# RUTAS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRUEBAS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Pruebas"))
MODEL_PATH = os.path.join(PRUEBAS_DIR, "magnitud.pkl")
SCALER_PATH = os.path.join(PRUEBAS_DIR, "scaler_magnitud.pkl")   # asumimos mismo nombre
FALLAS_FILE = os.path.join(PRUEBAS_DIR, "fallas.txt")

os.makedirs(PRUEBAS_DIR, exist_ok=True)


# =============================================================================
# CONFIGURACIÓN DEL ADC
# =============================================================================

REF = 5.08
RATE_STR = "ADS1263_38400SPS"


# =============================================================================
# FACTORES DE CONVERSIÓN CALIBRADOS POR CANAL
# =============================================================================

SCALE = {
    0: 525.4606,   # VA
    1: 572.9528,   # VB
    2: 538.9874,   # VC
    5: 29.4007,    # IA
    6: 30.2130,    # IB
    7: 30.5864,    # IC
}


# =============================================================================
# CANALES
# =============================================================================

CHANNELS = [0, 1, 2, 5, 6, 7]

LABELS = {
    0: "VA",
    1: "VB",
    2: "VC",
    5: "IA",
    6: "IB",
    7: "IC",
}

V_CHNLS = [0, 1, 2]
I_CHNLS = [5, 6, 7]
PHASES = ["A", "B", "C"]


# =============================================================================
# VENTANA Y DISPLAY
# =============================================================================

WINDOW_SIZE = 50
DISPLAY_INTERVAL = 0.5


# =============================================================================
# UMBRALES DE RUIDO
# =============================================================================

ENABLE_ZERO_CLAMP = True

ZERO_RMS_ADC_THRESHOLD = {
    0: 0.0200,   # VA
    1: 0.0200,   # VB
    2: 0.0200,   # VC
    5: 0.0050,   # IA
    6: 0.0050,   # IB
    7: 0.0050,   # IC
}


# =============================================================================
# GPIO
# =============================================================================

GPIO_PIN = 40  # pin físico 40

GPIO.setmode(GPIO.BOARD)
GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)


# =============================================================================
# HORA BOGOTÁ
# =============================================================================

BOGOTA_TZ = datetime.timezone(datetime.timedelta(hours=-5), name="America/Bogota")


# =============================================================================
# CARGA DEL MODELO
# =============================================================================

def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"No se encontró el scaler: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# =============================================================================
# INICIALIZACIÓN DEL ADC
# =============================================================================

ADC = ADS1263.ADS1263()

if ADC.ADS1263_init_ADC1(RATE_STR) == -1:
    print("[ERROR] No se pudo inicializar el ADS1263.")
    GPIO.output(GPIO_PIN, GPIO.LOW)
    GPIO.cleanup()
    sys.exit(1)

ADC.ADS1263_SetMode(0)


# =============================================================================
# CONVERSIÓN RAW ADC → VOLTIOS
# =============================================================================

def raw_to_voltage(raw: int) -> float:
    """
    Convierte el valor raw del ADS1263 a voltios en la entrada del ADC.
    """

    if raw >> 31:
        return -(REF * 2 - raw / 2_147_483_648.0 * REF)

    return raw / 2_147_483_647.0 * REF


# =============================================================================
# MAGNITUD
# =============================================================================

def compute_ac_rms_adc(samples):
    """
    Calcula RMS de componente AC en voltios ADC.

    x_ac = samples - mean(samples)
    rms_adc_ac = sqrt(mean(x_ac^2))
    """

    samples = np.asarray(samples, dtype=float)

    if len(samples) == 0:
        return 0.0

    x_ac = samples - np.mean(samples)
    rms_adc_ac = np.sqrt(np.mean(x_ac ** 2))

    if rms_adc_ac < 1e-12:
        rms_adc_ac = 0.0

    return float(rms_adc_ac)


def compute_ac_rms_value(samples, scale, ch):
    """
    Magnitud real usando RMS de componente AC.
    """

    rms_adc_ac = compute_ac_rms_adc(samples)

    if ENABLE_ZERO_CLAMP:
        if rms_adc_ac < ZERO_RMS_ADC_THRESHOLD[ch]:
            return 0.0

    return rms_adc_ac * scale


# =============================================================================
# VECTOR DE FEATURES PARA EL MODELO (SOLO MAGNITUDES)
# =============================================================================

def construir_features(results):
    """
    Orden del modelo:

      VA_mag, VB_mag, VC_mag,
      IA_mag, IB_mag, IC_mag
    """

    features = np.array([[
        results[0],
        results[1],
        results[2],
        results[5],
        results[6],
        results[7],
    ]], dtype=float)

    return features


def es_falla(pred):
    """
    Interpreta la salida del modelo.

    Se asume:
      0 = normal
      1 = falla
    """

    try:
        return int(pred) == 1
    except Exception:
        return str(pred).strip().lower() in ["1", "falla", "fault", "true"]


# =============================================================================
# REGISTRO DE FALLAS
# =============================================================================

def registrar_falla(results, pred):
    """
    Guarda el evento en ../Pruebas/fallas.txt.
    """

    now = datetime.datetime.now(BOGOTA_TZ)
    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M:%S")

    linea = (
        f"{fecha} | {hora} | TZ=America/Bogota | pred={pred} | "
        f"VA={results[0]:.4f} V  "
        f"VB={results[1]:.4f} V  "
        f"VC={results[2]:.4f} V  | "
        f"IA={results[5]:.4f} A  "
        f"IB={results[6]:.4f} A  "
        f"IC={results[7]:.4f} A\n"
    )

    with open(FALLAS_FILE, "a", encoding="utf-8") as f:
        f.write(linea)

    return fecha, hora


# =============================================================================
# FORMATOS
# =============================================================================

def fmt_voltage(value):
    return f"{value:>9.4f}"


def fmt_current(value):
    return f"{value:>7.4f}"


# =============================================================================
# DISPLAY
# =============================================================================

def print_display(results, elapsed, ready, fs_ch, pred_actual):
    lines = []

    lines.append("╔════════════════════════════════════════════════════════════╗")
    lines.append(f"║      ADS1263 — Detección IA · Solo magnitudes             ║")
    lines.append("╠═══════════╦══════════════════╦══════════════════╣")
    lines.append("║  Fase     ║  Tensión         ║  Corriente       ║")
    lines.append("╠═══════════╬══════════════════╬══════════════════╣")

    for i, ph in enumerate(PHASES):
        vch = V_CHNLS[i]
        ich = I_CHNLS[i]

        if ready:
            v_str = f"{fmt_voltage(results[vch])} V"
            i_str = f"{fmt_current(results[ich])} A"
        else:
            v_str = " acumulando..."
            i_str = " acumulando..."

        lines.append(f"║  Fase {ph}   ║  {v_str:<16}║  {i_str:<16}║")

    lines.append("╚═══════════╩══════════════════╩══════════════════╝")
    lines.append(f"  t={elapsed:>6.1f}s | ventana={WINDOW_SIZE} muestras | fs_ch≈{fs_ch:>8.2f} Sa/s")
    lines.append(f"  Modelo: magnitud.pkl | pred_actual={pred_actual}")
    lines.append("  Ctrl+C para detener manualmente.")

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def mostrar_alerta_falla(fecha, hora, results, pred):
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    print("╔" + "═" * 72 + "╗")
    print("║" + " " * 25 + "FALLA DETECTADA" + " " * 31 + "║")
    print("╠" + "═" * 72 + "╣")
    print(f"║ Fecha   : {fecha:<61}║")
    print(f"║ Hora    : {hora} Bogotá, Colombia{' ' * 37}║")
    print(f"║ Pred    : {str(pred):<61}║")
    print("╠" + "═" * 72 + "╣")
    print(f"║ VA = {results[0]:>9.4f} V{' ' * 53}║")
    print(f"║ VB = {results[1]:>9.4f} V{' ' * 53}║")
    print(f"║ VC = {results[2]:>9.4f} V{' ' * 53}║")
    print(f"║ IA = {results[5]:>9.4f} A{' ' * 53}║")
    print(f"║ IB = {results[6]:>9.4f} A{' ' * 53}║")
    print(f"║ IC = {results[7]:>9.4f} A{' ' * 53}║")
    print("╠" + "═" * 72 + "╣")
    print(f"║ Señal GPIO física 40: ALTA{' ' * 45}║")
    print(f"║ Registro guardado en: ../Pruebas/fallas.txt{' ' * 29}║")
    print("║ Para finalizar escriba: falla despejada" + " " * 30 + "║")
    print("╚" + "═" * 72 + "╝")


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        model, scaler = cargar_modelo()
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar los modelos: {e}")
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup()
        try:
            ADC.ADS1263_Exit()
        except Exception:
            pass
        sys.exit(1)

    print(f"\n[OK] Modelo de detección cargado: {MODEL_PATH}")
    print(f"[OK] Registro de fallas: {FALLAS_FILE}")
    time.sleep(1.0)

    buffer_raw = {ch: deque(maxlen=WINDOW_SIZE) for ch in CHANNELS}
    time_buffer = deque(maxlen=WINDOW_SIZE)

    results = {ch: 0.0 for ch in CHANNELS}

    t_start = time.time()
    last_display = 0.0
    total_samp = 0

    fs_ch = 1.0
    pred_actual = "-"

    while True:
        # ---------------------------------------------------------------------
        # 1) Adquisición
        # ---------------------------------------------------------------------
        now = time.time()
        time_buffer.append(now)

        for ch in CHANNELS:
            raw = ADC.ADS1263_GetChannalValue(ch)
            v_adc = raw_to_voltage(raw)
            buffer_raw[ch].append(v_adc)

        total_samp += 1
        ready = total_samp >= WINDOW_SIZE

        # ---------------------------------------------------------------------
        # 2) Cálculo + inferencia + display
        # ---------------------------------------------------------------------
        now_display = time.time()

        if now_display - last_display >= DISPLAY_INTERVAL:
            last_display = now_display

            if ready:
                if len(time_buffer) >= 2:
                    dt_window = time_buffer[-1] - time_buffer[0]

                    if dt_window > 0:
                        fs_ch = (len(time_buffer) - 1) / dt_window
                    else:
                        fs_ch = 1.0

                for ch in CHANNELS:
                    results[ch] = compute_ac_rms_value(
                        samples=buffer_raw[ch],
                        scale=SCALE[ch],
                        ch=ch
                    )

                features = construir_features(results)

                try:
                    features_scaled = scaler.transform(features)
                    pred = model.predict(features_scaled)[0]
                    pred_actual = pred
                except Exception as e:
                    pred = 0
                    pred_actual = f"ERROR_MODELO: {e}"

                if es_falla(pred):
                    # ---------------------------------------------------------
                    # FALLA DETECTADA
                    # ---------------------------------------------------------
                    GPIO.output(GPIO_PIN, GPIO.HIGH)

                    fecha, hora = registrar_falla(
                        results=results,
                        pred=pred
                    )

                    mostrar_alerta_falla(
                        fecha=fecha,
                        hora=hora,
                        results=results,
                        pred=pred
                    )

                    # Detener medición hasta que el operador escriba falla despejada.
                    while True:
                        cmd = input("\nDigite 'falla despejada' para finalizar el proceso: ").strip().lower()

                        if cmd == "falla despejada":
                            break

                        print("Comando no reconocido. Debe escribir exactamente: falla despejada")

                    # ---------------------------------------------------------
                    # FINALIZAR PROCESO
                    # ---------------------------------------------------------
                    GPIO.output(GPIO_PIN, GPIO.LOW)

                    print("\n[OK] Falla despejada.")
                    print("[INFO] Alarma GPIO apagada.")
                    print("[INFO] Finalizando proceso de detección...\n")

                    sys.exit(0)

            elapsed = time.time() - t_start

            print_display(
                results=results,
                elapsed=elapsed,
                ready=ready,
                fs_ch=fs_ch,
                pred_actual=pred_actual
            )


# =============================================================================
# EJECUCIÓN
# =============================================================================

try:
    main()

except KeyboardInterrupt:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    print("\n[INFO] Detenido por el usuario.")

finally:
    try:
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup()
    except Exception:
        pass

    try:
        ADC.ADS1263_Exit()
    except Exception:
        pass