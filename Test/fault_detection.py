"""
fault_detection.py — Detección de fallas con ADS1263 + modelo IA

Función:
  - Lee VA, VB, VC, IA, IB, IC desde ADS1263.
  - Calcula magnitudes reales usando RMS de componente AC.
  - Calcula ángulos por FFT cerca de 60 Hz.
  - Usa VA como referencia angular de 0°.
  - Invierte el signo de los ángulos, igual que en read_adc_phasor.py.
  - Construye el vector de entrada para el modelo de detección:

      VA_mag, VB_mag, VC_mag,
      VA_ang, VB_ang, VC_ang,
      IA_mag, IB_mag, IC_mag,
      IA_ang, IB_ang, IC_ang

  - Pregunta si el sistema es IEEE 5 o IEEE 13.
  - Carga el modelo y scaler correspondientes.
  - Si detecta falla:
      1. Detiene la medición.
      2. Muestra alerta.
      3. Activa GPIO físico 40 en alto.
      4. Registra el evento en Results/fallas.txt.
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
MODELS_DIR = os.path.join(BASE_DIR, "Models")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
FALLAS_FILE = os.path.join(RESULTS_DIR, "fallas.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATHS = {
    "ieee5": {
        "model": os.path.join(MODELS_DIR, "model_det_ieee5.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler_det_ieee5.pkl"),
    },
    "ieee13": {
        "model": os.path.join(MODELS_DIR, "model_det_ieee13.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler_det_ieee13.pkl"),
    },
}


# =============================================================================
# CONFIGURACIÓN DEL ADC
# =============================================================================

REF = 5.08
RATE_STR = "ADS1263_38400SPS"
FREQ = 60.0


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
# CORRECCIONES ANGULARES
# =============================================================================

ANGLE_CORR = {
    0: 0.0,   # VA
    1: 0.0,   # VB
    2: 0.0,   # VC
    5: 0.0,   # IA
    6: 0.0,   # IB
    7: 0.0,   # IC
}


# =============================================================================
# INVERSIÓN DE ÁNGULO
# =============================================================================

ANGLE_SIGN = -1.0


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
# PARÁMETROS FFT
# =============================================================================

SEARCH_BAND_HZ = 5.0
MIN_FFT_MAG = 1e-12


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
# SELECCIÓN DEL SISTEMA
# =============================================================================

def seleccionar_sistema():
    while True:
        print("\nSeleccione el sistema:")
        print("  [1] IEEE 5")
        print("  [2] IEEE 13")
        opcion = input("Opción: ").strip()

        if opcion == "1":
            return "ieee5"

        if opcion == "2":
            return "ieee13"

        print("[ERROR] Opción inválida. Escriba 1 o 2.")


def cargar_modelos(sistema):
    paths = MODEL_PATHS[sistema]

    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(f"No se encontró el modelo: {paths['model']}")

    if not os.path.exists(paths["scaler"]):
        raise FileNotFoundError(f"No se encontró el scaler: {paths['scaler']}")

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])

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
# ÁNGULO / FFT
# =============================================================================

def normalize_angle(angle):
    """
    Normaliza un ángulo al rango [-180, 180].
    """

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def true_fft_phase(channel_samples):
    """
    Calcula FFT de la componente AC.
    """

    x = np.asarray(channel_samples, dtype=float)

    if len(x) == 0:
        return np.array([0.0])

    x_ac = x - np.mean(x)

    if len(x_ac) > 1:
        window = np.hanning(len(x_ac))
        x_ac = x_ac * window

    return np.fft.rfft(x_ac)


def data_angle(fft_ref, fft_ch, k):
    """
    Ángulo relativo respecto a VA.
    """

    angle_ref = np.degrees(np.angle(fft_ref[k]))
    angle_ch = np.degrees(np.angle(fft_ch[k]))

    return normalize_angle(angle_ch - angle_ref)


def find_fft_bin_near_60hz(fft_va, n_samples, fs):
    """
    Busca el bin cercano a 60 Hz.
    """

    if fs <= 0 or n_samples <= 1:
        return 0, 0.0, "ADVERTENCIA: fs inválida."

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    if len(freqs) == 0:
        return 0, 0.0, "ADVERTENCIA: no hay bins FFT."

    f_min = FREQ - SEARCH_BAND_HZ
    f_max = FREQ + SEARCH_BAND_HZ

    idx_band = np.where((freqs >= f_min) & (freqs <= f_max))[0]

    if len(idx_band) > 0:
        mags_va = np.abs(fft_va[idx_band])
        k = int(idx_band[np.argmax(mags_va)])
        freq_used = float(freqs[k])
        return k, freq_used, ""

    k = int(np.argmin(np.abs(freqs - FREQ)))
    freq_used = float(freqs[k])

    return k, freq_used, "ADVERTENCIA: fs_ch baja; FFT no alcanza 60 Hz."


def compute_fft_angles(raw_buffers, fs, rms_ac_adc):
    """
    Calcula ángulos usando FFT y VA como referencia.

    Los ángulos se guardan con signo invertido.
    """

    n = len(raw_buffers[0])

    FFT = {}
    for ch in CHANNELS:
        FFT[ch] = true_fft_phase(raw_buffers[ch])

    k, freq_used, warning = find_fft_bin_near_60hz(
        fft_va=FFT[0],
        n_samples=n,
        fs=fs
    )

    angles = {}
    valid = {}

    mag_ref = float(np.abs(FFT[0][k])) if k < len(FFT[0]) else 0.0

    ref_has_signal = True
    if ENABLE_ZERO_CLAMP:
        ref_has_signal = rms_ac_adc[0] >= ZERO_RMS_ADC_THRESHOLD[0]

    ref_valid = (mag_ref >= MIN_FFT_MAG) and ref_has_signal

    for ch in CHANNELS:
        mag_ch = float(np.abs(FFT[ch][k])) if k < len(FFT[ch]) else 0.0

        has_signal = True
        if ENABLE_ZERO_CLAMP:
            has_signal = rms_ac_adc[ch] >= ZERO_RMS_ADC_THRESHOLD[ch]

        if ref_valid and mag_ch >= MIN_FFT_MAG and has_signal:
            raw_ang = data_angle(FFT[0], FFT[ch], k)

            final_ang = ANGLE_SIGN * normalize_angle(raw_ang + ANGLE_CORR[ch])
            final_ang = normalize_angle(final_ang)

            angles[ch] = final_ang
            valid[ch] = True
        else:
            angles[ch] = 0.0
            valid[ch] = False

    if valid[0]:
        angles[0] = 0.0

    return angles, valid, freq_used, warning


# =============================================================================
# VECTOR DE FEATURES PARA EL MODELO
# =============================================================================

def construir_features(results, angles):
    """
    Orden estricto de entrada del modelo:

      VA_mag, VB_mag, VC_mag,
      VA_ang, VB_ang, VC_ang,
      IA_mag, IB_mag, IC_mag,
      IA_ang, IB_ang, IC_ang
    """

    VA = results[0]
    VB = results[1]
    VC = results[2]

    IA = results[5]
    IB = results[6]
    IC = results[7]

    VAP = angles[0]
    VBP = angles[1]
    VCP = angles[2]

    IAP = angles[5]
    IBP = angles[6]
    ICP = angles[7]

    features = np.array([[
        VA, VB, VC,
        VAP, VBP, VCP,
        IA, IB, IC,
        IAP, IBP, ICP,
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

def registrar_falla(sistema, results, angles, pred):
    """
    Guarda el evento en Results/fallas.txt.
    """

    now = datetime.datetime.now(BOGOTA_TZ)
    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M:%S")

    linea = (
        f"{fecha} | {hora} | TZ=America/Bogota | sistema={sistema.upper()} | pred={pred} | "
        f"VA={results[0]:.4f}V∠{angles[0]:+.2f}° "
        f"VB={results[1]:.4f}V∠{angles[1]:+.2f}° "
        f"VC={results[2]:.4f}V∠{angles[2]:+.2f}° | "
        f"IA={results[5]:.4f}A∠{angles[5]:+.2f}° "
        f"IB={results[6]:.4f}A∠{angles[6]:+.2f}° "
        f"IC={results[7]:.4f}A∠{angles[7]:+.2f}°\n"
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


def fmt_angle(angle, valid=True):
    if not valid:
        return "---.--°"

    angle = normalize_angle(angle)

    return f"{angle:+07.2f}°"


# =============================================================================
# DISPLAY
# =============================================================================

def print_display(results, angles, valid, elapsed, ready, fs_ch, freq_used, fft_warning, sistema, pred_actual):
    lines = []

    lines.append("╔════════════════════════════════════════════════════════════════════════════╗")
    lines.append(f"║      ADS1263 — Detección IA · {sistema.upper()} · Estado: NORMAL                 ║")
    lines.append("╠═══════════╦══════════════════════════════╦═══════════════════════════════╣")
    lines.append("║  Fase     ║  Tensión generador            ║  Corriente generador          ║")
    lines.append("╠═══════════╬══════════════════════════════╬═══════════════════════════════╣")

    for i, ph in enumerate(PHASES):
        vch = V_CHNLS[i]
        ich = I_CHNLS[i]

        if ready:
            v_str = f"{fmt_voltage(results[vch])} V ∠ {fmt_angle(angles[vch], valid[vch])}"
            i_str = f"{fmt_current(results[ich])} A ∠ {fmt_angle(angles[ich], valid[ich])}"
        else:
            v_str = " acumulando..."
            i_str = " acumulando..."

        lines.append(f"║  Fase {ph}   ║  {v_str:<28}║  {i_str:<29}║")

    lines.append("╚═══════════╩══════════════════════════════╩═══════════════════════════════╝")
    lines.append(f"  t={elapsed:>6.1f}s | ventana={WINDOW_SIZE} muestras | fs_ch≈{fs_ch:>8.2f} Sa/s")
    lines.append(f"  Modelo: {sistema.upper()} | pred_actual={pred_actual}")
    lines.append(f"  FFT: VA=0° | ángulos con signo invertido | f≈{freq_used:>6.2f} Hz")

    if fft_warning:
        lines.append(f"  {fft_warning}")
    else:
        lines.append("  FFT OK: resolución cercana a 60 Hz.")

    lines.append("  Ctrl+C para detener manualmente.")

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def mostrar_alerta_falla(sistema, fecha, hora, results, angles, pred):
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    print("╔" + "═" * 72 + "╗")
    print("║" + " " * 25 + "FALLA DETECTADA" + " " * 31 + "║")
    print("╠" + "═" * 72 + "╣")
    print(f"║ Sistema : {sistema.upper():<61}║")
    print(f"║ Fecha   : {fecha:<61}║")
    print(f"║ Hora    : {hora} Bogotá, Colombia{' ' * 37}║")
    print(f"║ Pred    : {str(pred):<61}║")
    print("╠" + "═" * 72 + "╣")
    print(f"║ VA = {results[0]:>9.4f} V ∠ {angles[0]:+7.2f}°{' ' * 39}║")
    print(f"║ VB = {results[1]:>9.4f} V ∠ {angles[1]:+7.2f}°{' ' * 39}║")
    print(f"║ VC = {results[2]:>9.4f} V ∠ {angles[2]:+7.2f}°{' ' * 39}║")
    print(f"║ IA = {results[5]:>9.4f} A ∠ {angles[5]:+7.2f}°{' ' * 39}║")
    print(f"║ IB = {results[6]:>9.4f} A ∠ {angles[6]:+7.2f}°{' ' * 39}║")
    print(f"║ IC = {results[7]:>9.4f} A ∠ {angles[7]:+7.2f}°{' ' * 39}║")
    print("╠" + "═" * 72 + "╣")
    print(f"║ Señal GPIO física 40: ALTA{' ' * 45}║")
    print(f"║ Registro guardado en: Results/fallas.txt{' ' * 32}║")
    print("║ Para finalizar escriba: falla despejada" + " " * 30 + "║")
    print("╚" + "═" * 72 + "╝")


# =============================================================================
# MAIN
# =============================================================================

def main():
    sistema = seleccionar_sistema()

    try:
        model, scaler = cargar_modelos(sistema)
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar los modelos: {e}")
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup()
        try:
            ADC.ADS1263_Exit()
        except Exception:
            pass
        sys.exit(1)

    print(f"\n[OK] Modelo de detección cargado: {sistema.upper()}")
    print(f"[OK] Registro de fallas: {FALLAS_FILE}")
    time.sleep(1.0)

    buffer_raw = {ch: deque(maxlen=WINDOW_SIZE) for ch in CHANNELS}
    time_buffer = deque(maxlen=WINDOW_SIZE)

    results = {ch: 0.0 for ch in CHANNELS}
    angles = {ch: 0.0 for ch in CHANNELS}
    valid = {ch: False for ch in CHANNELS}
    rms_ac_adc = {ch: 0.0 for ch in CHANNELS}

    t_start = time.time()
    last_display = 0.0
    total_samp = 0

    fs_ch = 1.0
    freq_used = 0.0
    fft_warning = ""
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
                    rms_ac_adc[ch] = compute_ac_rms_adc(buffer_raw[ch])

                for ch in CHANNELS:
                    results[ch] = compute_ac_rms_value(
                        samples=buffer_raw[ch],
                        scale=SCALE[ch],
                        ch=ch
                    )

                angles, valid, freq_used, fft_warning = compute_fft_angles(
                    raw_buffers=buffer_raw,
                    fs=fs_ch,
                    rms_ac_adc=rms_ac_adc
                )

                features = construir_features(results, angles)

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
                        sistema=sistema,
                        results=results,
                        angles=angles,
                        pred=pred
                    )

                    mostrar_alerta_falla(
                        sistema=sistema,
                        fecha=fecha,
                        hora=hora,
                        results=results,
                        angles=angles,
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
                angles=angles,
                valid=valid,
                elapsed=elapsed,
                ready=ready,
                fs_ch=fs_ch,
                freq_used=freq_used,
                fft_warning=fft_warning,
                sistema=sistema,
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
