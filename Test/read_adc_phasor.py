"""
read_adc_phasor.py — Lectura trifásica ADS1263 con magnitud AC-RMS + ángulo FFT

Objetivo:
  - Leer señales desde ADS1263.
  - Usar SIEMPRE las últimas 300 muestras registradas por canal.
  - Quitar offset DC de cada canal.
  - Calcular magnitud real usando RMS de la componente AC.
  - Calcular ángulo por FFT cerca de 60 Hz.
  - Usar VA como referencia angular de 0°.
  - Invertir el signo de los ángulos calculados para representarlos
    como secuencia positiva en las variables internas y en pantalla.

Canales:
  IN0 = VA
  IN1 = VB
  IN2 = VC
  IN5 = IA
  IN6 = IB
  IN7 = IC

Uso:
  sudo python3 read_adc_phasor.py
"""

import sys
import time
import numpy as np
from collections import deque

# ── Importar librería ADS1263 ─────────────────────────────────────────────────
sys.path.insert(0, '/home/santiago/PF_Smart-Relay/SPI/High-Pricision_AD_HAT/python')
import ADS1263


# =============================================================================
# CONFIGURACIÓN DEL ADC
# =============================================================================

REF = 5.08
RATE_STR = 'ADS1263_38400SPS'
FREQ = 60.0


# =============================================================================
# FACTORES DE CONVERSIÓN CALIBRADOS POR CANAL — NO MODIFICAR
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
# CORRECCIONES ANGULARES — TODAS EN CERO
# =============================================================================

ANGLE_CORR = {
    0: 0.0,   # VA
    1: 0.0,   # VB
    2: 0.0,   # VC
    5: 145.0,   # IA
    6: 145.0,   # IB
    7: 145.0,   # IC
}


# =============================================================================
# INVERSIÓN DE ÁNGULO
# =============================================================================
#
# Si el cálculo natural entrega orientación tipo secuencia negativa,
# se invierte el signo para guardar y mostrar los ángulos como secuencia positiva.
#
#   angulo_final = -angulo_calculado
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

WINDOW_SIZE = 300
DISPLAY_INTERVAL = 0.5


# =============================================================================
# PARÁMETROS FFT
# =============================================================================

SEARCH_BAND_HZ = 5.0
MIN_FFT_MAG = 1e-12


# =============================================================================
# UMBRALES DE RUIDO / GENERADOR APAGADO
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
# INICIALIZACIÓN DEL ADC
# =============================================================================

ADC = ADS1263.ADS1263()

if ADC.ADS1263_init_ADC1(RATE_STR) == -1:
    print("[ERROR] No se pudo inicializar el ADS1263.")
    sys.exit(1)

ADC.ADS1263_SetMode(0)   # Single-ended


# =============================================================================
# BUFFERS
# =============================================================================

buffer_raw = {ch: deque(maxlen=WINDOW_SIZE) for ch in CHANNELS}
time_buffer = deque(maxlen=WINDOW_SIZE)


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
# FUNCIONES DE MAGNITUD
# =============================================================================

def compute_ac_rms_adc(samples):
    """
    Calcula el RMS de la componente AC en voltios ADC.

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
    Calcula magnitud real usando RMS de componente AC.
    """

    rms_adc_ac = compute_ac_rms_adc(samples)

    if ENABLE_ZERO_CLAMP:
        if rms_adc_ac < ZERO_RMS_ADC_THRESHOLD[ch]:
            return 0.0

    return rms_adc_ac * scale


def get_rms_ac(samples):
    """
    Devuelve RMS AC para validación de señal.
    """

    return compute_ac_rms_adc(samples)


# =============================================================================
# FUNCIONES DE ÁNGULO / FFT
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
    Calcula FFT de la componente AC de un canal.
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
    Calcula el ángulo relativo de un canal respecto a VA.

    Base:
      angle = angle_ch - angle_ref

    Luego se invierte globalmente con ANGLE_SIGN.
    """

    angle_ref = np.degrees(np.angle(fft_ref[k]))
    angle_ch = np.degrees(np.angle(fft_ch[k]))

    return normalize_angle(angle_ch - angle_ref)


def find_fft_bin_near_60hz(fft_va, n_samples, fs):
    """
    Busca el bin de FFT cercano a 60 Hz.
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

    Los ángulos se guardan ya invertidos por ANGLE_SIGN.
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
# FORMATOS
# =============================================================================

def fmt_voltage(value):
    """
    Voltaje: 4 enteros y 4 decimales.
    """
    return f"{value:>9.4f}"


def fmt_current(value):
    """
    Corriente: 2 enteros y 4 decimales.
    """
    return f"{value:>7.4f}"


def fmt_angle(angle, valid=True):
    """
    Ángulo: 3 enteros y 2 decimales.
    """

    if not valid:
        return "---.--°"

    angle = normalize_angle(angle)

    return f"{angle:+07.2f}°"


# =============================================================================
# DISPLAY
# =============================================================================

def print_display(results, angles, valid, elapsed, ready, fs_ch, freq_used, fft_warning):
    lines = []

    lines.append("╔════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║          ADS1263 — Magnitud AC-RMS + ángulos por FFT                     ║")
    lines.append("╠═══════════╦══════════════════════════════╦═══════════════════════════════╣")
    lines.append("║  Fase     ║  Tensión generador            ║  Corriente generador          ║")
    lines.append("╠═══════════╬══════════════════════════════╬═══════════════════════════════╣")

    for i, ph in enumerate(PHASES):
        vch = V_CHNLS[i]
        ich = I_CHNLS[i]

        if ready:
            v_real = results[vch]
            i_real = results[ich]

            v_ang = angles[vch]
            i_ang = angles[ich]

            v_ok = valid[vch]
            i_ok = valid[ich]

            v_str = f"{fmt_voltage(v_real)} V ∠ {fmt_angle(v_ang, v_ok)}"
            i_str = f"{fmt_current(i_real)} A ∠ {fmt_angle(i_ang, i_ok)}"
        else:
            v_str = " acumulando..."
            i_str = " acumulando..."

        lines.append(f"║  Fase {ph}   ║  {v_str:<28}║  {i_str:<29}║")

    lines.append("╚═══════════╩══════════════════════════════╩═══════════════════════════════╝")
    lines.append(f"  t={elapsed:>6.1f}s | ventana={WINDOW_SIZE} muestras | fs_ch≈{fs_ch:>8.2f} Sa/s")
    lines.append(
        f"  Escalas calibradas: "
        f"VA×{SCALE[0]:.2f}, VB×{SCALE[1]:.2f}, VC×{SCALE[2]:.2f} | "
        f"IA×{SCALE[5]:.2f}, IB×{SCALE[6]:.2f}, IC×{SCALE[7]:.2f}"
    )
    lines.append(f"  FFT: VA=0° | ángulos guardados con signo invertido | f≈{freq_used:>6.2f} Hz")

    if fft_warning:
        lines.append(f"  {fft_warning}")
    else:
        lines.append("  FFT OK: resolución cercana a 60 Hz.")

    if ENABLE_ZERO_CLAMP:
        lines.append("  Filtro ruido: activo. Señales bajo umbral se muestran como 0 y ángulo ---.--°.")
    else:
        lines.append("  Filtro ruido: desactivado.")

    # Limpiar pantalla completa y volver al inicio.
    # Esto evita residuos visuales en la terminal.
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()

    return len(lines)


# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================

print("\033[2J\033[H", end="")

print()
print("============================================================")
print(" Lectura ADS1263 con sensores AC/DC")
print(" Magnitud por AC-RMS + ángulos por FFT")
print("============================================================")
print(" VA se usa como referencia de 0°.")
print(" Los ángulos se guardan y muestran con signo invertido.")
print(f" Ventana móvil   : últimas {WINDOW_SIZE} muestras por canal")
print(f" Display         : cada {DISPLAY_INTERVAL:.2f} s")
print(" Escalas calibradas:")
for ch in CHANNELS:
    print(f"   {LABELS[ch]}: ×{SCALE[ch]:.4f} | corr_ángulo={ANGLE_CORR[ch]:+.2f}°")
print("============================================================")
print()

time.sleep(0.5)

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

try:
    while True:

        # ---------------------------------------------------------------------
        # 1) Leer todos los canales lo más rápido posible
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
        # 2) Calcular y actualizar display cada DISPLAY_INTERVAL
        # ---------------------------------------------------------------------
        now_display = time.time()

        if now_display - last_display >= DISPLAY_INTERVAL:
            last_display = now_display

            if ready:
                # Frecuencia de muestreo real estimada en la ventana actual.
                if len(time_buffer) >= 2:
                    dt_window = time_buffer[-1] - time_buffer[0]

                    if dt_window > 0:
                        fs_ch = (len(time_buffer) - 1) / dt_window
                    else:
                        fs_ch = 1.0

                # RMS AC ADC por canal.
                for ch in CHANNELS:
                    rms_ac_adc[ch] = get_rms_ac(buffer_raw[ch])

                # Magnitudes reales usando RMS de componente AC.
                for ch in CHANNELS:
                    results[ch] = compute_ac_rms_value(
                        samples=buffer_raw[ch],
                        scale=SCALE[ch],
                        ch=ch
                    )

                # Ángulos por FFT usando VA como referencia.
                # Aquí ya quedan guardados con signo invertido.
                angles, valid, freq_used, fft_warning = compute_fft_angles(
                    raw_buffers=buffer_raw,
                    fs=fs_ch,
                    rms_ac_adc=rms_ac_adc
                )

            elapsed = time.time() - t_start

            print_display(
                results=results,
                angles=angles,
                valid=valid,
                elapsed=elapsed,
                ready=ready,
                fs_ch=fs_ch,
                freq_used=freq_used,
                fft_warning=fft_warning
            )


except KeyboardInterrupt:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    print("\n[INFO] Detenido por el usuario.")


finally:
    try:
        ADC.ADS1263_Exit()
    except AttributeError:
        pass
