"""
read_adc_rms.py — Lectura trifásica ADS1263 solo magnitud AC-RMS

Versión simplificada: solo mide y escala valores RMS (V y A).
No realiza análisis de ángulo ni correcciones de fase.
"""

import sys
import time
import os
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
FREQ = 60.0  # solo informativa, no se usa para FFT


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

V_CHNLS = [0, 1, 2]   # VA, VB, VC
I_CHNLS = [5, 6, 7]   # IA, IB, IC
PHASES = ["A", "B", "C"]


# =============================================================================
# VALORES OBJETIVO PARA CALIBRACIÓN DE MAGNITUD
# =============================================================================

DEFAULT_TARGET_RMS = {
    0: 125.50,   # VA
    1: 125.40,   # VB
    2: 125.45,   # VC
    5: 0.464,    # IA
    6: 0.460,    # IB
    7: 0.479,    # IC
}


# =============================================================================
# FACTORES DE CONVERSIÓN POR DEFECTO
# =============================================================================

SCALE = {
    0: 570.513822,   # VA
    1: 558.371793,   # VB
    2: 543.843577,   # VC
    5: 28.308563,    # IA
    6: 29.052098,    # IB
    7: 30.749694,    # IC
}


# =============================================================================
# ARCHIVO DE CALIBRACIÓN (ahora solo SCALE)
# =============================================================================

CALIBRATION_FILE = "rms_calibration.txt"


# =============================================================================
# VENTANA Y DISPLAY
# =============================================================================

WINDOW_SIZE = 1000
DISPLAY_INTERVAL = 0.5
CALIBRATION_SAMPLES = 1000


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

ADC.ADS1263_SetMode(0)


# =============================================================================
# BUFFERS
# =============================================================================

buffer_raw = {ch: deque(maxlen=WINDOW_SIZE) for ch in CHANNELS}
time_buffer = deque(maxlen=WINDOW_SIZE)   # solo para estimar fs_ch


# =============================================================================
# UTILIDADES
# =============================================================================

def ask_float(prompt, default_value=None):
    """
    Entrada numérica segura.
    """
    while True:
        txt = input(prompt).strip()

        if txt == "" and default_value is not None:
            return float(default_value)

        try:
            return float(txt)
        except ValueError:
            print("  ❌ Ingresa un número válido.")


# =============================================================================
# CONVERSIÓN RAW ADC → VOLTIOS
# =============================================================================

def raw_to_voltage(raw: int) -> float:
    """
    Convierte lectura cruda del ADS1263 a voltios.
    """
    if raw >> 31:
        return -(REF * 2 - raw / 2_147_483_648.0 * REF)

    return raw / 2_147_483_647.0 * REF


# =============================================================================
# FUNCIONES DE MAGNITUD
# =============================================================================

def compute_ac_rms_adc(samples):
    """
    Calcula RMS AC eliminando offset DC.
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
    Calcula valor RMS real a partir del RMS ADC y el factor del canal.
    """
    rms_adc_ac = compute_ac_rms_adc(samples)

    if ENABLE_ZERO_CLAMP:
        if rms_adc_ac < ZERO_RMS_ADC_THRESHOLD[ch]:
            return 0.0

    return rms_adc_ac * scale


# =============================================================================
# CALIBRACIÓN (solo magnitud)
# =============================================================================

def acquire_calibration_buffers(adc, channels, num_samples=1000):
    """
    Toma una ventana de muestras para calibración.
    """
    buffers = {ch: [] for ch in channels}
    t_buf = []

    print(f"\nAdquiriendo {num_samples} muestras por canal...")

    for _ in range(num_samples):
        now = time.time()
        t_buf.append(now)

        for ch in channels:
            raw = adc.ADS1263_GetChannalValue(ch)
            buffers[ch].append(raw_to_voltage(raw))

    return buffers, t_buf


def get_target_rms_from_user():
    """
    Permite usar valores objetivo por defecto o ingresar valores reales medidos.
    """
    print("\nValores RMS objetivo por defecto:")
    for ch in CHANNELS:
        unit = "V" if ch in V_CHNLS else "A"
        print(f"  {LABELS[ch]}: {DEFAULT_TARGET_RMS[ch]} {unit}")

    resp = input("\n¿Usar estos valores objetivo para calibrar magnitud? (s/n): ").strip().lower()

    target = {}

    if resp == "s":
        target.update(DEFAULT_TARGET_RMS)
        return target

    print("\nIngresa los valores RMS reales medidos para esta condición de calibración.")
    print("Ejemplo: VA=125.5, IA=0.464, etc.\n")

    for ch in CHANNELS:
        unit = "V" if ch in V_CHNLS else "A"
        default = DEFAULT_TARGET_RMS[ch]
        target[ch] = ask_float(
            f"  {LABELS[ch]} real [{default} {unit}]: ",
            default_value=default
        )

    return target


def calibrate_magnitude(adc, channels, labels, num_samples=1000):
    """
    Calibración de magnitud: SCALE[ch] = RMS_real / RMS_ADC
    """
    global SCALE

    print("\n🔧 CALIBRACIÓN DE MAGNITUD AC-RMS")
    print("Asegúrate de que la fuente esté estable y con carga conocida.\n")

    target_rms = get_target_rms_from_user()

    buffers, _ = acquire_calibration_buffers(adc, channels, num_samples)

    print("\nResultados de calibración de magnitud:")

    for ch in channels:
        rms_adc = compute_ac_rms_adc(buffers[ch])
        real = target_rms[ch]

        if rms_adc > 0:
            SCALE[ch] = real / rms_adc
        else:
            SCALE[ch] = 1.0

        unit = "V" if ch in V_CHNLS else "A"

        print(
            f"  {labels[ch]} | "
            f"RMS_ADC={rms_adc:.6f} V_ADC | "
            f"Real={real:.6f} {unit} | "
            f"SCALE={SCALE[ch]:.9f}"
        )

    print("\n✅ Calibración terminada.\n")


def save_calibration(filename=CALIBRATION_FILE):
    """
    Guarda solo los factores SCALE.
    """
    with open(filename, "w") as f:
        f.write("# Archivo de calibración RMS\n")
        f.write("# SCALE canal valor\n")
        for ch in CHANNELS:
            f.write(f"SCALE {ch} {SCALE[ch]:.12f}\n")

    print(f"📁 Calibración guardada en {filename}")


def load_calibration(filename=CALIBRATION_FILE):
    """
    Carga factores SCALE desde archivo.
    """
    if not os.path.exists(filename):
        return False

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) < 3:
                    continue

                key = parts[0].upper()
                ch = int(parts[1])
                value = float(parts[2])

                if key == "SCALE" and ch in SCALE:
                    SCALE[ch] = value

        return True

    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo cargar {filename}: {e}")
        return False


# =============================================================================
# FORMATOS DE DISPLAY
# =============================================================================

def fmt_voltage(value):
    return f"{value:>9.4f}"


def fmt_current(value):
    return f"{value:>7.4f}"


def print_display(results, elapsed, ready, fs_ch):
    """
    Muestra solo magnitudes RMS (sin ángulos).
    """
    lines = []

    lines.append("╔════════════════════════════════════════════════════════════╗")
    lines.append("║     ADS1263 — Magnitud AC-RMS (sin ángulos)              ║")
    lines.append("╠═══════════╦══════════════════════╦══════════════════════╣")
    lines.append("║  Fase     ║  Tensión             ║  Corriente           ║")
    lines.append("╠═══════════╬══════════════════════╬══════════════════════╣")

    for i, ph in enumerate(PHASES):
        vch = V_CHNLS[i]
        ich = I_CHNLS[i]

        if ready:
            v_real = results[vch]
            i_real = results[ich]

            v_str = f"{fmt_voltage(v_real)} V"
            i_str = f"{fmt_current(i_real)} A"
        else:
            v_str = " acumulando..."
            i_str = " acumulando..."

        lines.append(f"║  Fase {ph}   ║  {v_str:<20}║  {i_str:<20}║")

    lines.append("╚═══════════╩══════════════════════╩══════════════════════╝")
    lines.append(f"  t={elapsed:>6.1f}s | ventana={WINDOW_SIZE} muestras | fs_ch≈{fs_ch:>8.2f} Sa/s")
    lines.append(
        f"  Escalas: "
        f"VA×{SCALE[0]:.2f}, VB×{SCALE[1]:.2f}, VC×{SCALE[2]:.2f} | "
        f"IA×{SCALE[5]:.2f}, IB×{SCALE[6]:.2f}, IC×{SCALE[7]:.2f}"
    )
    if ENABLE_ZERO_CLAMP:
        lines.append("  Filtro ruido: activo. Señales bajo umbral → 0.")
    else:
        lines.append("  Filtro ruido: desactivado.")

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# =============================================================================
# INICIO DEL PROGRAMA
# =============================================================================

print("\033[2J\033[H", end="")
print("============================================================")
print(" Lectura ADS1263 solo magnitud AC-RMS")
print("============================================================\n")

print("Asignación fija de canales:")
print("  VA = canal 0")
print("  VB = canal 1")
print("  VC = canal 2")
print("  IA = canal 5")
print("  IB = canal 6")
print("  IC = canal 7")
print("============================================================\n")

resp = input("¿Desea calibrar ahora? (s/n): ").strip().lower()

if resp == "s":
    calibrate_magnitude(ADC, CHANNELS, LABELS, CALIBRATION_SAMPLES)
    save_calibration()
else:
    if load_calibration():
        print(f"Calibración cargada desde {CALIBRATION_FILE}.\n")
    else:
        print(f"No se encontró {CALIBRATION_FILE}. Se usarán valores por defecto.\n")

print("Factores actuales:")
for ch in CHANNELS:
    unit = "V" if ch in V_CHNLS else "A"
    print(f"   {LABELS[ch]}: ×{SCALE[ch]:.6f} {unit}/V_ADC")

print("============================================================\n")
time.sleep(1.0)


# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================

results = {ch: 0.0 for ch in CHANNELS}

t_start = time.time()
last_display = 0.0
total_samp = 0
fs_ch = 1.0

try:
    while True:
        now = time.time()
        time_buffer.append(now)

        for ch in CHANNELS:
            raw = ADC.ADS1263_GetChannalValue(ch)
            v_adc = raw_to_voltage(raw)
            buffer_raw[ch].append(v_adc)

        total_samp += 1
        ready = total_samp >= WINDOW_SIZE

        now_display = time.time()

        if now_display - last_display >= DISPLAY_INTERVAL:
            last_display = now_display

            if ready:
                # Estimar frecuencia de muestreo real
                if len(time_buffer) >= 2:
                    dt_window = time_buffer[-1] - time_buffer[0]
                    fs_ch = (len(time_buffer) - 1) / dt_window if dt_window > 0 else 1.0

                for ch in CHANNELS:
                    results[ch] = compute_ac_rms_value(buffer_raw[ch], SCALE[ch], ch)

            elapsed = time.time() - t_start

            print_display(
                results,
                elapsed,
                ready,
                fs_ch
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