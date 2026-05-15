"""
read_adc.py — Lectura de canales IN0, IN1, IN2, IN5, IN6, IN7 del ADS1263
Hardware: Jetson Nano Dev Kit + Waveshare High-Precision AD HAT (ADS1263)
Modo: single-ended, tasa máxima disponible (38400 SPS)

Uso:
    sudo python3 read_adc.py

Salida: se sobreescribe en pantalla mostrando voltaje + muestras por canal.
Ctrl+C para salir.
"""

import time
import sys

# ── Importar la librería del HAT ──────────────────────────────────────────────
# Ajusta el import según la ruta real de tu proyecto.
# Si ADS1263.py está en lib/, usa:  import lib.ADS1263 as ADS1263
# Si está en el mismo directorio:   import ADS1263

sys.path.insert(0, '/home/santiago/PF_Smart-Relay/SPI/High-Pricision_AD_HAT/python')
import ADS1263


# ── Configuración ──────────────────────────────────────────────────────────────
REF          = 5.08         # Tensión de referencia (AVDD - AVSS). Ajusta si usas ref interna (2.5 V)
CHANNELS     = [0, 1, 2, 5, 6, 7]   # IN0 IN1 IN2 IN5 IN6 IN7
RATE         = 'ADS1263_38400SPS'   # Tasa máxima del ADC1

# Nombres de display para cada canal
CHANNEL_NAMES = {0: "IN0", 1: "IN1", 2: "IN2", 5: "IN5", 6: "IN6", 7: "IN7"}

# ── Inicialización del ADC ─────────────────────────────────────────────────────
ADC = ADS1263.ADS1263()

if ADC.ADS1263_init_ADC1(RATE) == -1:
    print("[ERROR] No se pudo inicializar el ADS1263. Verifica la conexión SPI.")
    sys.exit(1)

ADC.ADS1263_SetMode(0)   # 0 = single-ended (10 canales), 1 = diferencial (5 canales)

# ── Función de conversión raw → voltaje ───────────────────────────────────────
def raw_to_voltage(raw: int) -> float:
    """
    Convierte el valor raw de 32 bits con signo del ADS1263 a voltios.
    El ADS1263 entrega un entero de complemento a 2 de 32 bits.
    Rango: -REF  a  +REF  (modo single-ended respecto a COM/GND)
    """
    if raw >> 31:                          # bit 31 = 1  →  negativo
        return -(REF * 2 - raw / 2_147_483_648.0 * REF)
    else:
        return raw / 2_147_483_647.0 * REF

# ── Estado de muestras por canal ──────────────────────────────────────────────
sample_count  = {ch: 0 for ch in CHANNELS}
voltages      = {ch: 0.0 for ch in CHANNELS}

# ── Encabezado inicial ────────────────────────────────────────────────────────
HEADER_LINES  = 2 + len(CHANNELS) + 2   # separadores + filas de datos + pie

def print_display(elapsed: float):
    """Imprime el bloque de mediciones (una vez por iteración)."""
    lines = []
    lines.append("╔══════════════════════════════════════════════════════╗")
    lines.append("║        ADS1263 — Lectura en tiempo real              ║")
    lines.append("╠══════════╦════════════════╦════════════════╦═════════╣")
    lines.append("║  Canal   ║  Voltaje (V)   ║  Muestras      ║  kSPS   ║")
    lines.append("╠══════════╬════════════════╬════════════════╬═════════╣")
    for ch in CHANNELS:
        v    = voltages[ch]
        cnt  = sample_count[ch]
        ksps = cnt / elapsed / 1000 if elapsed > 0 else 0.0
        sign = "+" if v >= 0 else ""
        lines.append(f"║  {CHANNEL_NAMES[ch]:<6}  ║  {sign}{v:>10.5f}   ║  {cnt:>12,}  ║  {ksps:>5.2f}  ║")
    lines.append("╚══════════╩════════════════╩════════════════╩═════════╝")
    lines.append(f"  Tiempo transcurrido: {elapsed:>8.1f} s    [Ctrl+C para salir]")

    # Primera vez: imprimir normal; luego: subir cursor y sobreescribir
    sys.stdout.write("\n".join(lines) + "\n")
    # Subir tantas líneas como se imprimieron para sobreescribir en la próxima iteración
    sys.stdout.write(f"\033[{len(lines)}A")
    sys.stdout.flush()

# ── Bucle principal ───────────────────────────────────────────────────────────
print("\nIniciando lectura… (puede tardar un momento en estabilizarse)\n")
time.sleep(0.5)

t_start      = time.time()
first_print  = True

try:
    while True:
        # Leer cada canal en secuencia (channel-by-channel single-ended)
        for ch in CHANNELS:
            raw = ADC.ADS1263_GetChannalValue(ch)
            voltages[ch]       = raw_to_voltage(raw)
            sample_count[ch]  += 1

        # Actualizar pantalla cada vuelta completa (los 6 canales)
        elapsed = time.time() - t_start

        if first_print:
            # Imprimir encabezado vacío primero para que el cursor quede en posición
            first_print = False

        print_display(elapsed)

except KeyboardInterrupt:
    # Bajar cursor al final del bloque antes de salir
    sys.stdout.write(f"\033[{10 + len(CHANNELS)}B\n")
    sys.stdout.flush()
    print("\n[INFO] Lectura detenida por el usuario.")
    print(f"[INFO] Total de muestras por canal:")
    for ch in CHANNELS:
        print(f"       {CHANNEL_NAMES[ch]}: {sample_count[ch]:,} muestras")

finally:
    # Liberar el ADC correctamente
    try:
        ADC.ADS1263_Exit()
    except AttributeError:
        pass   # Algunas versiones de la lib no tienen Exit()
