import Jetson.GPIO as GPIO
import spidev
import time

# --- CONFIGURACIÓN DE PINES (Numeración Física BOARD) ---
# Según la tabla de Waveshare:
# DRDY  -> BCM 17 -> Pin Físico 11
# RESET -> BCM 18 -> Pin Físico 12
# CS    -> BCM 22 -> Pin Físico 15
DRDY_PIN = 11
RST_PIN  = 12
CS_PIN   = 15

# Configuración de la librería GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(CS_PIN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(DRDY_PIN, GPIO.IN)

# Configuración del bus SPI
spi = spidev.SpiDev()
spi.open(0, 0) # Bus 0, Dispositivo 0 (SPI0)
spi.max_speed_hz = 1000000
spi.mode = 0b01 # El ADS1263 requiere Modo 1 (CPOL=0, CPHA=1)

def test_connection():
    print("Reiniciando el ADS1263...")
    # Pulso de Reset para despertar el chip
    GPIO.output(RST_PIN, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.1)

    print("Leyendo registro de ID (Registro 0x00)...")
    GPIO.output(CS_PIN, GPIO.LOW)
    
    # Protocolo ADS1263 para leer 1 registro:
    # 1. Comando RREG (0x20 | dirección del registro 0x00)
    # 2. Número de registros adicionales a leer (0x00 para leer solo uno)
    spi.xfer2([0x20, 0x00])
    
    # 3. Recibir la respuesta
    response = spi.xfer2([0x00])
    GPIO.output(CS_PIN, GPIO.HIGH)
    
    return response[0]

try:
    id_chip = test_connection()
    print(f"\nResultado -> ID detectado: {hex(id_chip)}")
    
    # El ADS1263 suele devolver un ID donde los bits altos indican la familia
    # Comúnmente 0x22, 0x23 o similar.
    if id_chip != 0x00 and id_chip != 0xFF:
        print("¡CONEXIÓN EXITOSA! La Jetson y el ADS1263 se están comunicando.")
    else:
        print("FALLO: Se recibió 0x0 o 0xff. Verifica que los jumpers amarillos")
        print("del HAT estén puestos en la posición SPI.")

except Exception as e:
    print(f"Error durante la prueba: {e}")

finally:
    spi.close()
    GPIO.cleanup()
