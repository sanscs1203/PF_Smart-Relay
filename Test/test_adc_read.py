import Jetson.GPIO as GPIO
import spidev
import time

DRDY = 11
RST  = 12
CS   = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(CS, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(RST, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(DRDY, GPIO.IN)

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 1000000
spi.mode = 0b01

# Reset
GPIO.output(RST, GPIO.LOW)
time.sleep(0.001)
GPIO.output(RST, GPIO.HIGH)
time.sleep(0.1)

# Configurar canal 0 vs AINCOM
def write_reg(reg, data):
    GPIO.output(CS, GPIO.LOW)
    spi.xfer2([0x40 | reg, 0x00, data])
    GPIO.output(CS, GPIO.HIGH)

write_reg(0x01, 0x11)  # POWER: internal ref on
write_reg(0x03, 0x00)  # MODE0: continuous
write_reg(0x04, 0x02)  # MODE1: 1000 SPS
write_reg(0x05, 0x00)  # MODE2: gain=1
write_reg(0x06, 0x0F)  # INPMUX: AIN0=0, AINCOM=15 -> 0x0F? wait, need (pos<<4)|neg: 0<<4|15=0x0F. OK.
write_reg(0x07, 0x10)  # REF: internal positive

# Comando START
GPIO.output(CS, GPIO.LOW)
spi.xfer2([0x08])
GPIO.output(CS, GPIO.HIGH)

print("Esperando DRDY...")
while GPIO.input(DRDY) == 1:
    time.sleep(0.001)
print("DRDY bajo, leyendo...")

# Leer datos
GPIO.output(CS, GPIO.LOW)
spi.xfer2([0x12])
data = spi.readbytes(5)
GPIO.output(CS, GPIO.HIGH)

code = int.from_bytes(data[1:5], 'big', signed=True)
voltage = code * 2.5 / (1 * 2**31)
print(f"Código: {code}, Voltaje: {voltage:.6f} V")

spi.close()
GPIO.cleanup()
