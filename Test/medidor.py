#!/usr/bin/env python3
"""
Medidor trifásico de ultra-alto rendimiento — ADS1263 + Jetson Nano
Optimizado para muestreo multiplexado agresivo, red de 60Hz y mapeo físico final.
"""

import os
import sys
import time
import numpy as np
import Jetson.GPIO as GPIO
import spidev

# ─── Configuración de Hardware (Físico) ───────────────────
DRDY_PIN = 11
RST_PIN  = 12
CS_PIN   = 15

# Ajusta estas escalas según la relación de tus sensores físicos
V_SCALE = 50.0  
I_SCALE = 30.0  

# Parámetros de muestreo optimizados para 60 Hz
ADC_RATE = 19200               # Filtro configurado a 19,200 SPS
LINE_FREQ = 60.0               # Configurado para red de 60 Hz
CYCLES = 4                     # Ciclos de red en buffer

# Entrada negativa común apuntando a AINCOM (Polo a tierra mediante jumper COM-GND de Waveshare)
AINCOM = 10 

# Mapeo físico definitivo solicitado
CHANNELS = {
    'V_A': (0, AINCOM),  # ZMPT101B Fase A -> IN0 (Header AIN)
    'V_B': (1, AINCOM),  # ZMPT101B Fase B -> IN1 (Header AIN)
    'V_C': (2, AINCOM),  # ZMPT101B Fase C -> IN2 (Header AIN)
    'I_A': (5, AINCOM),  # SCT013 Fase A   -> IN5 (Regleta de tornillos)
    'I_B': (6, AINCOM),  # SCT013 Fase B   -> IN6 (Regleta de tornillos)
    'I_C': (7, AINCOM),  # SCT013 Fase C   -> IN7 (Regleta de tornillos)
}

# Tamaño de buffer optimizado para 60 Hz (~128 muestras por canal)
BUFFER_SIZE = int((ADC_RATE / 6) * (CYCLES / LINE_FREQ))
WINDOW = np.hanning(BUFFER_SIZE)

class ADS1263_HighSpeed:
    CMD_RESET  = 0x06
    CMD_START  = 0x08
    CMD_WREG   = 0x40
    REG_POWER  = 0x01
    REG_MODE0  = 0x03
    REG_MODE1  = 0x04
    REG_MODE2  = 0x05
    REG_INPMUX = 0x06
    REG_REF    = 0x07

    def __init__(self, cs, rst, drdy):
        self.cs = cs
        self.rst = rst
        self.drdy = drdy
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.cs, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.rst, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.drdy, GPIO.IN)
        
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 4000000 # 4 MHz para asegurar integridad física de datos
        self.spi.mode = 0b01

    def _write_reg(self, reg, data):
        GPIO.output(self.cs, GPIO.LOW)
        self.spi.xfer2([self.CMD_WREG | reg, 0x00, data])
        GPIO.output(self.cs, GPIO.HIGH)

    def reset(self):
        GPIO.output(self.rst, GPIO.LOW)
        time.sleep(0.002)
        GPIO.output(self.rst, GPIO.HIGH)
        time.sleep(0.05)
        
        GPIO.output(self.cs, GPIO.LOW)
        self.spi.xfer2([self.CMD_RESET])
        GPIO.output(self.cs, GPIO.HIGH)
        time.sleep(0.05)
        
        self._write_reg(self.REG_POWER, 0x11)  # Habilitar VBIAS y bypass interno
        self._write_reg(self.REG_MODE0, 0x00)  # Modo continuo
        self._write_reg(self.REG_MODE1, 0x0D)  # 19,200 SPS
        self._write_reg(self.REG_MODE2, 0x00)  # Ganancia = 1 (+-2.5V)
        self._write_reg(self.REG_REF, 0x10)    # Referencia interna de 2.5V

    def start(self):
        GPIO.output(self.cs, GPIO.LOW)
        self.spi.xfer2([self.CMD_START])
        GPIO.output(self.cs, GPIO.HIGH)

    def set_channel_fast(self, pos, neg):
        mux = (pos << 4) | (neg & 0x0F)
        GPIO.output(self.cs, GPIO.LOW)
        self.spi.xfer2([self.CMD_WREG | self.REG_INPMUX, 0x00, mux])
        GPIO.output(self.cs, GPIO.HIGH)

    def read_sample_fast(self):
        while GPIO.input(self.drdy):
            pass  # Espera activa por hardware (absorbe de forma óptima el settling time)
        
        GPIO.output(self.cs, GPIO.LOW)
        raw = self.spi.xfer2([0x12, 0x00, 0x00, 0x00, 0x00])
        GPIO.output(self.cs, GPIO.HIGH)
        
        code = (raw[1] << 24) | (raw[2] << 16) | (raw[3] << 8) | raw[4]
        if code & 0x80000000:
            code -= 0x100000000
            
        return code * 2.5 / 2147483648.0

    def close(self):
        self.spi.close()
        GPIO.cleanup()

class FastCircularBuffer:
    def __init__(self, size):
        self.buf = np.zeros(size, dtype=np.float32)
        self.idx = 0
        self.size = size
        self.full = False

    def add(self, val):
        self.buf[self.idx] = val
        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.full = True

    def get_array(self):
        if not self.full:
            return self.buf[:self.idx]
        return np.concatenate((self.buf[self.idx:], self.buf[:self.idx]))

def compute_rms_angle(arr, scale, fs_efectiva):
    N = len(arr)
    if N < 32: 
        return 0.0, 0.0
    
    ac = arr - np.mean(arr)
    windowed = ac * (np.hanning(N) if N != BUFFER_SIZE else WINDOW)
    
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N, d=1.0/fs_efectiva)
    
    mask = (freqs >= LINE_FREQ - 5) & (freqs <= LINE_FREQ + 5)
    if np.any(mask):
        idx_mask = np.where(mask)[0]
        bin_idx = np.argmax(np.abs(fft[mask]))
        real_idx = idx_mask[bin_idx]
        
        mag_peak = (4.0 * np.abs(fft[real_idx])) / N
        rms_fund = (mag_peak / 1.41421356) * scale
        ang_deg = np.degrees(np.angle(fft[real_idx])) % 360
        return rms_fund, ang_deg
    return 0.0, 0.0

def update_display(data):
    sys.stdout.write("\033[H\033[J")
    sys.stdout.write("==================================================\n")
    sys.stdout.write("   MEDIDOR TRIFÁSICO ADS1263 - OPTIMIZADO 60Hz    \n")
    sys.stdout.write("==================================================\n")
    for phase, v, i in data:
        sys.stdout.write(f"\nFASE {phase}\n")
        sys.stdout.write(f"  Voltaje  : {v[0]:12.6f} V  ∠ {v[1]:6.1f}°\n")
        sys.stdout.write(f"  Corriente: {i[0]:12.6f} A  ∠ {i[1]:6.1f}°\n")
    sys.stdout.write("\n==================================================\n")
    sys.stdout.flush()

def main():
    adc = ADS1263_HighSpeed(CS_PIN, RST_PIN, DRDY_PIN)
    adc.reset()
    adc.start()

    buffers = {name: FastCircularBuffer(BUFFER_SIZE) for name in CHANNELS}
    ch_list = [(name, config[0], config[1]) for name, config in CHANNELS.items()]
    n_channels = len(ch_list)
    
    fs_efectiva = ADC_RATE / n_channels
    last_display = time.time()
    ch_idx = 0

    print(f"Medidor iniciado a 60 Hz. Frecuencia efectiva por canal: {fs_efectiva:.2f} Hz.")

    try:
        while True:
            name, pos, neg = ch_list[ch_idx]
            adc.set_channel_fast(pos, neg)
            try:
                val = adc.read_sample_fast()
                buffers[name].add(val)
            except Exception:
                pass

            ch_idx += 1
            if ch_idx >= n_channels:
                ch_idx = 0
                
                now = time.time()
                if now - last_display > 0.1:
                    display_data = []
                    for ph in ['A', 'B', 'C']:
                        v_arr = buffers[f'V_{ph}'].get_array()
                        i_arr = buffers[f'I_{ph}'].get_array()
                        
                        v_rms, v_ang = compute_rms_angle(v_arr, V_SCALE, fs_efectiva)
                        i_rms, i_ang = compute_rms_angle(i_arr, I_SCALE, fs_efectiva)
                        
                        display_data.append((ph, (v_rms, v_ang), (i_rms, i_ang)))
                    
                    update_display(display_data)
                    last_display = now

    except KeyboardInterrupt:
        print("\nPrograma terminado.")
    finally:
        adc.close()

if __name__ == "__main__":
    main()
