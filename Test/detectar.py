#!/usr/bin/env python3
"""
Smart Relay – Detección de fallas IEEE 5/13 Bus (ADS1263 + Jetson Nano)
Menú de selección, monitoreo continuo, alarma, logs y cierre limpio.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import Jetson.GPIO as GPIO
import spidev
from datetime import datetime, timedelta

# ─── Hardware ─────────────────────────────────────────────
DRDY_PIN = 11
RST_PIN  = 12
CS_PIN   = 15
TRIP_PIN = 40  # Pin físico 40 (D21) para señal de falla

V_SCALE = 50.0
I_SCALE = 30.0
ADC_RATE = 19200
LINE_FREQ = 60.0
CYCLES = 4

AINCOM = 10
CHANNELS = {
    'V_A': (0, AINCOM), 'V_B': (1, AINCOM), 'V_C': (2, AINCOM),
    'I_A': (5, AINCOM), 'I_B': (6, AINCOM), 'I_C': (7, AINCOM),
}

# Frecuencia efectiva de muestreo multiplexado (6 canales)
FS_EFECTIVA = ADC_RATE / len(CHANNELS)  # 3200 Hz por canal
BUFFER_SIZE = int(FS_EFECTIVA * (CYCLES / LINE_FREQ))  # ≈213 muestras por canal

BASE_MODELS_PATH = "Models/"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "fallas.txt")

FEATURE_COLS = [
    'Va', 'Vb', 'Vc', 
    'phi_Va', 'phi_Vb', 'phi_Vc', 
    'Ia', 'Ib', 'Ic', 
    'phi_Ia', 'phi_Ib', 'phi_Ic'
]

# ─── Buffer Circular Estructurado ─────────────────────────
class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.index = 0
        self.full = False

    def add(self, val):
        self.buffer[self.index] = val
        self.index += 1
        if self.index >= self.size:
            self.index = 0
            self.full = True

    def get_array(self):
        if not self.full:
            return self.buffer[:self.index]
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))

# ─── Driver de Baja Ralea ADS1263 ──────────────────────────
class ADS1263:
    def __init__(self):
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 8000000
        self.spi.mode = 0b01
        
    def init_hardware(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(RST_PIN, GPIO.OUT)
        GPIO.setup(CS_PIN, GPIO.OUT)
        GPIO.setup(DRDY_PIN, GPIO.IN)
        GPIO.setup(TRIP_PIN, GPIO.OUT)
        GPIO.output(TRIP_PIN, GPIO.LOW)
        
        # Reset físico
        GPIO.output(RST_PIN, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.1)
        
        # Configurar tasa a 19200 SPS (MODE1)
        GPIO.output(CS_PIN, GPIO.LOW)
        self.spi.xfer2([0x40 | 0x03, 0x00, 0x0E])
        GPIO.output(CS_PIN, GPIO.HIGH)
        time.sleep(0.01)

    def set_channel_fast(self, pos, neg):
        mux_val = (pos << 4) | neg
        GPIO.output(CS_PIN, GPIO.LOW)
        self.spi.xfer2([0x40 | 0x02, 0x00, mux_val])
        GPIO.output(CS_PIN, GPIO.HIGH)

    def read_sample_fast(self):
        GPIO.output(CS_PIN, GPIO.LOW)
        raw = self.spi.xfer2([0x12, 0x00, 0x00, 0x00, 0x00])
        GPIO.output(CS_PIN, GPIO.HIGH)
        
        val = (raw[1] << 24) | (raw[2] << 16) | (raw[3] << 8) | raw[4]
        if val & 0x80000000:
            val -= 0x100000000
            
        # Conversión básica a voltaje de referencia (2.5V de Waveshare)
        return val * (2.5 / 2147483647.0)

# ─── Procesamiento Matemático FFT ─────────────────────────
def compute_rms_angle(arr, scale, fs):
    N = len(arr)
    if N < 10:
        return 0.0, 0.0
        
    # Aplicar ventana de Hanning para mitigar filtrado espectral
    window = np.hanning(N)
    v_detrend = arr - np.mean(arr)
    fft_res = np.fft.rfft(v_detrend * window)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    # Encontrar el componente fundamental de la red (60 Hz)
    idx = np.argmin(np.abs(freqs - LINE_FREQ))
    
    # Corrección de ganancia por enventanado energético
    v_rms = (np.abs(fft_res[idx]) / N) * 2.0 * (1.0 / 0.707) * scale
    angle = np.angle(fft_res[idx], deg=True)
    
    return float(v_rms), float(angle)

# ─── Guardado de Registro de Eventos ──────────────────────
def log_failure(display_data, sys_name):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(LOG_FILE, "a") as f:
        f.write(f"\n==================================================\n")
        f.write(f"REGISTRO DE FALLA - {sys_name} - {now_str}\n")
        f.write(f"==================================================\n")
        for ph, v, i in display_data:
            f.write(f"Fase {ph}: V={v[0]:12.6f} V ∠{v[1]:6.2f}° | I={i[0]:12.6f} A ∠{i[1]:6.2f}°\n")
        f.write(f"Acción: SEÑAL DE DISPARO (TRIP_PIN HIGH) ENVIADA\n")

# ─── Bucle Principal de Ejecución ─────────────────────────
def main():
    adc = ADS1263()
    adc.init_hardware()
    
    # ── Selección de Sistema ──
    print("="*50)
    print("       SMART RELAY - SELECCIÓN DE SISTEMA")
    print("="*50)
    print("1. IEEE 5-Bus")
    print("2. IEEE 13-Bus")
    opc = input("Seleccione una opción (1-2): ")
    
    if opc == '1':
        sys_name = "IEEE 5-Bus"
        model_file = "model_det_ieee5.pkl"
        scaler_file = "scaler_det_ieee5.pkl"
    elif opc == '2':
        sys_name = "IEEE 13-Bus"
        model_file = "model_det_ieee13.pkl"
        scaler_file = "scaler_det_ieee13.pkl"
    else:
        print("Opción inválida.")
        GPIO.cleanup()
        return

    model_path = os.path.join(BASE_MODELS_PATH, model_file)
    scaler_path = os.path.join(BASE_MODELS_PATH, scaler_file)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error Crítico: Asegúrese de que {model_path} y {scaler_path} existan.")
        GPIO.cleanup()
        return

    print(f"\nCargando Modelo Inteligente...")
    model = joblib.load(model_path)
    print(f"Cargando Escalador Lineal ({scaler_file})...")
    scaler = joblib.load(scaler_path)

    # Inicializar buffers individuales para cada canal físico
    buffers = {name: CircularBuffer(BUFFER_SIZE) for name in CHANNELS.keys()}
    ch_list = [(name, config[0], config[1]) for name, config in CHANNELS.items()]
    n_channels = len(ch_list)
    
    last_display = time.time()
    ch_idx = 0

    print(f"\nMonitoreo activo para {sys_name}. Presione Ctrl+C para detener.")
    
    try:
        while True:
            # Adquisición Round-Robin
            name, pos, neg = ch_list[ch_idx]
            adc.set_channel_fast(pos, neg)
            
            val = adc.read_sample_fast()
            buffers[name].add(val)

            ch_idx += 1
            if ch_idx >= n_channels:
                ch_idx = 0
                
                # Cada barrido completo, procesar variables de entrada
                display_data = []
                for ph in ['A', 'B', 'C']:
                    v_arr = buffers[f'V_{ph}'].get_array()
                    i_arr = buffers[f'I_{ph}'].get_array()
                    
                    v_rms, v_ang = compute_rms_angle(v_arr, V_SCALE, FS_EFECTIVA)
                    i_rms, i_ang = compute_rms_angle(i_arr, I_SCALE, FS_EFECTIVA)
                    
                    display_data.append((ph, (v_rms, v_ang), (i_rms, i_ang)))
                
                # Ejecutar canal de inferencia estructurado
                try:
                    v_list = [display_data[i][1][0] for i in range(3)]
                    v_ang_list = [display_data[i][1][1] for i in range(3)]
                    i_list = [display_data[i][2][0] for i in range(3)]
                    i_ang_list = [display_data[i][2][1] for i in range(3)]
                    
                    raw_vector = v_list + v_ang_list + i_list + i_ang_list
                    
                    # Convertir a DataFrame antes de escalar
                    df_raw = pd.DataFrame([raw_vector], columns=FEATURE_COLS)
                    df_scaled = pd.DataFrame(scaler.transform(df_raw), columns=FEATURE_COLS)
                    
                    pred = model.predict(df_scaled)[0]
                except Exception:
                    pred = 0

                # Sistema de disparo Físico y lógico
                if pred == 1:
                    GPIO.output(TRIP_PIN, GPIO.HIGH)
                    log_failure(display_data, sys_name)
                    print("\n⚠️ >>> ¡FALLA DETECTADA POR EL MODELO DE IA! TRIPPING DISPOSITIVO <<< ⚠️")
                
                # Control de renderizado en terminal (Frecuencia controlada a ~10 Hz)
                now = time.time()
                if now - last_display > 0.1:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print(f"==================================================")
                    print(f"      SMART RELAY ACTIVO - SISTEMA: {sys_name.upper()}")
                    print(f"==================================================")
                    for ph, v, i in display_data:
                        print(f"Fase {ph}: V = {v[0]:8.3f} Vrms ∠ {v[1]:6.1f}° | I = {i[0]:6.3f} Arms ∠ {i[1]:6.1f}°")
                    print(f"--------------------------------------------------")
                    print(f"Estado de salida del Relé (Pin 40): {'[FALLA - TRIP]' if pred == 1 else '[NORMAL]'}")
                    last_display = now

    except KeyboardInterrupt:
        print("\nMonitoreo detenido por interrupción del usuario.")
    finally:
        GPIO.output(TRIP_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("Recursos de Hardware liberados. Salida limpia ejecutada.")

if __name__ == "__main__":
    main()
