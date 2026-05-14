import joblib
import pandas as pd
import os

# Rutas locales relativas a la estructura actual de tu carpeta Models
MODEL_PATH = "Models/model_det_ieee13.pkl"
SCALER_PATH = "Models/scaler_det_ieee13.pkl"

def manual_test():
    # 1. Verificar si el modelo y el escalador existen
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return
    if not os.path.exists(SCALER_PATH):
        print(f"Error: No se encontró el escalador en {SCALER_PATH}")
        return

    # 2. Cargar el modelo y el escalador
    print("Cargando modelo de detección IEEE 13-Bus...")
    model = joblib.load(MODEL_PATH)
    print("Cargando escalador de detección...")
    scaler = joblib.load(SCALER_PATH)

    # 3. Definir las columnas en el orden estricto del entrenamiento
    columns = [
        'Va', 'Vb', 'Vc', 
        'phi_Va', 'phi_Vb', 'phi_Vc', 
        'Ia', 'Ib', 'Ic', 
        'phi_Ia', 'phi_Ib', 'phi_Ic'
    ]

    print("\n--- Ingrese los valores para la prueba ---")
    data = {}
    try:
        for col in columns:
            val = input(f"Ingrese valor para {col}: ")
            data[col] = float(val)
    except ValueError:
        print("Error: Por favor ingrese solo números.")
        return

    # 4. Crear DataFrame para la inferencia
    df_raw = pd.DataFrame([data], columns=columns)

    # 5. Escalar los datos de entrada
    df_scaled = pd.DataFrame(scaler.transform(df_raw), columns=columns)

    # 6. Realizar la predicción con los datos normalizados
    prediction = model.predict(df_scaled)[0]

    print("\n" + "="*35)
    if prediction == 1:
        print(">>> RESULTADO: ¡FALLA DETECTADA! <<<")
    else:
        print(">>> RESULTADO: SISTEMA NORMAL (SIN FALLA) <<<")
    print("="*35)

if __name__ == "__main__":
    manual_test()
