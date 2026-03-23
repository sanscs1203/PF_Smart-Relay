import itertools
import csv
import numpy as np

# --- CONFIGURACIÓN DE DATOS ---
NODOS_DISPONIBLES = [611, 632, 633, 634, 645, 646, 652, 671, 675, 680, 684, 692]
RESISTENCIAS = [25, 50, 100]
VALORES_CARGA_FALLA = [0, 0.7, 1, 1.3]
NUM_CARGAS = 4 

# Configuración para clase "Sin_Falla" (9 pasos entre 0.7 y 1.3 + el valor 0)
VALORES_SIN_FALLA = [0] + list(np.round(np.linspace(0.7, 1.3, 9), 3))

FALLAS_POR_FASE = {
    'A': ['L A'], 'B': ['L B'], 'C': ['L C'],
    'AB': ['L A', 'L B', 'LL AB', 'LLG AB'],
    'BC': ['L B', 'L C', 'LL BC', 'LLG BC'],
    'CA': ['L A', 'L C', 'LL CA', 'LLG CA'],
    'ABC': ['L A', 'L B', 'L C', 'LL AB', 'LL BC', 'LL CA', 'LLG AB', 'LLG BC', 'LLG CA', 'LLL ABC']
}

def obtener_configuracion_nodos():
    config = {}
    print("\n--- Configuración de Fases por Nodo ---")
    for nodo in NODOS_DISPONIBLES:
        fases = input(f"¿Qué fases tiene el nodo {nodo}? (A, B, C, AB, BC, CA, ABC): ").upper().strip()
        while fases not in FALLAS_POR_FASE:
            fases = input(f"Error. Ingrese fases válidas (A, B, C, AB, BC, CA, ABC): ").upper().strip()
        config[nodo] = FALLAS_POR_FASE[fases]
    return config

def generar_archivo():
    config_nodos = obtener_configuracion_nodos()
    nombre_archivo = "simulacion_potencia.csv"
    conteo_fallas = {"Sin_Falla": 0}

    with open(nombre_archivo, mode='w', newline='') as f:
        escritor = csv.writer(f, delimiter=';')
        # Encabezados
        cargas_headers = [f"Carga_{i+1}" for i in range(NUM_CARGAS)]
        escritor.writerow(["Nodo_Falla", "Tipo_Falla", "Resistencia_Falla"] + cargas_headers)

        # 1. Clase Sin_Falla (N.A)
        print("Generando clase Sin_Falla...")
        for combo in itertools.product(VALORES_SIN_FALLA, repeat=NUM_CARGAS):
            fila = ["N.A", "Sin_Falla", "N.A"] + [str(c).replace('.', ',') for c in combo]
            escritor.writerow(fila)
            conteo_fallas["Sin_Falla"] += 1

        # 2. Clases con Falla
        print("Generando nodos con falla...")
        comb_cargas_falla = list(itertools.product(VALORES_CARGA_FALLA, repeat=NUM_CARGAS))
        for nodo, fallas in config_nodos.items():
            for falla in fallas:
                conteo_fallas[falla] = conteo_fallas.get(falla, 0) + (len(RESISTENCIAS) * len(comb_cargas_falla))
                for res in RESISTENCIAS:
                    for carga in comb_cargas_falla:
                        fila = [nodo, falla, str(res).replace('.', ',')] + [str(c).replace('.', ',') for c in carga]
                        escritor.writerow(fila)

    print(f"\n✅ Proceso terminado. Archivo: {nombre_archivo}")
    print("\n--- RESUMEN ---")
    for k, v in sorted(conteo_fallas.items()):
        print(f"{k}: {v} registros")

if __name__ == "__main__":
    generar_archivo()
