import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    make_scorer
)

# Load DataSet
file = 'DataSet_Nodes_5.csv'
df = pd.read_csv(file, sep=';', decimal=',')

# Fix: convert RF and Factor_Carga to float
df['RF'] = pd.to_numeric(df['RF'], errors='coerce')
df['Factor_Carga'] = pd.to_numeric(df['Factor_Carga'], errors='coerce')
df['Nodo_Falla'] = df['Nodo_Falla'].apply(lambda x: np.nan if x == 'Sin_Falla' else x)

# Define roles of each column
features = [
    'Va', 'Vb', 'Vc',
    'Ia', 'Ib', 'Ic'
]

metadata = [
    'RF', 'Factor_Carga', 'Nodo_Falla'
]

label = 'Tipo_Falla'

# General information of the file
print("=" * 50)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 50)
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nColumnas encontradas:\n{df.columns.tolist()}")
print(f"\nTipos de datos:\n{df.dtypes}")

# Check the existance of all the expected columns
expected_columns = features + metadata + [label]
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    print(f"\n⚠️ COLUMNAS FALTANTES: {missing_columns}")
else:
    print("\n✅ Todas las columnas esperadas están presentes")
    
# Descriptive statistics of characteristics
print("\n" + "=" * 50)
print("ESTADÍSTICAS DESCRIPTIVAS (FEATURES)")
print("=" * 50)
print(df[features].describe().round(2))

# Distribution of the target variable
print("\n" + "=" * 50)
print("DISTRIBUCIÓN DE CLASES (Tipo_Falla)")
print("=" * 50)
conteo = df[label].value_counts()
print(conteo)
print(f"\nTotal de clases: {df[label].nunique()}")

# Distribution of the metadata
print("\n" + "=" * 50)
print("DISTRIBUCIÓN DE METADATA")
print("=" * 50)
print(f"\nValores únicos de RF: {sorted(df['RF'].dropna().unique())}")
print(f"Valores únicos de Factor_Carga: {sorted(df['Factor_Carga'].dropna().unique())}")
print(f"Valores únicos de Nodo_Falla: {sorted(df['Nodo_Falla'].dropna().unique())}")

# Binary label for detection model
df['Label_detection'] = df['Tipo_Falla'].apply(
    lambda x: 0 if x == 'Sin_Falla' else 1
)

print("=" * 50)
print("DISTRIBUCIÓN - MODELO DE DETECCIÓN")
print("=" * 50)
print(df['Label_detection'].value_counts())
print(f"\nProporción clase 1 (Falla): {df['Label_detection'].mean():.2%}")
print(f"Proporción clase 0 (No Falla): {1 - df['Label_detection'].mean():.2%}")

X = df[features]
y = df['Label_detection']

print("\n" + "=" * 50)
print("DIMENSIONES")
print("=" * 50)
print(f"Features (X): {X.shape}")
print(f"Label (y): {y.shape}")

# Verify classes balance
ratio = df['Label_detection'].value_counts()
ratio_min_max = ratio.min() / ratio.max()

print("\n" + "=" * 50)
print("BALANCE DE CLASES")
print("=" * 50)
print(f"Ratio minoritaria/mayoritaria: {ratio_min_max:.2%}")

if ratio_min_max >= 0.8:
    print("✅ Clases balanceadas")
elif ratio_min_max >= 0.5:
    print("⚠️ Desbalance moderado - Considerar class_weight='balanced'")
else:
    print("🚨 Desbalance severo - Aplicar SMOTE o class_weight='balanced'")
    
    
# Train/Test - 80/20

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,       # Reproducibilidad
    stratify=y             # Mantener proporción de clases
)

# Check dimensions
print("=" * 50)
print("DIMENSIONES TRAS LA DIVISIÓN")
print("=" * 50)
print(f"Train: {X_train.shape[0]} muestras ({X_train.shape[0]/X.shape[0]:.0%})")
print(f"Test:  {X_test.shape[0]} muestras ({X_test.shape[0]/X.shape[0]:.0%})")

# Check proportion
print("\n" + "=" * 50)
print("PROPORCIÓN DE CLASES")
print("=" * 50)
print(f"Dataset original:  Falla={y.mean():.2%}  |  No Falla={1-y.mean():.2%}")
print(f"Train:             Falla={y_train.mean():.2%}  |  No Falla={1-y_train.mean():.2%}")
print(f"Test:              Falla={y_test.mean():.2%}  |  No Falla={1-y_test.mean():.2%}")


# BASELINE
# Create model with defaul hyperparameters
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',   
    random_state=42,
    n_jobs=-1                  
)

# Train the model
print("=" * 50)
print("ENTRENAMIENTO - RANDOM FOREST BASELINE")
print("=" * 50)

start = time.time()
rf_baseline.fit(X_train, y_train)
train_time = time.time() - start

print(f"Modelo entrenado exitosamente")
print(f"Tiempo de entrenamiento: {train_time:.3f} segundos")

# Prediction and time of inference
start = time.time()
y_pred = rf_baseline.predict(X_test)
total_inference_time = time.time() - start

time_per_sample = (total_inference_time / X_test.shape[0]) * 1000  # In miliseconds

print(f"\nTiempo de inferencia total: {total_inference_time:.4f} segundos")
print(f"Tiempo por muestra: {time_per_sample:.4f} milisegundos")

# Get probabilities
y_proba = rf_baseline.predict_proba(X_test)[:, 1]  # Probabilities of class 1 _ Fail

# General view of the results
print("\n" + "=" * 50)
print("VISTA RÁPIDA")
print("=" * 50)
print(f"Predicciones positivas (Falla): {(y_pred == 1).sum()}")
print(f"Predicciones negativas (No Falla): {(y_pred == 0).sum()}")
print(f"valuees reales positivos (Falla): {(y_test == 1).sum()}")
print(f"valuees reales negativos (No Falla): {(y_test == 0).sum()}")

# Individual measures
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)
roc_auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)

# Specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("=" * 50)
print("MÉTRICAS - RANDOM FOREST BASELINE")
print("=" * 50)

# Critical measures
print("\n🔴 MÉTRICAS CRÍTICAS:")
print(f"  Recall:       {recall:.4f}")
print(f"  F2-Score:     {f2:.4f}")
print(f"  ROC-AUC:      {roc_auc:.4f}")
print(f"  Tiempo/muestra: {time_per_sample:.4f} ms")

# Complementary measures
print("\n🟡 MÉTRICAS COMPLEMENTARIAS:")
print(f"  Precision:    {precision:.4f}")
print(f"  Specificity:  {specificity:.4f}")
print(f"  F1-Score:     {f1:.4f}")
print(f"  Accuracy:     {accuracy:.4f}")

# Confusion matrix
print("\n" + "=" * 50)
print("MATRIZ DE CONFUSIÓN")
print("=" * 50)
print(f"""
                  Predicho
                  No Falla    Falla
Real No Falla      {tn}          {fp}
Real Falla         {fn}          {tp}
""")
print(f"  Verdaderos Negativos (TN): {tn} → No Falla correctamente detectada")
print(f"  Falsos Positivos (FP):     {fp} → Disparo innecesario")
print(f"  Falsos Negativos (FN):     {fn} → ⚠️ FALLA NO DETECTADA (crítico)")
print(f"  Verdaderos Positivos (TP): {tp} → Falla correctamente detectada")

# Clasiffication report
print("\n" + "=" * 50)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 50)
print(classification_report(
    y_test, y_pred,
    target_names=['No Falla', 'Falla']
))

# Weighted score
# Normalize inference time (inverted, lower is better)
# Use 1/(1+t) so that it is between 0 and 1
tiempo_norm = 1 / (1 + time_per_sample)

score_balanced = (
    0.30 * recall +
    0.20 * f2 +
    0.20 * roc_auc +
    0.20 * specificity +
    0.10 * tiempo_norm
)

print("=" * 50)
print("SCORE PONDERADO")
print("=" * 50)
print(f"  Recall (0.30):       {0.30 * recall:.4f}")
print(f"  F2-Score (0.20):     {0.20 * f2:.4f}")
print(f"  ROC-AUC (0.20):     {0.20 * roc_auc:.4f}")
print(f"  Specificity (0.20): {0.20 * specificity:.4f}")
print(f"  Tiempo (0.10):      {0.10 * tiempo_norm:.4f}")
print(f"\n  📊 SCORE TOTAL:      {score_balanced:.4f}")

# OPTIMIZED
def score_balanced_func(y_true, y_proba):
    """
    Weighted score WITHOUT time component.
    Time is evaluated separately after the search.
    Maximum possible = 0.90 (0.10 for time is missing)
    """
    y_pred = (y_proba >= 0.5).astype(int)
    
    rec = recall_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0    
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    score = (
        0.30 * rec +
        0.20 * f2 +
        0.20 * auc +
        0.20 * spec
    )
    
    return score

custom_scorer = make_scorer( 
    score_balanced_func, 
    response_method='predict_proba'
)

# Define param_grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("=" * 50)
print("CONFIGURACIÓN DEL GRIDSEARCHCV")
print("=" * 50)
total_combinations = 4 * 5 * 3 * 3 * 2
print(f"Total combinaciones: {total_combinations}")
print(f"Total entrenamientos (x5 folds): {total_combinations * 5}")

# Configure and executeGridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=1              
    ),
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=cv,
    n_jobs=-1,                
    verbose=0,
    return_train_score=True
)

print("\nIniciando búsqueda...\n")

start = time.time()
grid_search.fit(X_train, y_train)
time_grid = time.time() - start

print(f"\n✅ Búsqueda completada en {time_grid:.1f} segundos")

# Shown progress of the score
print("\n" + "=" * 50)
print("PROGRESO: MEJORAS DEL SCORE")
print("=" * 50)

results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values('rank_test_score', ascending=True)

best_so_far = -1
mejora_num = 0

for _, row in results.iterrows():
    score = row['mean_test_score']
    if score > best_so_far:
        mejora_num += 1
        best_so_far = score
        params = row['params']
        print(f"\n  Mejora #{mejora_num} | Score: {score:.4f} ± {row['std_test_score']:.4f}")
        for param, value in params.items():
            print(f"    {param}: {value}")

# Top 5 - Hyperparameters found
print("\n" + "=" * 50)
print("TOP 5 MODELOS - SCORE COMPLETO (CON TIEMPO)")
print("=" * 50)

results = pd.DataFrame(grid_search.cv_results_)
top5 = results.nsmallest(5, 'rank_test_score')

best_score_final = -1
best_model_final = None

for i, (_, row) in enumerate(top5.iterrows()):
    
    # Recreate the model with the hyperparameters of that combination
    params = row['params']
    model_temp = RandomForestClassifier(
        **params,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model_temp.fit(X_train, y_train)
    
    # Predictions
    y_pred_temp = model_temp.predict(X_test)
    y_proba_temp = model_temp.predict_proba(X_test)[:, 1]
    
    # Metrics
    rec = recall_score(y_test, y_pred_temp)
    f2 = fbeta_score(y_test, y_pred_temp, beta=2)
    try:
        auc = roc_auc_score(y_test, y_proba_temp)
    except ValueError:
        auc = 0.0
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Inference time 
    tiempos = []
    for _ in range(10):
        t_start = time.time()
        model_temp.predict(X_test)
        t_end = time.time()
        tiempos.append((t_end - t_start) / X_test.shape[0] * 1000)
    t_inf = np.mean(tiempos)
    t_norm = 1 / (1 + t_inf)
    
    # Score with the time
    score_final = (
        0.30 * rec +
        0.20 * f2 +
        0.20 * auc +
        0.20 * spec +
        0.10 * t_norm
    )
    
    print(f"\n  Modelo #{i+1}")
    print(f"  {'Métrica':<20} {'Valor':>8} {'Peso':>6} {'Aporte':>8}")
    print(f"  {'-'*44}")
    print(f"  {'Recall':<20} {rec:>8.4f} {'0.30':>6} {0.30*rec:>8.4f}")
    print(f"  {'F2-Score':<20} {f2:>8.4f} {'0.20':>6} {0.20*f2:>8.4f}")
    print(f"  {'ROC-AUC':<20} {auc:>8.4f} {'0.20':>6} {0.20*auc:>8.4f}")
    print(f"  {'Specificity':<20} {spec:>8.4f} {'0.20':>6} {0.20*spec:>8.4f}")
    print(f"  {'Tiempo (norm)':<20} {t_norm:>8.4f} {'0.10':>6} {0.10*t_norm:>8.4f}")
    print(f"  {'-'*44}")
    print(f"  {'SCORE TOTAL':<20} {score_final:>8.4f}")
    print(f"  Tiempo/muestra: {t_inf:.4f} ms")
    print(f"  Params: {params}")
    
    # Save the best
    if score_final > best_score_final:
        best_score_final = score_final
        best_model_final = model_temp
        best_params_final = params

# Shown the best one
print("\n" + "=" * 50)
print("MODELO DEFINITIVO (MEJOR SCORE CON TIEMPO)")
print("=" * 50)
print(f"\n  Score final: {best_score_final:.4f}")
for param, value in best_params_final.items():
    print(f"  {param}: {value}")

rf_optimizado = best_model_final