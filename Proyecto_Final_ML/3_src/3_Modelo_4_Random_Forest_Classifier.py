#Importar librerias

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

#Cargo nuevo dataset que utilizaré a a partir de ahora para trabajar con modelos de clasificación

df_mod_clas = pd.read_csv('../1_Data/processed_clas.csv')
df_mod_clas.head()


# Variables predictoras
X = df_mod_clas.drop(
    columns=['Terr_hist', 'Sexo', 'Edad_interv', 'Lugar_nacim', 'Nacionalidad',
       'Nivel_estudios', 'Actividad', 'Tipo_hogar','Target_Satisfacc_Vida', 'Bienestar_alto']
)

# Variable objetivo
y = df_mod_clas['Bienestar_alto']


#separo en test y train
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10,
    stratify=y
)

print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)


print("Train:")
print(y_train.value_counts(normalize=True))

print("\nTest:")
print(y_test.value_counts(normalize=True))


#Escalo
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Creo y entreno el modelo

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=10
)

rf.fit(X_train_scaled, y_train)


#Hago las predicciones

y_pred_rf = rf.predict(X_test_scaled)
y_pred_rf


#Metricas: 

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")



ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    cmap='Blues'
)

plt.title('Matriz de confusión - Random Forest')
plt.show()


#Importancia de las variables:

importancias_rf = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': rf.feature_importances_
})

importancias_rf.sort_values(
    by='Importancia',
    ascending=False
)


# ------------------------------------------------------------
# Ante los peores resultados obtenidos por el árbol de decisión, 
# hago una prueba reduciendo la profundidad máxima del árbol a 4 niveles, 
# con el objetivo de valorar si se está produciendo un overfitting que dificulta una adecuada generalización: 


#Vuelvo a entrenar el modelo limitando el riesgo de sobreajuste
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=10,
    random_state=10
)

rf.fit(X_train_scaled, y_train)


#Hago las predicciones

y_pred_rf = rf.predict(X_test_scaled)
y_pred_rf


#Metricas: 

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")



ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    cmap='Blues'
)

plt.title('Matriz de confusión - Random Forest Classifier')
plt.show()



# Guardar el modelo
with open("../4_Models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)

# Guardar el scaler
with open("../4_Models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

