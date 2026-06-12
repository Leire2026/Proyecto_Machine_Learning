
# importar librerias

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

#carga datos
df_modelo = pd.read_csv("../1_Data/processed.csv", sep = ",")

df_modelo.head()

df_modelo.info()

df_modelo.describe(include='all')



#Saco frecuencias y represento gráficamente la distribución de valores de la target para establecer los dos grupos de clasificación

frecuencias = df_modelo['Target_Satisfacc_Vida'].value_counts().sort_index()

plt.figure(figsize=(8,5))

bars = plt.bar(
    frecuencias.index,
    frecuencias.values,
    color='lightsteelblue'
)

# Etiquetas dentro de las barras
for bar in bars:
    altura = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        altura*0.95,
        f'{int(altura)}',
        ha='center',
        va='top'
    )

plt.title('Distribución de la satisfacción con la vida')
plt.xlabel('Nivel de satisfacción')
plt.ylabel('Frecuencia')

plt.xticks(frecuencias.index)

plt.tight_layout()
plt.show()



tabla_target = pd.DataFrame({
    'Frecuencia': df_modelo['Target_Satisfacc_Vida'].value_counts().sort_index()
})

tabla_target['Porcentaje'] = (
    tabla_target['Frecuencia'] /
    tabla_target['Frecuencia'].sum() * 100
).round(2)

tabla_target

media_target = df_modelo['Target_Satisfacc_Vida'].mean()
media_target



df_modelo['Bienestar_alto'] = (
    df_modelo['Target_Satisfacc_Vida'] >= 8
).astype(int)



tabla_clases = pd.DataFrame({
    'Frecuencia': df_modelo['Bienestar_alto'].value_counts().sort_index()
})

tabla_clases['Porcentaje'] = (
    tabla_clases['Frecuencia'] /
    tabla_clases['Frecuencia'].sum() * 100
).round(2)

tabla_clases.index = [
    'No bienestar alto (0-7)',
    'Bienestar alto (8-10)'
]

tabla_clases



# Guardo el dataframe final para modelado
df_modelo.to_csv('processed_clas.csv', index=False, encoding='utf-8')
df_modelo.head()



#Cargo nuevo dataset que utilizaré a a partir de ahora para trabajar con modelos de clasificación

df_mod_clas = pd.read_csv('../1_Data/processed_clas.csv')
df_mod_clas.head()



# Variables predictoras
X = df_mod_clas.drop(
    columns=['Target_Satisfacc_Vida', 'Bienestar_alto']
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
logreg = LogisticRegression(
    random_state=10
)

logreg.fit(X_train_scaled, y_train)



#Realizo las predicciones
y_pred_log = logreg.predict(X_test_scaled)
y_pred_log



#Métricas de evaluación

accuracy = accuracy_score(y_test, y_pred_log)
precision = precision_score(y_test, y_pred_log)
recall = recall_score(y_test, y_pred_log)
f1 = f1_score(y_test, y_pred_log)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")

#Matriz de confusión: 

cm = confusion_matrix(y_test, y_pred_log)

print(cm)


#Representación gráfica

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_log,
    cmap='Blues'
)

plt.title('Matriz de confusión - Regresión Logística')
plt.show()

