#importar librerias 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#carga archivo
df_modelo = pd.read_csv("../1_Data/processed.csv", sep = ",")

df_modelo


df_modelo.info()


df_modelo.describe(include='all')


#creo dataset para modelo eliminando variables demográficas:

df_modelo_reducido = df_modelo[[
    'Target_Satisfacc_Vida',
    'Satisf_relac_pers',
    'Sentido_propia_vida',
    'Satisf_econom_domest',
    'Valor_Estado_animo',
    'Satisf_vivienda',
    'Satisf_tiempo_disp',
    'Satisf_viv_y_entorno',
    'Salud_percibida',
    'Confianza_personas',
    'Confianza_poderes_pub'
]].copy()

# Separo X e y

X = df_modelo_reducido.drop(columns=['Target_Satisfacc_Vida'])

y = df_modelo_reducido['Target_Satisfacc_Vida']

print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)


# Separo en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=10
)


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Estandarización mediante StandardScaler. Esta transformación centra cada variable en media cero y desviación típica uno. El escalador se ajusta exclusivamente sobre el conjunto de entrenamiento y posteriormente se aplica al conjunto de prueba.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(X_train_scaled.shape)
print(X_test_scaled.shape)



pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
).head()


from sklearn.linear_model import LinearRegression

# Creo y entreno el modelo
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)


# Hago predicciones sobre el conjunto de test
y_pred_lr = lr.predict(X_test_scaled)


#metricas de evaluacion comparando proyecciones con real:
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


r2 = r2_score(y_test, y_pred_lr)
mae = mean_absolute_error(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)

print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")


coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': lr.coef_
})

coeficientes.sort_values(
    by='Coeficiente',
    ascending=False
)


coeficientes = coeficientes.sort_values(
    by='Coeficiente',
    ascending=True
)

plt.figure(figsize=(8, 5))

bars = plt.barh(
    coeficientes['Variable'],
    coeficientes['Coeficiente'],
    color='steelblue'
)

for bar in bars:
    plt.text(
        bar.get_width(),
        bar.get_y() + bar.get_height()/2,
        f'{bar.get_width():.3f}',
        va='center'
    )

plt.title('Coeficientes de la Regresión Lineal')
plt.xlabel('Coeficiente')

plt.tight_layout()
plt.show()

