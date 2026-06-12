# importar librerias

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle


# --- carga datos ---
df_modelo = pd.read_csv("../1_Data/processed.csv", sep = ",")

df_modelo

df_modelo.info()

df_modelo.describe(include='all')


# Separo X e y
# Variables predictoras
X = df_modelo.drop(columns=['Target_Satisfacc_Vida'])

# Variable target
y = df_modelo['Target_Satisfacc_Vida']

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

# Escalado

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(X_train_scaled.shape)
print(X_test_scaled.shape)


pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
).head()


# Creo y entreno el modelo
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)


# Hago predicciones sobre el conjunto de test
y_pred_lr = lr.predict(X_test_scaled)


#metricas de evaluacion comparando proyecciones con real:

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


#Guardo el modelo:

with open("../4_Models/regresion_lineal.pkl", "wb") as f:
    pickle.dump(lr, f)

