
#Importar librerias

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle


#Cargo nuevo dataset que utilizaré a a partir de ahora para trabajar con modelos de clasificación

df_mod_cluster = pd.read_csv('../1_Data/processed_clas.csv')
df_mod_cluster.head()


#creo dataframe para trabajar cluster: 

variables_cluster = [
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
]

X_cluster = df_mod_cluster[variables_cluster]
X_cluster.head()


# Escalado

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)


#Calculo de la inercia

inercias = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=10,
        n_init=10
    )
    
    kmeans.fit(X_cluster_scaled)
    inercias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(2, 11), inercias, marker='o')

plt.title('Método del codo')
plt.xlabel('Número de clústeres')
plt.ylabel('Inercia')
plt.xticks(range(2, 11))

plt.show()


#Calculo el silhouette

silhouettes = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=10,
        n_init=10
    )
    
    labels = kmeans.fit_predict(X_cluster_scaled)
    silhouettes.append(silhouette_score(X_cluster_scaled, labels))

plt.figure(figsize=(8,5))

plt.plot(range(2, 11), silhouettes, marker='o')

plt.title('Silhouette Score por número de clústeres')
plt.xlabel('Número de clústeres')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))

plt.show()

# ------------------------------------------------------------

# Creo el modelo K-Means con 2 clusters
kmeans = KMeans(
    n_clusters=2,
    random_state=10,
    n_init=10
)

# Ajusto el modelo y obtengo las etiquetas
df_mod_cluster['Cluster'] = kmeans.fit_predict(X_cluster_scaled)


df_mod_cluster['Cluster'].value_counts()


df_mod_cluster['Cluster'].value_counts(normalize=True) * 100


df_mod_cluster.groupby('Cluster')[variables_cluster].mean().round(2)


df_mod_cluster.groupby('Cluster')[
    ['Target_Satisfacc_Vida', 'Bienestar_alto']
].mean().round(2)



#Guardo el modelo:

with open("../4_Models/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)


#Analisis de algunas características demográficas de cada cluster:
variables_demo = [
    'Terr_hist',
    'Sexo',
    'Edad_interv',
    'Lugar_nacim',
    'Nacionalidad',
    'Nivel_estudios',
    'Actividad',
    'Tipo_hogar'
]


def tabla_satisfaccion(variable):

    # Media de satisfacción por categoría y cluster
    tabla = pd.pivot_table(
        df_mod_cluster,
        values='Target_Satisfacc_Vida',
        index=variable,
        columns='Cluster',
        aggfunc='mean'
    )

    # Renombrar clusters
    tabla.columns = [
        'Alto nivel de bienestar',
        'Bajo nivel de bienestar'
    ]

    # Media para el total de la población
    tabla['Total población'] = (
        df_mod_cluster
        .groupby(variable)['Target_Satisfacc_Vida']
        .mean()
    )

    # Renombrar categorías de Sexo
    if variable == 'Sexo':
        tabla.index = tabla.index.map({
            1: 'Hombre',
            6: 'Mujer'
        })
        
    # Renombrar categorías de Terr Hist
    if variable == 'Terr_hist':
        tabla.index = tabla.index.map({
            1: 'Araba',
            20: 'Gipuzkoa',
            48: 'Bizkaia'
        })

    # Renombrar categorías de Actividad
    if variable == 'Actividad':
        tabla.index = tabla.index.map({
            1: 'Ocupados',
            2: 'Parados',
            3: 'Inactivos'
        })
    # Renombrar categorías de Nacionalidad
    if variable == 'Nacionalidad':
        tabla.index = tabla.index.map({
            1: 'Nacional',
            6: 'Extranjera'
        })
        
    # Renombrar categorías de Edad
    if variable == 'Edad_interv':
        tabla.index = tabla.index.map({
            1: '16 a 24',
            2: '25 a 34',
            3: '35 a 44',
            4: '45 a 54',
            5: '55 a 64',
            6: '65 a 75',
            7: '>75'
        })

    # Añadir fila resumen
    tabla.loc['Total población'] = [
        df_mod_cluster.loc[
            df_mod_cluster['Cluster'] == 0,
            'Target_Satisfacc_Vida'
        ].mean(),

        df_mod_cluster.loc[
            df_mod_cluster['Cluster'] == 1,
            'Target_Satisfacc_Vida'
        ].mean(),

        df_mod_cluster['Target_Satisfacc_Vida'].mean()
    ]

    return tabla.round(2)


tabla_satisfaccion('Sexo')


tabla_satisfaccion('Terr_hist')


tabla_satisfaccion('Edad_interv')


tabla_satisfaccion('Actividad')


tabla_satisfaccion('Nacionalidad')

