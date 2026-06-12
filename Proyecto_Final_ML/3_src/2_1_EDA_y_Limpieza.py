# El fichero incluye un total de <u>5.284 registros (número total de encuestas realizadas) y 45 columnas.</u>
# La Encuesta de Bienestar Personal ofrece información detallada y puntual sobre la percepción del bienestar por parte de la población de la C.A. de Euskadi, plasmada en <u>nueve indicadores:</u>
# Se incluyen también un conjunto de <u>variables de tipo categórico</u>, relacionadas con características de la población (lugar de residencia, sexo, edad, tipo de familia,...), <u>así como variables que recogen las respuestas dadas por las personas encuestadas que ayudan a entender la situación de la población en términos de bienestar</u> (estado de salud, estado de ánimo,...).
# 
# DATASET A UTILIZAR: *FICHERO DE MICRODATOS DE LA ENCUESTA DE BIENESTAR PERSONAL 2024, EUSTAT*</span> 


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../1_Data/Raw_Microdatos_2024.csv", sep = ";")   

df.head()


df.info()


df.describe(include='all').T

# -----------------------------------------------------------
# *LIMPIEZA DEL DATASET Y TRANSFORMACIÓN DE VARIABLES*

# Elimino un conjunto de variables que:
# 
# - considero no tienen interés para el análisis: aenc, numc, nind, elep, capital, stiem.
# - se encuentran vacías: nivi.
# - o están incluidas/son la base para el cálculo de otras variables de síntesis, por lo que resultarían redundantes: tipf, hablar, ayfam, tenso, moral, calma, depre, feliz, cpol, cjus, certz.
# 
# Genero nuevo dataset de limpieza: df_limp.
# 
# Como resultado de esta primera limpieza quedan 27 variables (de las 45 originales).


#nuevo dataframe eliminando columnas: 
cols_elim = (
    list(range(0, 4)) +   
    [5, 10, 13, 23] +
    list(range(25, 32)) + 
    list(range(33, 36))   
)

df_limp = df.drop(df.columns[cols_elim], axis=1)


df_limp.info()

# ------------------------------------------------------------
#*PASO 2: REETIQUETADO DE VARIABLES*
# 
# Tenemos las columnas de interés seleccionadas en nuevo dataframe: df_limp.
# Procedemos a reetiquetar las variables para saber de qué estamos hablando. 


df_limp.columns.tolist()


# diccionario de renombrado
renombrar = {
    'TH': 'Terr_hist',
    'SEXO' : 'Sexo',
    'EDAD': 'Edad_interv',
    'LNAC': 'Lugar_nacim',
    'NACI': 'Nacionalidad',
    'LEST3': 'Nivel_estudios',
    'RELA': 'Actividad',
    'TGRU': 'Tipo_hogar',
    'SVIDA': 'Target_Satisfacc_Vida',
    'SRELA': 'Satisf_relac_pers',
    'STGUS': 'Satisf_tiempo_disp',
    'SECON': 'Satisf_econom_hogar',
    'SVIVI': 'Satisf_vivienda',
    'SZONA': 'Satisf_zona_resid',
    'SRECR': 'Satisf_zonas_recreat',
    'STRAB': 'Satisf_trabajo',
    'MERECE': 'Sentido_propia_vida',
    'CPER': 'Confianza_personas',
    'SENSE': 'Sensacion_seguridad_zona',
    'SVIVENT': 'Satisf_viv_y_entorno',
    'SHOGTR': 'Satisf_econom_domest',
    'ANIMO': 'Valor_Estado_animo',
    'RELPER': 'Valor_Relac_personales',
    'CPUBLI': 'Confianza_poderes_pub',
    'SITEC1': 'Situacion_econ_subj',
    'SAPER': 'Salud_percibida',
    'RELFA1': 'Intensidad_relac_fam'
}


#renombramos columnas con etiquetas más comprensibles
df_limp = df_limp.rename(columns=renombrar)
df_limp.columns

df_limp.describe().T

ncols = 3

nvars = len(df_limp.columns)

nrows = math.ceil(nvars / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(18, 5*nrows)
)

axes = axes.flatten()

for i, col in enumerate(df_limp.columns):

    # Frecuencias ordenadas
    freq = df_limp[col].value_counts().sort_index()

    sns.barplot(
        x=freq.index,
        y=freq.values,
        ax=axes[i]
    )

    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# *PASO 3: SUSTITUCIÓN DE "NO PROCEDE" POR NULOS EN VARIABLE SATISF_TRABAJO*

# ------------------------------------------------------------
# Se ha identificado un código especial (97) en la variable Satisf_trabajo correspondiente 
# a respuestas no aplicables, que deberá ser tratado antes de la fase de modelado. 
# Momentáneamente sustituyo el 97 (no procede) por nulo.

df_limp['Satisf_trabajo'] = df_limp['Satisf_trabajo'].replace(97, np.nan)

# ------------------------------------------------------------
# *PASO 4: INVERSIÓN DEL SENTIDO DE LA RESPUESTA DE ALGUNAS VARIABLES*
# 
# Hay algunas variables cuyas respuestas siguen un sentido inverso a las valoraciones de satisfacción 
# y confianza en que estamos trabajando, que pueden complicar la interpretación de los datos. 
# Vamos a cambiar el sentido de estas variables para facilitar la comprensión y análisis.
# # Las variables son: 'Sensacion_seguridad_zona', 'Salud_percibida', 'Intensidad_relac_fam'


# cambiar escalas 1-4
df_limp['Salud_percibida'] = 5 - df_limp['Salud_percibida']
df_limp['Intensidad_relac_fam'] = 5 - df_limp['Intensidad_relac_fam']

# cambiar escala 1-3
df_limp['Sensacion_seguridad_zona'] = 4 - df_limp['Sensacion_seguridad_zona']


#para ver las gráficas con los valores invertidos: 

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

variables = [
    'Sensacion_seguridad_zona',
    'Salud_percibida',
    'Intensidad_relac_fam'
]

for i, col in enumerate(variables):

    freq = df_limp[col].value_counts().sort_index()

    sns.barplot(
        x=freq.index,
        y=freq.values,
        ax=axes[i]
    )

    axes[i].set_title(col)
    axes[i].set_xlabel("Valor")
    axes[i].set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# *SELECCIÓN DE VARIABLES PARA CONSTRUCCIÓN DE MODELOS*

# *PASO 1: ANÁLISIS DE LA RELACIÓN ENTRE TODAS LAS VARIABLES*
# 
# Hacemos un heatmap para ver las correlaciones entre todas las variables.

corr = df_limp.corr(numeric_only=True)

plt.figure(figsize=(16, 12))

sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True,
    linewidths=0.5,
    vmin= -1
)

plt.title(
    'Matriz de correlaciones entre variables',
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
#PASO 2: ANÁLISIS DE LA RELACIÓN DE LA TARGET (SATISFACCIÓN GENERAL CON LA VIDA) CON EL RESTO DE VARIABLES


corr_target = (
    df_limp.corr(numeric_only=True)['Target_Satisfacc_Vida']
    .sort_values(ascending=False)
)

corr_target.to_frame(name='Correlacion')


# Grafica de correlaciones con la target:

corr_target = (
    df_limp.corr(numeric_only=True)['Target_Satisfacc_Vida']
    .drop('Target_Satisfacc_Vida')
    .sort_values()
)

colores = []

for valor in corr_target:

    if valor > 0.50:
        colores.append('steelblue')      # fuerte

    elif valor > 0.30:
        colores.append('skyblue')        # moderada

    elif valor >= 0:
        colores.append('#D9EAF4')           # débil 

    else:
        colores.append('indianred')      # negativa

# Figura
plt.figure(figsize=(10,8))

ax = corr_target.plot(
    kind='barh',
    color=colores
)

plt.title(
    'Correlación de las variables con la Satisfacción con la vida',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Coeficiente de correlación')
plt.ylabel('')

for i, valor in enumerate(corr_target.values):

    ax.text(
        valor + 0.01 if valor >= 0 else valor - 0.06,
        i,
        f'{valor:.2f}',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

plt.axvline(0, color='black', linewidth=1)
plt.axvline(0.30, color='grey', linestyle='--', alpha=0.6)
plt.axvline(0.50, color='grey', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------

# Ahora hay que analizar el HEATMAP para ver qué variables pueden ser redundantes y no merecen se incluidas.
#saco una par de gráficas donde se confirma la relación de regresión lineal entre variables:

sns.regplot(
    data=df_limp,
    x='Satisf_relac_pers',
    y='Target_Satisfacc_Vida'
)

plt.show()



sns.regplot(
    data=df_limp,
    x='Sentido_propia_vida',
    y='Target_Satisfacc_Vida'
)

plt.show()

# ------------------------------------------------------------
#PASO 3: SELECCIÓN FINAL DE LAS VARIABLES A INCLUIR EN EL MODELO*


df_limp.describe().T

# Variables elegidas (correlación con target):
# 
# * satisf_relac_pers (0,69)
# * sentido_propia_vida (0,63)
# * satisf_econom_domest (0,53)
# * valor_estado_animo (0,47)
# * satisf_trabajo (0,47)
# * satisf_vivienda (0,43)
# * satisf_tiempo_disp (0,43)
# * satisf_viv_y_entorno (0,41)
# * salud_percibida (0,41)
# * confianza_personas (0,30)
# * confianza_poderes_pub (0,20)
# # 
# Para la creación de los modelos se utilizará, a priori, un total de 11 variables.

# ------------------------------------------------------------
# Las características de la variable denominada  <u>satisf_trabajo </u> obligan a hacer una nueva 
# transformación sobre la misma, que permita que pueda funcionar adecuadamente en el modelo. 
# Esta variable presenta un elevado número de valores nulos (2.816 de 5.284 registros). 
# La decisión tomada es sustituir los valores nulos por el valor medio del resto de registros y 
# volver a repetir el gráfico de correlaciones para comprobar si mantiene una relación elevada con la target.


df_limp['Satisf_trabajo'] = (
    df_limp['Satisf_trabajo']
    .fillna(df_limp['Satisf_trabajo'].mean())
)

df_limp['Satisf_trabajo']


corr_trabajo = df_limp['Satisf_trabajo'].corr(
    df_limp['Target_Satisfacc_Vida']
)

print(f"Correlación: {corr_trabajo:.3f}")

# ------------------------------------------------------------
# La transformación no funciona bien: sustituyendo los valores nulos por el valor medio, 
# Finalmente, opto por no considerar esta variable, ya que resulta problemática.

# ------------------------------------------------------------
#*PASO 4: CREACIÓN DATAFRAME Y CSV PARA MODELO*

# ------------------------------------------------------------
# Procedo a crear el dataframe con el conjunto definitivo de variables a trabajar, 
# así como el archivo CSV que vamos a utilizar para los modelos (processed.csv).


variables_modelo = [

    # Variables demográficas
    'Terr_hist',
    'Sexo',
    'Edad_interv',
    'Lugar_nacim',
    'Nacionalidad',
    'Nivel_estudios',
    'Actividad',
    'Tipo_hogar',

    # Variable objetivo
    'Target_Satisfacc_Vida',

    # Variables predictoras seleccionadas
    'Satisf_relac_pers',
    'Sentido_propia_vida',
    'Satisf_econom_domest',
    'Valor_Estado_animo',
    'Satisf_vivienda',
    'Satisf_tiempo_disp',
    'Satisf_viv_y_entorno',
    'Salud_percibida',
    'Confianza_personas' ,
    'Confianza_poderes_pub'
]

df_modelo = df_limp[variables_modelo].copy()

df_modelo.head()


df_modelo.info()


# Guardo el dataframe final para modelado
df_modelo.to_csv('processed.csv', index=False, encoding='utf-8')
df_modelo.head()

