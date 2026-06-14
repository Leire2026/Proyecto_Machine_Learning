# Proyecto de Machine Learning:

## ¿Podemos predecir la satisfacción con la vida? Aplicación de técnicas de Machine Learning a datos de bienestar personal en Euskadi

## Objetivo del proyecto: 

Analizar los <u>factores que influyen en la satisfacción con la vida de las personas residentes en Euskadi, y construir un modelo capaz de predecir el nivel de satisfacción vital</u> a partir de variables sociodemográficas, económicas y de bienestar personal.   


## <span style="color:green; font-size:25px; font-weight:bold;"> Fuentes de Información:  </span>

Para el trabajo utilizaremos como base la <u>Encuesta de Bienestar Personal del Eustat</u> [(link)](https://es.eustat.eus/estadisticas/tema_509/opt_0/tipo_11/ti_bienestar-personal/temas.html#el)  . Se trata de una encuesta de carácter anual realizada en el ámbito del País Vasco. Se incluye dentro de una operación estadística superior, la Encuesta de Condiciones de Vida, si bien nos quedamos únicamente con este módulo, que se centra en aspectos relacionados con el bienestar personal y la satisfacción con diversos aspectos de la vida.

El dataset a utilizar es el  <u>fichero de microdatos</u> relativo a la operación estadística realizada en 2024, última encuesta disponible, publicada en abril de 2025. 

¿Qué es un fichero de microdatos?: es el fichero donde se recoge en bruto toda la información recogida en el proceso de encuestación, que se utilizará con posterioridad para su explotación y extracción de resultados y diversas estadísticas.


## <span style="color:green; font-size:23px; font-weight:bold;"> Trabajo a realizar:  </span> 

El objetivo es llevar a cabo un análisis de los datos que nos permita aproximarnos a los factores que más influyen en el estado de bienestar de las personas en Euskadi, y diseñar un modelo que ayude a predecir ese nivel de satisfacción vital a partir de variables sociodemográficas, económicas y de bienestar personal. 

En este sentido, se plantea un <u>doble objetivo:</u> 

<span style="color:green; font-size:16px; font-weight:bold;"> un Objetivo Explicativo:  </span> "Identificar qué factores están más asociados a la satisfacción vital"

<span style="color:green; font-size:16px; font-weight:bold;"> y un Objetivo Predictivo:  </span>
"Construir el mejor modelo posible para predecir el estado de bienestar personal"

Para ello, los <u>pasos a seguir</u> serán:

1. Entender el problema y los objetivos.
2. Analizar los datos.
3. Limpiar y preparar los datos.
4. Explorar los datos (EDA).
5. Seleccionar variables.
6. Entrenar modelos.
7. Evaluar resultados.
8. Elegir el  mejor modelo que mejor se ajuste al objetivo.
9. Interpretar el modelo.
10. Documentar y presentar el proyecto.  

## <span style="color:green; font-size:23px; font-weight:bold;"> Principales conclusiones:  </span> 

Tras el análisis de los datos y el proceso de construcción de los diferentes modelos realizado, encontramos que:  
- Los modelos de regresión permitieron identificar los principales factores asociados a la satisfacción vital. 
- Los modelos de clasificación evaluaron la capacidad predictiva para identificar individuos con elevados niveles de bienestar.
- Y el análisis de clustering permitió descubrir perfiles naturales de bienestar dentro de la población analizada.

Esto nos conduce a elegir no uno, sino tres modelos, que responde a tres preguntas distintas:

1. El modelo de Regresión Lineal (sin incluir variables sociodemográficas), que responde a la pregunta: ¿cuáles son los principales factores que explican el estado de bienestar y satisfacción general de la población en Euskadi?

    Y la respuesta es interesante, porque frente a lo que podríamos esperar, no son los elementos materiales y objetivos (como pueden ser la situación económica familiar, la satisfacción con la vivienda o el estado de salud) aquellos a los que las personas confieren más valor a la hora de calibrar el nivel de satisfacción en su vida. Por el contrario los factores a los que las personas conceden más importancia y contribuyen en mayor medida a ese estado de bienestar subjetivo son dos: las relaciones personales y el sentido de la propia vida, la percepción de que lo que hacemos en la vida tiene un sentido (el famoso "ikigai"), aspectos ambos no tangibles que apelan a la parte trascendente e inmaterial de la vida humana.

2. El modelo de clasificación elegido, en este caso, el Random Forest Classifier (en la versión ajustada), que responde a la pregunta: ¿podemos predecir el nivel de bienestar de las personas?

    El modelo destaca por su capacidad predictiva, concretamente por su elevada capacidad para identificar correctamente a los individuos pertenecientes al grupo objetivo, es decir, aquéllos con un elevado nivel de bienestar personal.

    Este modelo se acompaña con la creación de una herramienta en Streamlit para la identificación de personas con bajo o alto nivel de bienestar personal.

3. El modelo no supervisado elegido, esto es, el Cluster kmeans, que responde a la pregunta: ¿podemos segmentar/agrupar la población en función de su nivel de bienestar y satisfacción con la vida?

    El modelo permite separar dos grandes grupos de personas, que encajan con los dos grandes perfiles previamente identificados en los modelos de clasificación. La principal virtud de este modelo es que ha permitido identificar perfiles naturales de bienestar en la población sin utilizar la variable objetivo. Esta segmentación emerge así de manera completamente no supervisada, sin incorporar la variable objetivo utilizada en los modelos de clasificación, hallazgo que aporta evidencia adicional sobre la existencia de dos grandes perfiles de bienestar subjetivo en la población estudiada y refuerza la coherencia de los resultados obtenidos a lo largo del proyecto mediante diferentes aproximaciones analíticas.


Tres modelos, tres productos diferentes que permiten plantearse tres alternativas de negocio que podemos dirigir a diferentes públicos: organismos públicos, empresas de gestión de personas, organismos estadísticos, ...


