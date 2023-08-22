
# ICR - Identifying Age-Related Conditions


## Objetivo de la competición

El objetivo de esta competencia es predecir si una persona tiene alguna de tres condiciones médicas. Se te pide que predigas si la persona tiene una o más de estas tres condiciones médicas (Clase 1), o ninguna de las tres condiciones médicas (Clase 0). Crearás un modelo entrenado en medidas de características de salud.

Para determinar si alguien tiene estas condiciones médicas, se requiere un proceso largo e intrusivo para recopilar información de los pacientes. Con modelos predictivos, podemos acortar este proceso y mantener los detalles de los pacientes privados al recopilar características clave relacionadas con las condiciones y luego codificar estas características.

Tu trabajo ayudará a los investigadores a descubrir la relación entre las medidas de ciertas características y las posibles condiciones de los pacientes.

## Contexto

Dicen que la edad es solo un número, pero envejecer conlleva una serie de problemas de salud. Desde enfermedades cardíacas y demencia hasta pérdida de audición y artritis, el envejecimiento es un factor de riesgo para numerosas enfermedades y complicaciones. El campo creciente de la bioinformática incluye investigaciones sobre intervenciones que pueden ayudar a frenar y revertir el envejecimiento biológico y prevenir enfermedades importantes relacionadas con la edad. La ciencia de datos podría tener un papel en el desarrollo de nuevos métodos para resolver problemas con datos diversos, incluso si el número de muestras es pequeño.

Actualmente, se utilizan modelos como XGBoost y random forest para predecir condiciones médicas, pero el rendimiento de los modelos no es lo suficientemente bueno. Al enfrentar problemas críticos en los que están en juego vidas humanas, los modelos deben hacer predicciones correctas de manera confiable y consistente en diferentes casos.

Fundada en 2015, InVitro Cell Research, LLC (ICR), la organización anfitriona de la competencia, es una empresa financiada de forma privada centrada en la medicina regenerativa y preventiva personalizada. Sus oficinas y laboratorios en el área metropolitana de la ciudad de Nueva York ofrecen un espacio de investigación de vanguardia. Los científicos de InVitro Cell Research son lo que los distingue, ayudando a guiar y definir su misión de investigar cómo reparar rápidamente a las personas que envejecen.

En esta competencia, trabajarás con datos de características de salud para resolver problemas críticos en bioinformática. Basándote en un entrenamiento mínimo, crearás un modelo para predecir si una persona tiene alguna de tres condiciones médicas, con el objetivo de mejorar los métodos existentes.

Podrías ayudar a avanzar en el campo creciente de la bioinformática y explorar nuevos métodos para resolver problemas complejos con datos diversos.

## Evaluación

Las presentaciones se evalúan utilizando una pérdida logarítmica balanceada. El efecto general es tal que cada clase tiene aproximadamente la misma importancia para la puntuación final.

Cada observación pertenece a la clase 0 o a la clase 1. Para cada observación, debes enviar una probabilidad para cada clase.


El Log Loss se calcula mediante la siguiente fórmula:

![formula](src/utils/loss_function.png)

Donde:
- `N_0` es el número de observaciones de la clase 0.
- `N_1` es el número de observaciones de la clase 1.
- `y_{0i}` es 1 si la observación i pertenece a la clase 0, y 0 en caso contrario.
- `y_{1i}` es 1 si la observación i pertenece a la clase 1, y 0 en caso contrario.
- `p_{0i}` es la probabilidad predicha de que la observación i pertenezca a la clase 0.
- `p_{1i}` es la probabilidad predicha de que la observación i pertenezca a la clase 1.


## Dataset Description
The competition data comprises over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions -- a binary classification problem.

Note that this is a Code Competition, in which the actual test set is hidden. In this version, we give some sample data in the correct format to help you author your solutions. When your submission is scored, this example test data will be replaced with the full test set. There are about 400 rows in the full test set.

### Files and Field Descriptions
- **train.csv** - The training set.
  - `Id Unique` identifier for each observation.
  - `AB-GL` Fifty-six anonymized health characteristics. All are numeric except for EJ, which is categorical.
  - `Class` A binary target: 1 indicates the subject has been diagnosed with one of the three conditions, 0 indicates they have not.
- **test.csv** - The test set. Your goal is to predict the probability that a subject in this set belongs to each of the two classes.
- **greeks.csv** - Supplemental metadata, only available for the training set.
  - `Alpha` Identifies the type of age-related condition, if present.
    - `A` No age-related condition. Corresponds to class 0.
    - `B, D, G` The three age-related conditions. Correspond to class 1.
  - `Beta, Gamma, Delta` Three experimental characteristics.
  - `Epsilon` The date the data for this subject was collected. Note that all of the data in the test set was collected after the training set was collected.
- sample_submission.csv - A sample submission file in the correct format. See the Evaluation page for more details.

## Timeline

- **Mayo 11, 2023**: Fecha de inicio.
- **Agosto 3, 2023**: Fecha límite de inscripción. Debes aceptar las reglas del concurso antes de esta fecha para poder competir.
- **Agosto 3, 2023**: Fecha límite para la fusión de equipos. Este es el último día en el que los participantes pueden unirse o fusionar equipos.
- **Agosto 10, 2023**: Fecha límite de envío final.

Todas las fechas límite son a las 11:59 PM UTC del día correspondiente, a menos que se indique lo contrario. Los organizadores del concurso se reservan el derecho de actualizar la línea de tiempo del concurso si lo consideran necesario.

## This is a Code Competition

Las presentaciones para esta competencia deben realizarse a través de Notebooks. Para que el botón "Enviar" esté activo después de confirmar, se deben cumplir las siguientes condiciones:

- `Notebook de CPU`: tiempo de ejecución máximo de 9 horas.
- `Notebook de GPU`: tiempo de ejecución máximo de 9 horas.
-  Acceso a Internet desactivado.
-  El archivo de presentación debe llamarse submission.csv.

Por favor, consulta las preguntas frecuentes de la competencia de código para obtener más información sobre cómo enviar tu trabajo. Además, revisa el documento de depuración de código si encuentras errores al realizar la presentación.

## Code Competitions - Errors & Debugging Tips

### Goose with a question

Estás obteniendo un error en una competencia de código. ¿Y ahora qué? Es difícil escribir código que funcione perfectamente con datos no vistos, incluso para los expertos. No te desanimes ni pienses que eres el único que está atascado.

Para evitar la exploración, Kaggle no proporciona mensajes de depuración altamente específicos en las competencias de código (donde Kaggle vuelve a ejecutar tu código en un conjunto de datos oculto). Las presentaciones que producen errores también se cuentan dentro del límite diario de presentaciones de tu equipo, de lo contrario, dichas presentaciones podrían utilizarse para extraer información oculta. Sin embargo, proporcionamos un tipo general de error para orientarte en la dirección correcta.

Todas las ejecuciones siguen el mismo flujo básico. Primero, ejecutamos en privado tu notebook de principio a fin con un conjunto de datos de la competencia reemplazado por una versión oculta con datos diferentes. Luego, validamos esa ejecución frente a las restricciones de la competencia y extraemos el archivo de presentación desde el directorio de salida. Finalmente, evaluamos ese archivo de presentación. Cada uno de estos pasos puede tener errores.

A continuación se muestran los tipos de errores que tu notebook podría recibir. Recuerda que el hecho de que dos notebooks compartan el mismo tipo de error no implica que tengan la misma causa raíz.

- `Notebook Timeout`: Tu notebook de presentación excedió el tiempo de ejecución permitido. Revisa la página de Requisitos de código de la competencia para conocer los límites de tiempo. Ten en cuenta que el conjunto de datos oculto puede ser más grande, más pequeño o diferente al conjunto de datos público.
- `Notebook Threw Exception`: Durante la ejecución de tu código, tu notebook encontró un error no controlado. Ten en cuenta que el conjunto de datos oculto puede ser más grande, más pequeño o diferente al conjunto de datos público.
- `Notebook Exceeded Allowed Compute`: Esto indica que has violado una restricción de requisito de código durante la ejecución. Esto incluye limitaciones en el entorno de ejecución, como solicitar más RAM o espacio en disco del disponible, o restricciones de la competencia, como el tipo de origen de datos de entrada o los límites de tamaño.
- `Submission CSV Not Found`: Tu notebook no generó el archivo de presentación esperado (generalmente submission.csv). La ejecución de tu notebook parece haberse completado, pero cuando buscamos el archivo de presentación, no estaba allí. Esto significa que es posible que haya errores o problemas aguas arriba que detuvieron la ejecución y evitaron la escritura del archivo. Intentar leer un directorio/archivo inexistente de un conjunto de datos es una razón común por la que se detiene la ejecución (causando Submission CSV Not Found o Notebook Threw Exception).
- `Submission Scoring Error`: Tu notebook generó un archivo de presentación con un formato incorrecto. Algunos ejemplos de esto son: número incorrecto de filas o columnas, valores vacíos, un tipo de dato incorrecto para un valor o valores de presentación inválidos en comparación con lo esperado.
- `Kaggle Error`: Un error del sistema poco común. Por favor, intenta enviar de nuevo para resolver el error y ponte en contacto con el Soporte de Kaggle si persiste.

Si estás atascado, te animamos a aplicar los mismos pasos de depuración que harías cuando tienes una traza completa del código. Aquí tienes algunos consejos para prevenir errores y desbloquearte:

- Respira profundamente, aléjate del código, duerme o da un paseo, distrae tu mente y luego vuelve a examinarlo con ojos frescos.
- Utiliza una presentación para verificar si los fundamentos básicos están funcionando. Por ejemplo, prueba si hay un modelo preentrenado presente intentando cargarlo y envía la presentación de muestra si tiene éxito, o genera un error si falla. También puedes idear formas de comunicarte contigo mismo a través de la puntuación o el tiempo de tu presentación (por ejemplo, enviar intencionalmente una presentación muy mala o usar la función sleep() durante un tiempo determinado).
- Puedes utilizar decoradores de funciones (ver aquí un ejemplo) para hacer que tu código sea más robusto. Trabaja hacia algo que falle de manera elegante y siga produciendo una puntuación, y luego analiza las puntuaciones para determinar con qué frecuencia crees que el código podría estar fallando de manera elegante.
- Incorpora verificaciones de coherencia en tu pipeline, como verificar si la presentación que produces tiene el mismo número de líneas que la presentación de muestra. ¿Está todo lo que debería ser positivo realmente positivo? etc.
- Siempre que leas un archivo, considera el caso en el que pueda no existir.
- A veces es más fácil empezar con la presentación de muestra, en lugar de intentar crear una presentación directamente a partir del conjunto de pruebas. Es fácil perder una Id entre los bucles, joins, agrupaciones y procesamiento de datos.
- Si todo lo demás falla, desmonta tu pipeline y reconstrúyelo de una manera que comience con una presentación válida, como enviar la presentación de muestra, agregar componentes uno a la vez y verificar que la salida aún tenga una puntuación. Es muy común que el error sea trivial o se encuentre al principio del código, y una vez que lo corriges, todo el pipeline vuelve a funcionar.

