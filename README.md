# ml_engineer_test

Este repositorio contiene un pipeline completo para entrenar y evaluar un modelo de summarization en español, usando datos del archivo apli_challenge_data.csv. Se incluyen scripts para preprocesamiento, configuración del modelo, evaluación de métricas (ROUGE) y recomendaciones de negocio.

### Estructura del Proyecto
.
├── data/
│   └── apli_challenge_data.csv         # Datos originales sin procesar
├── notebooks/
│   └── 1_EDA.ipynb                     # Análisis exploratorio de datos
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Clase para limpieza y transformación de textos
│   ├── model.py                        # Clases y funciones relacionadas con el modelo
│   ├── evaluation.py                   # Clases y funciones para evaluación
│   └── pipeline/
│       └── pipeline.py                 # Pipeline principal (preprocesamiento -> modelo -> evaluación)
├── results/
│   ├── testing_pipeline_v1.ipynb       # Pruebas en Colab, modelo 1
│   ├── testing_pipeline_v2.ipynb       # Pruebas en Colab, modelo 2
│   └── testing_pipeline_v3.ipynb       # Pruebas en Colab, modelo 2 con 2 épocas
├── main.py                             # Punto de entrada para ejecutar el pipeline
├── requirements.txt                    # Dependencias de Python
└── README.md                           # (Este archivo)


### Requisitos
Python 3.8+
Asegúrate de contar con una versión reciente de Python.

### Entorno Virtual
Se recomienda usar un virtualenv o conda (por ejemplo, la carpeta ml_test/ podría ser tu entorno virtual).

bash

python -m venv ml_test
source ml_test/bin/activate  # Linux/Mac
ml_test\Scripts\activate     # Windows
Dependencias
Instala las dependencias listadas en requirements.txt:

bash

pip install -r requirements.txt
API Keys

Hugging Face Access Token: Si usas un modelo privado o necesitas acceso a ciertas funcionalidades, requerirás un token. Inicia sesión en Hugging Face (huggingface-cli login) o configura tu token en las variables de entorno.

Weaviate / WeDB Key: En caso de que el proyecto requiera conectarse a Weaviate o WeDB para almacenar vectores o metadatos, necesitas la respectiva clave de API o credenciales. Define la variable de entorno WENDB_API_KEY (o similar) con tu token.

Uso
Ejecutar el pipeline
Lanza el pipeline completo desde la línea de comandos:

bash

python main.py
Este script cargará los datos, ejecutará el preprocesamiento, entrenará (o ajustará) el modelo y realizará la evaluación final.

Notebooks de Ejemplo

1_EDA.ipynb: Muestra un análisis exploratorio inicial.

testing_pipeline_v*.ipynb: Contienen experimentos ejecutados en Google Colab, incluyendo resultados y métricas de ROUGE.

Configurar tu Propio Modelo
Si deseas usar un modelo diferente (por ejemplo, un frontier model como gpt-4o-mini-2024-07-18), ajusta el nombre del checkpoint en src/model.py o en src/pipeline/pipeline.py, y asegura que tu token de Hugging Face tenga permisos de acceso a ese repositorio.

##### Próximos Pasos / Consideraciones
Integración de MLflow
Para trackear experimentos, versiones de modelo y métricas, se recomienda integrar MLflow o una herramienta similar.

##### Fine-Tuning en un Frontier Model
Si buscas mayor eficiencia o mejores resultados, considera realizar el fine-tuning sobre un modelo como gpt-4o-mini-2024-07-18. Ajusta las clases en model.py y pipeline.py para cargar el nuevo modelo.

##### Optimización de Hiperparámetros
Experimenta con el número de épocas, learning rate, tamaños de batch, etc., para mejorar la calidad del resumen.

##### Manejo de Datos Sensibles
Si tu dataset contiene información personal, revisa la configuración de privacidad y asegúrate de cumplir con regulaciones como GDPR.