from src.pipeline import FullPipeline

if __name__ == "__main__":
    # Parámetros de entrada
    csv_path = "data/cleaned_data.csv"  # Ruta a tu CSV preprocesado
    input_col = "news_body"         # Columna con el texto a resumir
    target_col = "news_header"      # Columna con el resumen (target)
    model_checkpoint = ""
    max_input_length = 1024
    max_target_length = 64
    test_size = 0.1

    # Crear e iniciar el pipeline completo
    pipeline = FullPipeline(csv_path, input_col, target_col, model_checkpoint,
                            max_input_length, max_target_length, test_size)
    pipeline.run_pipeline()