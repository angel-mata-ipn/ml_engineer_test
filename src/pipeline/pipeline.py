from src.data_preprocessing import DataPreprocessor
from src.model import TransformerFineTuner
from src.evaluation import ModelEvaluator


# =======================================================
# Clase 4: FullPipeline (con evaluación pre y post fine-tuning)
# =======================================================
class FullPipeline:
    def __init__(self, csv_path, input_col, target_col, model_checkpoint,
                 max_input_length=1024, max_target_length=64, test_size=0.1):
        """
        :param csv_path: Ruta al CSV.
        :param input_col: Columna con el texto a resumir.
        :param target_col: Columna con el resumen (target).
        :param model_checkpoint: Modelo preentrenado de HuggingFace.
        :param max_input_length: Longitud máxima para el input.
        :param max_target_length: Longitud máxima para el target.
        :param test_size: Fracción del dataset para evaluación.
        """
        self.csv_path = csv_path
        self.input_col = input_col
        self.target_col = target_col
        self.model_checkpoint = model_checkpoint
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.test_size = test_size

        self.preprocessor = DataPreprocessor(csv_path, input_col, target_col)
        self.tuner = None
        self.evaluator = None
        self.baseline_metrics = None
        self.post_tuning_metrics = None

    def run_pipeline(self):
        # Preprocesamiento
        print("=== Preprocesamiento y Exploración ===")
        self.preprocessor.load_data()
        self.preprocessor.apply_cleaning()
        self.preprocessor.remove_duplicates()
        cleaned_df = self.preprocessor.get_cleaned_data()
        # Crear columnas finales (se usan las mismas columnas limpiadas)
        cleaned_df["final_" + self.input_col] = cleaned_df[self.input_col]
        cleaned_df["final_" + self.target_col] = cleaned_df[self.target_col]

        # Fine-Tuning del Modelo
        print("\n=== Preparación para Fine-Tuning ===")
        self.tuner = TransformerFineTuner(
            df=cleaned_df,
            input_col="final_" + self.input_col,
            target_col="final_" + self.target_col,
            model_checkpoint=self.model_checkpoint,
            max_input_length=self.max_input_length,
            max_target_length=self.max_target_length,
            test_size=self.test_size
        )
        self.tuner.prepare_dataset()
        self.tuner.split_dataset()
        self.tuner.setup_trainer()

        # Evaluación del modelo preentrenado (baseline)
        print("\n=== Evaluación del Modelo Preentrenado (Baseline) ===")
        baseline_evaluator = ModelEvaluator(
            model=self.tuner.model,
            tokenizer=self.tuner.tokenizer,
            eval_dataset=self.tuner.eval_dataset,
            max_target_length=self.max_target_length,
            batch_size=8
        )
        self.baseline_metrics = baseline_evaluator.evaluate()
        baseline_evaluator.print_recommendations()
        baseline_evaluator.plot_scores_distribution()
        baseline_evaluator.show_low_score_examples(n=3)

        # Fine-Tuning
        print("\n=== Fine-Tuning del Modelo ===")
        self.tuner.train_model()

        # Evaluación del modelo fine-tuneado
        print("\n=== Evaluación del Modelo Fine-Tuneado ===")
        post_evaluator = ModelEvaluator(
            model=self.tuner.model,
            tokenizer=self.tuner.tokenizer,
            eval_dataset=self.tuner.eval_dataset,
            max_target_length=self.max_target_length,
            batch_size=64
        )
        self.post_tuning_metrics = post_evaluator.evaluate()
        post_evaluator.print_recommendations()
        post_evaluator.plot_scores_distribution()
        post_evaluator.show_low_score_examples(n=3)

        print("\n=== Comparación de Métricas ===")
        print("Métricas Baseline:", self.baseline_metrics)
        print("Métricas Fine-Tuneado:", self.post_tuning_metrics)