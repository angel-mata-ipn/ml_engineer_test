import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =======================================================
# Clase 3: ModelEvaluator
# =======================================================
class ModelEvaluator:

    def __init__(self, model, tokenizer, eval_dataset, max_target_length=64, batch_size=8):
        """
        :param model: Modelo a evaluar.
        :param tokenizer: Tokenizer correspondiente.
        :param eval_dataset: Dataset de evaluación (formateado en torch).
        :param max_target_length: Longitud máxima para la generación.
        :param batch_size: Tamaño de batch para la evaluación.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.rouge = evaluate.load("rouge")
        self.predictions = None
        self.references = None
        self.metrics = None

    def generate_predictions(self):
        from transformers import DataCollatorForSeq2Seq
        from tqdm.auto import tqdm  # Importa tqdm para mostrar progreso

        collate_fn = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, padding=True)
        
        dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=0,       # Evita problemas de shared memory en MPS
            pin_memory=False     # No usar pin_memory en MPS
        )
        
        # Configura el dispositivo (MPS si está disponible, de lo contrario CPU)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model.to(device)
        self.model.eval()
        
        all_preds = []
        all_refs = []
        
        # Usa tqdm para mostrar el progreso de la evaluación
        total_batches = len(dataloader)
        for batch in tqdm(dataloader, total=total_batches, desc="Evaluating", leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_target_length,
                num_beams=2,
                early_stopping=True
            )
            outputs_list = outputs.cpu().numpy().tolist()
            decoded_preds = self.tokenizer.batch_decode(outputs_list, skip_special_tokens=True)
            
            labels = batch["labels"].detach().cpu().numpy()
            labels[labels == -100] = self.tokenizer.pad_token_id
            labels_list = labels.tolist()
            decoded_refs = self.tokenizer.batch_decode(labels_list, skip_special_tokens=True)
            
            all_preds.extend(decoded_preds)
            all_refs.extend(decoded_refs)
        
        self.predictions = all_preds
        self.references = all_refs
        return all_preds, all_refs

    def evaluate(self):
        preds, refs = self.generate_predictions()
        self.metrics = self.rouge.compute(predictions=preds, references=refs)
        print("Métricas globales ROUGE:", self.metrics)
        return self.metrics

    def compute_individual_scores(self):
        if self.predictions is None or self.references is None:
            self.generate_predictions()
        scores = []
        for pred, ref in zip(self.predictions, self.references):
            result = self.rouge.compute(predictions=[pred], references=[ref])
            scores.append(result["rougeL"])
        return scores

    def plot_scores_distribution(self):
        scores = self.compute_individual_scores()
        plt.figure(figsize=(10,6))
        sns.histplot(scores, bins=20, color="skyblue", edgecolor="black")
        plt.title("Distribución de ROUGE-L por ejemplo")
        plt.xlabel("ROUGE-L")
        plt.ylabel("Frecuencia")
        plt.show()

    def show_low_score_examples(self, n=5):
        scores = self.compute_individual_scores()
        indices = np.argsort(scores)
        print(f"\nEjemplos con los {n} menores scores ROUGE-L:")
        for idx in indices[:n]:
            print(f"Ejemplo {idx} - ROUGE-L: {scores[idx]:.4f}")
            print("Predicción:", self.predictions[idx])
            print("Referencia:", self.references[idx])
            print("-" * 80)

    def get_recommendations(self):
        if self.metrics is None:
            self.evaluate()
        recommendations = []
        rougeL = self.metrics.get("rougeL", 0)
        if rougeL > 0.5:
            recommendations.append("El modelo genera resúmenes de alta calidad, capturando la mayoría de los detalles clave.")
            recommendations.append("Se recomienda integrar este sistema en el servicio al cliente para agilizar la información.")
        else:
            recommendations.append("El modelo presenta deficiencias en capturar los detalles importantes.")
            recommendations.append("Se recomienda ajustar el fine-tuning o explorar modelos alternativos.")
        return recommendations

    def print_recommendations(self):
        recs = self.get_recommendations()
        print("Recomendaciones de Negocio:")
        for rec in recs:
            print("- " + rec)