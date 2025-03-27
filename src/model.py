import os
import nltk
import torch

nltk.download("stopwords")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dotenv import load_dotenv

# environment

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
hf_token = os.environ['HF_TOKEN']
print(hf_token)
# =======================================================
# Clase 2: TransformerFineTuner
# =======================================================
class TransformerFineTuner:
    def __init__(self, df, input_col, target_col, model_checkpoint, max_input_length=512, max_target_length=64, test_size=0.1):
        """
        :param df: DataFrame limpio.
        :param input_col: Columna con el texto a resumir.
        :param target_col: Columna con el resumen (target).
        :param model_checkpoint: Modelo preentrenado de HuggingFace.
        :param max_input_length: Longitud máxima para el input.
        :param max_target_length: Longitud máxima para el target.
        :param test_size: Fracción para el conjunto de evaluación.
        """
        self.df = df[[input_col, target_col]].copy()
        self.input_col = input_col
        self.target_col = target_col
        self.model_checkpoint = model_checkpoint
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.test_size = test_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        self.dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None

    def prepare_dataset(self):
        """Convierte el DataFrame en un Dataset de HuggingFace y tokeniza los ejemplos."""
        self.dataset = Dataset.from_pandas(self.df)
        
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples[self.input_col],
                max_length=self.max_input_length,
                truncation=True
            )
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    examples[self.target_col],
                    max_length=self.max_target_length,
                    truncation=True
                )
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        remove_cols = [self.input_col, self.target_col, "__index_level_0__"]
        self.dataset = self.dataset.remove_columns([col for col in remove_cols if col in self.dataset.column_names])
        self.dataset.set_format("torch")
        print("Dataset tokenizado y preparado.")
    
    def split_dataset(self):
        """Divide el dataset en entrenamiento y evaluación."""
        split_ds = self.dataset.train_test_split(test_size=self.test_size)
        self.train_dataset = split_ds["train"]
        self.eval_dataset = split_ds["test"]
        print(f"Dataset dividido: Entrenamiento = {len(self.train_dataset)} ejemplos, Evaluación = {len(self.eval_dataset)} ejemplos.")
    
    def setup_trainer(self, output_dir="./results", num_train_epochs=1, per_device_train_batch_size=4, per_device_eval_batch_size=4, learning_rate=2e-5):
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            logging_steps=50,
            fp16=False  # Desactivamos fp16 para MPS o Mac
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        print("Trainer configurado.")
    
    def train_model(self):
        if self.trainer is None:
            raise ValueError("Trainer no configurado. Llama a setup_trainer() primero.")
        print("Iniciando entrenamiento...")
        self.trainer.train()
    
    def evaluate_model(self):
        if self.trainer is None:
            raise ValueError("Trainer no configurado. Llama a setup_trainer() primero.")
        results = self.trainer.evaluate()
        print("Resultados de evaluación:", results)
        return results
    
