from nltk.corpus import stopwords
import pandas as pd
import re

# Descomentar la linea de abajo en caso de que no tenga descargadas las stopwords
# nltk.download("stopwords")

# =======================================================
# Clase 1: DataPreprocessorc
# =======================================================

OUTPUT_PATH = "data/cleaned_data.csv"

class DataPreprocessor:
    def __init__(self, csv_path, input_col, target_col):
        """
        :param csv_path: Ruta al archivo CSV.
        :param input_col: Nombre de la columna con el texto a resumir.
        :param target_col: Nombre de la columna con el resumen (target).
        """
        self.csv_path = csv_path
        self.input_col = input_col
        self.target_col = target_col
        self.df = None
        self.cleaned_df = None
        self.stopwords = set(stopwords.words("spanish"))
    
    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        print("Dataset cargado. Shape:", self.df.shape)
        return self.df
    
    def clean_text(self, text):
        """Limpieza básica: minúsculas, eliminación de etiquetas HTML y caracteres no alfanuméricos."""
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-záéíóúñü0-9\s]', '', text)
        return text
    
    def apply_cleaning(self):
        if self.df is None:
            self.load_data()
        self.cleaned_df = self.df.copy()
        self.cleaned_df[self.input_col] = self.cleaned_df[self.input_col].apply(self.clean_text)
        self.cleaned_df[self.target_col] = self.cleaned_df[self.target_col].apply(self.clean_text)
        print("Limpieza de texto aplicada.")
        return self.cleaned_df
    
    def remove_duplicates(self):
        if self.cleaned_df is None:
            self.apply_cleaning()
        original_shape = self.cleaned_df.shape
        self.cleaned_df = self.cleaned_df.drop_duplicates(subset=[self.input_col, self.target_col])
        print(f"Duplicados eliminados. Original: {original_shape}, Nuevo: {self.cleaned_df.shape}")
        return self.cleaned_df
    
    def get_cleaned_data(self):
        if self.cleaned_df is None:
            self.apply_cleaning()
        return self.cleaned_df
    
