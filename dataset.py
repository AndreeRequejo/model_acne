# dataset.py
import pandas as pd
import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

class ClassificationDataset:
    def __init__(self, data, data_path, transform, training=True):
        """Definir el conjunto de datos para la clasificación

        Args:
            data ([dataframe]): [un dataframe que contiene 2 columnas: nombre de la imagen y etiqueta]
            data_path ([str]): [ruta/a/carpeta que contiene el archivo de imagen]
            transform : [métodos de aumento y transformación de imágenes]
            training (bool, optional): []. Default True.
        """
        self.data = data
        self.imgs = data["path"].unique().tolist()
        self.data_path = data_path
        self.training = training
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_path, self.data.iloc[idx, 0])) # Carga imagen
        label = self.data.iloc[idx, 1] # Extrae etiqueta
        if self.transform is not None:
            img = self.transform(img) # Convierte a tensor 224x224x3
        return img, label # Retorna imagen como tensor y etiqueta

    def __len__(self):
        return len(self.imgs)


def make_loader(dataset, train_batch_size, validation_split=0.2):
    """Crear train_loader y test_loader a partir del dataset

    Args:
        dataset ([object]):
        train_batch_size ([int]):
        validation_split (float, opcional): Default to 0.2.

    Returns:
        [type]: [description]
    """
    # Número de muestras en conjuntos de entrenamiento y prueba
    train_len = int(len(dataset) * (1 - validation_split))
    test_len = len(dataset) - train_len
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
    # Crear cargador de entrenamiento
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True,
    )
    # Crear cargador de prueba
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,)
    return train_loader, test_loader


def data_split(data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(
        data, data["label"], test_size=test_size, stratify=data.iloc[:,1]
    )
    return x_train, x_test, y_train, y_test


def load_data(train_file, test_file):
    """Cargar datos de entrenamiento y prueba desde archivos"""
    train_df = pd.read_csv(train_file, names=['path','label','leisons'], sep=' ', engine='python')
    test_df = pd.read_csv(test_file, names=['path','label','leisons'], sep=' ', engine='python')
    return train_df, test_df