import pandas as pd
import os

# Obtener la ruta absoluta de la carpeta base (raíz del proyecto)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_FILES = [
    os.path.join(BASE_DIR, "ACNE04", f"NNEW_trainval_{i}.txt") for i in range(5)
]

TEST_FILES = [
    os.path.join(BASE_DIR, "ACNE04", f"NNEW_test_{i}.txt") for i in range(5)
]

def normalize_labels(file_list, prefix="NORMALIZED"):
    for file in file_list:
        # Leer archivo
        df = pd.read_csv(file, sep="\s+", header=None, names=['path','label','lesions'])

        # Normalizar (clase 3 → 2)
        df["label"] = df["label"].replace(3, 2)

        # Generar nombre de salida (misma carpeta, con prefijo)
        folder, filename = os.path.split(file)
        new_filename = os.path.join(folder, f"{prefix}_{filename}")

        # Guardar resultado
        df.to_csv(new_filename, sep=" ", index=False, header=False)

        print(f"✅ Procesado: {file} → {new_filename}")

# Procesar todos
normalize_labels(TRAIN_FILES, prefix="NORMALIZED")
normalize_labels(TEST_FILES, prefix="NORMALIZED")
