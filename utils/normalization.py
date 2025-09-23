import pandas as pd
import os

TRAIN_FILES = [
    "../ACNE04/NNEW_trainval_0.txt",
    "../ACNE04/NNEW_trainval_1.txt",
    "../ACNE04/NNEW_trainval_2.txt",
    "../ACNE04/NNEW_trainval_3.txt",
    "../ACNE04/NNEW_trainval_4.txt",
    "../ACNE04/NNEW_trainval_5.txt",
]

TEST_FILES = [
    "../ACNE04/NNEW_test_0.txt",
    "../ACNE04/NNEW_test_1.txt",
    "../ACNE04/NNEW_test_2.txt",
    "../ACNE04/NNEW_test_3.txt",
    "../ACNE04/NNEW_test_4.txt"
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
