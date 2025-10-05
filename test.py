import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Importar módulos locales
from config import TEST_FILES, IMAGE_PATH, TEST_TRANSFORM, CLASS_NAMES, MODEL_SAVE_PATH
from dataset import ClassificationDataset
from model import MyNet

if __name__ == "__main__":
    import sys
    
    # Usar nombre de modelo desde argumentos o default
    model_name = sys.argv[1] if len(sys.argv) > 1 else MODEL_SAVE_PATH
    
    print(f"Testing modelo: {model_name}")
    
    # Preparar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    weights_path = model_name.replace('.pt', '_weights.pt')
    
    try:
        # Crear modelo y cargar solo los pesos
        model = MyNet().cuda()
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model = model.to(device)
    except:
        # Si falla, usar el método original
        model = torch.load(model_name, weights_only=False)
        model = model.to(device)
    
    # Cargar datos de test
    print("Cargando datos de test...")
    test_df = pd.read_csv(TEST_FILES[5], names=['path','label','leisions'], sep=' ', engine='python')
    testset = ClassificationDataset(test_df, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    print(f"{len(test_df)} muestras de test cargadas")
    
    # Evaluar modelo
    model.eval()
    predictions = []
    true_labels = []
    
    print("Evaluando modelo...")
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = torch.argmax(output, axis=1).item()
            
            predictions.append(pred)
            true_labels.append(target.item())
    
    # Convertir a array numpy
    true_labels = np.array(true_labels)
    
    # Calcular métricas
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Mostrar reporte de clasificación
    print("\n" + "="*50)
    print("Reporte de clasificación:")
    print("="*50)
    print(classification_report(true_labels, predictions, target_names=CLASS_NAMES))
    print("="*50)

    # Calcular y mostrar matriz de confusión
    cm = confusion_matrix(true_labels, predictions)
    
    print("\nMatriz de confusión:")
    print("-"*30)
    print(cm)

    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.colorbar()

    # Agregar números manualmente
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    plt.ylabel('Valores Reales')
    plt.xlabel('Predicciones')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.tight_layout()
    plt.show()
    
    print("Evaluación completada!")
    print(f"Cantidad de predicciones del test: {len(predictions)}")