# test_simple.py - Script básico para test rápido
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Importar módulos locales
from config import TEST_FILES, IMAGE_PATH, TEST_TRANSFORM, CLASS_NAMES
from dataset import ClassificationDataset
from model import MyNet

def quick_test(model_path="model_acne.pt"):
    """Test rápido del modelo"""
    
    print("🔄 Cargando modelo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo (método simple)
    try:
        model = torch.load(model_path, weights_only=False, map_location=device)
        print("✅ Modelo cargado exitosamente")
    except:
        print("❌ Error cargando modelo. Intentando con MyNet()...")
        try:
            model = MyNet()
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            print("✅ Modelo cargado usando state_dict")
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    model = model.to(device)
    model.eval()
    
    print("🔄 Cargando datos de test...")
    # Cargar datos
    test_df = pd.read_csv(TEST_FILES[0], names=['path','label','leisons'], sep='  ', engine='python')
    testset = ClassificationDataset(test_df, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    print(f"✅ {len(test_df)} muestras de test cargadas")
    
    print("🔄 Evaluando...")
    # Evaluar
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = torch.argmax(output, axis=1).item()
            
            predictions.append(pred)
            true_labels.append(target.item())
    
    # Resultados
    accuracy = accuracy_score(true_labels, predictions)
    
    print("\n" + "="*50)
    print("📊 RESULTADOS:")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=CLASS_NAMES))
    print("="*50)
    
    return accuracy, predictions, true_labels

if __name__ == "__main__":
    import sys
    
    # Usar nombre de modelo desde argumentos o default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "model_acne.pt"
    
    print(f"🚀 Testing modelo: {model_name}")
    quick_test(model_name)