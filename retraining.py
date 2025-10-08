# retraining.py

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from config import *
from dataset import ClassificationDataset, data_split, load_data
from metrics import Metrics, plot_learning_curves_advanced, plot_training_history_detailed
from model import MyNet, LabelSmoothingLoss
from training import train_one_epoch, test_result


def create_dataloaders_for_fold(fold_index):
    """Crear cargadores de datos de entrenamiento, validación y prueba para un fold específico"""
    train_df, test_df = load_data(TRAIN_FILES[fold_index], TEST_FILES[fold_index])
    x_train, x_val, y_train, y_val = data_split(train_df, VALIDATION_SPLIT)
    
    train_dataset = ClassificationDataset(x_train, data_path=IMAGE_PATH, transform=TRAIN_TRANSFORM, training=True)
    val_dataset = ClassificationDataset(x_val, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    testset = ClassificationDataset(test_df, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader


def setup_model_and_training(device, previous_weights_path=None):
    """Configurar modelo, optimizador, programador de aprendizaje y función de pérdida"""
    model = MyNet().to(device)
    
    # Cargar pesos del modelo anterior si están disponibles
    if previous_weights_path and os.path.exists(previous_weights_path):
        try:
            model.load_state_dict(torch.load(previous_weights_path, map_location=device, weights_only=True))
            print(f"Pesos cargados desde: {previous_weights_path}")
        except Exception as e:
            print(f"No se pudieron cargar los pesos previos: {e}")
            print("Iniciando con pesos por defecto de EfficientNet")
    else:
        print("Iniciando con pesos por defecto de EfficientNet")
    
    # Probar modelo con entrada de prueba
    test_input = torch.ones((16, 3, 224, 224)).to(device)
    model(test_input)
    
    # Configurar componentes de entrenamiento
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.5)
    criterion = LabelSmoothingLoss(smoothing=0.1)
    train_metrics = Metrics(["accuracy_score", "f1_score"])
    val_metrics = Metrics(["accuracy_score", "f1_score"])
    
    return model, optimizer, scheduler, criterion, train_metrics, val_metrics


def get_model_save_paths(fold_index):
    """Obtener rutas de guardado para un fold específico"""
    model_path = f"core/fold_{fold_index}_best.pt"
    weights_path = f"core/fold_{fold_index}_best_weights.pt"
    return model_path, weights_path


def train_single_fold(fold_index, previous_weights_path=None):
    """Entrenar un solo fold con aprendizaje incremental opcional"""
    
    print(f"\nIniciando entrenamiento para fold {fold_index}")
    print("="*50)
    
    # Configurar dispositivo de cómputo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.empty_cache()
    
    # Crear cargadores de datos
    train_loader, val_loader, test_loader = create_dataloaders_for_fold(fold_index)
    print(f"Datos cargados - Entrenamiento: {len(train_loader.dataset)}, Validaci\u00f3n: {len(val_loader.dataset)}, Prueba: {len(test_loader.dataset)}")
    
    # Configurar modelo y componentes de entrenamiento
    model, optimizer, scheduler, criterion, train_metrics, val_metrics = setup_model_and_training(device, previous_weights_path)
    
    # Habilitar gradientes
    for param in model.parameters():
        param.requires_grad = True
    
    # Historial de entrenamiento
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Bucle de entrenamiento
    best_val_acc = 0.0
    print(f"Iniciando entrenamiento para {NUM_EPOCHS} \u00e9pocas...")
    
    for epoch in tqdm(range(NUM_EPOCHS)):
        loss, val_loss, train_result, val_result = train_one_epoch(
            model, train_loader, val_loader, device, optimizer, criterion, train_metrics, val_metrics
        )
        
        scheduler.step(val_loss)
        
        # Guardar historial
        train_losses.append(loss)
        val_losses.append(val_loss)
        train_accuracies.append(float(train_result["accuracy_score"]))
        val_accuracies.append(float(val_result["accuracy_score"]))

        print(f"Epoca {epoch + 1} / {NUM_EPOCHS} \nTraining loss: {loss:.4f} - Otras metricas de entrenamiento: ")
        print(f"Precision: {train_result['accuracy_score']:.4f} - F1 Score: {train_result['f1_score']:.4f}")
        print(f" \nValidation loss : {val_loss:.4f} - Otras metricas de validacion:")
        print(f"Precision: {val_result['accuracy_score']:.4f} - F1 Score: {val_result['f1_score']:.4f}")
        print("\n")
        
        # Guardar mejor modelo
        if best_val_acc < float(val_result["accuracy_score"]):
            best_val_acc = val_result["accuracy_score"]
            
            model_path, weights_path = get_model_save_paths(fold_index)
            torch.save(model, model_path)
            torch.save(model.state_dict(), weights_path)
            
            print(f"Precision de validacion: {best_val_acc:.4f} ===> Save best epoch")
        else:
            print(f"Precision de validacion: {val_result['accuracy_score']:.4f} ===> No saving")
    
    # Graficar resultados
    plot_learning_curves_advanced(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_training_history_detailed(train_losses, val_losses, train_accuracies, val_accuracies)

    print(f"Fold {fold_index} completado con mejor precision de validacion: {best_val_acc:.4f}")

    return test_loader, val_loader, device, weights_path


def evaluate_fold(fold_index, test_loader, val_loader, device):
    """Evaluar el modelo entrenado para un fold específico"""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    model_path, weights_path = get_model_save_paths(fold_index)
    
    # Cargar el mejor modelo
    try:
        test_model = MyNet().to(device)
        test_model.load_state_dict(torch.load(weights_path, weights_only=True))
    except:
        test_model = torch.load(model_path, weights_only=False)
    
    test_model = test_model.to(device)
    
    # Obtener predicciones
    val_preds, val_labels = test_result(test_model, val_loader, device)
    test_preds, test_labels = test_result(test_model, test_loader, device)
    
    # Calcular precisiones
    val_accuracy = accuracy_score(val_labels.flatten(), val_preds)
    test_accuracy = accuracy_score(test_labels.flatten(), test_preds)
    
    print(f"\nResultados del Fold {fold_index}:")
    print("=" * 55)
    print(f"Precisión de Validación: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Precisión de Prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\nReporte de Clasificación - Conjunto de Validación:")
    print("=" * 55)
    print(classification_report(val_labels.flatten(), val_preds, target_names=CLASS_NAMES))
    
    print("\nReporte de Clasificación - Conjunto de Prueba:")
    print("=" * 55)
    print(classification_report(test_labels.flatten(), test_preds, target_names=CLASS_NAMES))
    
    return {
        'fold': fold_index,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }


def train_fold_with_best_weights(fold_index):
    """
    Entrenar un fold específico usando los pesos de core/best.pt como punto de partida
    
    Args:
        fold_index: Índice del fold a entrenar (usa TRAIN_FILES[fold_index] y TEST_FILES[fold_index])
    """
    
    best_weights_path = "core/best.pt"
    
    print(f"Entrenando fold {fold_index} usando pesos de {best_weights_path}")
    print("="*50)
    
    # Verificar si existen los pesos
    if os.path.exists(best_weights_path):
        print(f"Cargando pesos desde: {best_weights_path}")
    else:
        print(f"Archivo {best_weights_path} no encontrado, iniciando con pesos de EfficientNet")
        best_weights_path = None
    
    # Entrenar el fold específico
    test_loader, val_loader, device, current_weights_path = train_single_fold(fold_index, best_weights_path)
    
    # Evaluar el fold
    result = evaluate_fold(fold_index, test_loader, val_loader, device)
    
    return result


if __name__ == "__main__":
    # Entrenar un fold específico usando pesos de core/best.pt
    result = train_fold_with_best_weights(9)
    
    print(f"\nModelos guardados en directorio 'core/'")