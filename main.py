# main.py

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
# Importar módulos locales
from config import *
from dataset import ClassificationDataset, data_split, load_data
from metrics import Metrics, plot_learning_curves_advanced, plot_training_history_detailed
from model import MyNet, LabelSmoothingLoss
from training import train_one_epoch, test_result


def create_dataloaders():
    """Create training, validation and test dataloaders"""
    # Load data
    train_df, test_df = load_data(TRAIN_FILES[7], TEST_FILES[7])
    
    # Split training data
    x_train, x_val, y_train, y_val = data_split(train_df, VALIDATION_SPLIT)
    
    # Create datasets
    train_dataset = ClassificationDataset(x_train, data_path=IMAGE_PATH, transform=TRAIN_TRANSFORM, training=True)
    val_dataset = ClassificationDataset(x_val, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    testset = ClassificationDataset(test_df, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader


def setup_training():
    """Setup model, optimizer, scheduler and loss function"""
    # Detectar dispositivo disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializar contexto CUDA si está disponible
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.empty_cache()
    
    # Model
    model = MyNet().to(device)
    
    # Test model with dummy input
    test_input = torch.ones((16, 3, 224, 224)).to(device)
    model(test_input)
    
    # Optimizer and scheduler - configuración más conservadora
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.5)
    
    # Loss function
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    # Metrics
    train_metrics = Metrics(["accuracy_score", "f1_score"])
    val_metrics = Metrics(["accuracy_score", "f1_score"])
    
    return model, optimizer, scheduler, criterion, train_metrics, val_metrics


def train_model():
    """Main training loop"""
    # Setup
    import matplotlib.pyplot as plt
    train_loader, val_loader, test_loader = create_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Si el modelo ya existe, cargarlo y continuar entrenamiento
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Modelo encontrado en {MODEL_SAVE_PATH}. Se continuará el entrenamiento desde el modelo guardado.")
        model = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
        model = model.to(device)
        # Se inicializan de nuevo optimizer, scheduler, etc. para continuar
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.5)
        
        # Loss function
        criterion = LabelSmoothingLoss(smoothing=0.1)

        train_metrics = Metrics(["accuracy_score", "f1_score"])
        val_metrics = Metrics(["accuracy_score", "f1_score"])
    else:
        # Si no existe, entrenar desde cero
        model, optimizer, scheduler, criterion, train_metrics, val_metrics = setup_training()
        model = model.to(device)

    # Enable gradients
    for param in model.parameters():
        param.requires_grad = True

    # Listas para almacenar historia
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    best_val_acc = 0.0
    print("Begin training process")

    # =================== VERSIÓN CON EARLY STOPPING ===================
    # patience = 5
    # no_improve = 0
    
    # =================== VERSIÓN SIN EARLY STOPPING ===================
    # Ejecutar todas las épocas completas
    
    for i in tqdm(range(NUM_EPOCHS)):
        loss, val_loss, train_result, val_result = train_one_epoch(
            model,
            train_loader,
            val_loader,
            device,
            optimizer,
            criterion,
            train_metrics,
            val_metrics,
        )

        scheduler.step(val_loss)

        # Guardar historia
        train_losses.append(loss)
        val_losses.append(val_loss)
        train_accuracies.append(float(train_result["accuracy_score"]))
        val_accuracies.append(float(val_result["accuracy_score"]))

        print(f"Epoch {i + 1} / {NUM_EPOCHS} \n Training loss: {loss} - Other training metrics: ")
        print(train_result)
        print(f" \n Validation loss : {val_loss} - Other validation metrics:")
        print(val_result)
        print("\n")

        # Save best model
        if best_val_acc < float(val_result["accuracy_score"]):
            best_val_acc = val_result["accuracy_score"]
            torch.save(model, MODEL_SAVE_PATH)
            torch.save(model.state_dict(), MODEL_PESOS_PATH)
            print(f"Validation accuracy= {best_val_acc} ===> Save best epoch")
            
            # =================== EARLY STOPPING (COMENTADO) ===================
            # no_improve = 0
        else:
            print(f"Validation accuracy= {val_result['accuracy_score']} ===> No saving")
            
            # =================== EARLY STOPPING (COMENTADO) ===================
            # no_improve += 1
            # if no_improve >= patience:
            #     print(f"Early stopping activado después de {i + 1} épocas")
            #     break
    
    # Usar las nuevas funciones de visualización
    plot_learning_curves_advanced(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_training_history_detailed(train_losses, val_losses, train_accuracies, val_accuracies)

    return test_loader, device


def evaluate_model(test_loader, device):
    """Evaluar el modelo en el conjunto de prueba"""
    from sklearn.metrics import classification_report as rp, confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    
    # Intentar cargar primero los pesos solamente
    weights_path = MODEL_SAVE_PATH.replace('.pt', '_weights.pt')
    
    try:
        # Crear modelo y cargar solo los pesos
        test_model = MyNet().to(device)
        test_model.load_state_dict(torch.load(weights_path, weights_only=True))
        test_model = test_model.to(device)
    except:
        # Si falla, usar el método original
        test_model = torch.load(MODEL_SAVE_PATH, weights_only=False)
        test_model = test_model.to(device)
    
    preds, labels = test_result(test_model, test_loader, device)

    # Calcular accuracy
    accuracy = accuracy_score(labels.flatten(), preds)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Mostrar reporte de clasificación
    print("\n" + "="*50)
    print("Reporte de clasificación:")
    print("="*50)
    print(rp(labels.flatten(), preds, target_names=CLASS_NAMES))
    print("="*50)

    # Calcular y mostrar matriz de confusión
    cm = confusion_matrix(labels.flatten(), preds)
    
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
    
    return preds, labels


if __name__ == "__main__":
    # Entrenar modelo
    test_loader, device = train_model()
    
    # Evaluar modelo
    predictions, true_labels = evaluate_model(test_loader, device)
    
    print(f"Cantidad de predicciones del test: {len(predictions)}")
    print("Entrenamiento completado!")