# main.py

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
# Importar módulos locales
from config import *
from dataset import ClassificationDataset, data_split, load_data
from metrics import Metrics
from model import MyNet, LabelSmoothingLoss
from training import train_one_epoch, test_result


def create_dataloaders():
    """Create training, validation and test dataloaders"""
    # Load data
    train_df, test_df = load_data(TRAIN_FILES[4], TEST_FILES[4])
    
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
    # Model
    model = MyNet().cuda()
    
    # Test model with dummy input
    test_input = torch.ones((16, 3, 224, 224)).cuda()
    model(test_input)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.5)
    
    # Loss function
    criterion = LabelSmoothingLoss(smoothing=SMOOTHING)
    
    # Metrics
    train_metrics = Metrics(["accuracy_score", "f1_score"])
    val_metrics = Metrics(["accuracy_score", "f1_score"])
    
    return model, optimizer, scheduler, criterion, train_metrics, val_metrics


def train_model():
    """Main training loop"""
    # Setup
    import matplotlib.pyplot as plt
    train_loader, val_loader, test_loader = create_dataloaders()
    device = torch.device("cuda")


    # Si el modelo ya existe, cargarlo y continuar entrenamiento
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Modelo encontrado en {MODEL_SAVE_PATH}. Se continuará el entrenamiento desde el modelo guardado.")
        model = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
        model = model.to(device)
        # Se inicializan de nuevo optimizer, scheduler, etc. para continuar
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.5)
        criterion = LabelSmoothingLoss(smoothing=SMOOTHING)
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
        if loss < 0.04:
            continue

        if best_val_acc < float(val_result["accuracy_score"]):
            print(f"Validation accuracy= {val_result['accuracy_score']} ===> Save best epoch")
            best_val_acc = val_result["accuracy_score"]
            torch.save(model, MODEL_SAVE_PATH)
        else:
            print(f"Validation accuracy= {val_result['accuracy_score']} ===> No saving")
            continue

    # Graficar curva de aprendizaje
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Pérdida entrenamiento')
    plt.plot(epochs, val_losses, label='Pérdida validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curva de pérdida')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Accuracy entrenamiento')
    plt.plot(epochs, val_accuracies, label='Accuracy validación')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.title('Curva de accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return test_loader, device


def evaluate_model(test_loader, device):
    """Evaluar el modelo en el conjunto de prueba"""
    from sklearn.metrics import classification_report as rp, confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    
    # Intentar cargar primero los pesos solamente
    weights_path = MODEL_SAVE_PATH.replace('.pt', '_weights.pt')
    
    try:
        # Crear modelo y cargar solo los pesos
        test_model = MyNet().cuda()
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