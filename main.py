# main.py

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import numpy as np
# Importar módulos locales
from config import *
from dataset import ClassificationDataset, data_split, load_data
from metrics import Metrics, plot_training_history_detailed
from model import MyNet, LabelSmoothingLoss
from training import train_one_epoch, test_result


def create_dataloaders():
    """Crear cargadores de datos de entrenamiento, validación y prueba"""
    # Cargar datos
    train_df, test_df = load_data(TRAIN_FILES[6], TEST_FILES[6])
    
    # Dividir datos de entrenamiento
    x_train, x_val, y_train, y_val = data_split(train_df, VALIDATION_SPLIT)
    
    # Crear conjuntos de datos
    train_dataset = ClassificationDataset(x_train, data_path=IMAGE_PATH, transform=TRAIN_TRANSFORM, training=True)
    val_dataset = ClassificationDataset(x_val, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    testset = ClassificationDataset(test_df, data_path=IMAGE_PATH, transform=TEST_TRANSFORM, training=True)
    
    # Crear cargadores de datos
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader


def setup_training():
    """Configurar modelo, optimizador, programador y función de pérdida"""
    # Detectar dispositivo disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializar contexto CUDA si está disponible
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.empty_cache()
    
    # Modelo
    model = MyNet().to(device)
    
    # Probar modelo con entrada de prueba
    test_input = torch.ones((16, 3, 224, 224)).to(device)
    model(test_input)
    
    # Optimizador y programador - configuración más conservadora
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.5)
    
    # Función de pérdida
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    # Métricas
    train_metrics = Metrics(["accuracy_score", "f1_score"])
    val_metrics = Metrics(["accuracy_score", "f1_score"])
    
    return model, optimizer, scheduler, criterion, train_metrics, val_metrics


def train_model():
    """Bucle principal de entrenamiento"""
    # Configuración inicial
    import matplotlib.pyplot as plt
    train_loader, val_loader, test_loader = create_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configurar modelo
    model, optimizer, scheduler, criterion, train_metrics, val_metrics = setup_training()
    model = model.to(device)

    # Habilitar gradientes
    for param in model.parameters():
        param.requires_grad = True

    # Listas para almacenar historia
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Bucle de entrenamiento
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

        print(f"Epoca {i + 1} / {NUM_EPOCHS} \nTraining loss: {loss:.4f} - Otras metricas de entrenamiento: ")
        print(f"Precision: {train_result['accuracy_score']:.4f} - F1 Score: {train_result['f1_score']:.4f}")
        print(f" \nValidation loss : {val_loss:.4f} - Otras metricas de validacion:")
        print(f"Precision: {val_result['accuracy_score']:.4f} - F1 Score: {val_result['f1_score']:.4f}")
        print("\n")

        # Guardar mejor modelo
        if best_val_acc < float(val_result["accuracy_score"]):
            best_val_acc = val_result["accuracy_score"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Precision de validacion: {best_val_acc:.4f} ===> Save best epoch")

            # =================== EARLY STOPPING (COMENTADO) ===================
            # no_improve = 0
        else:
            print(f"Precision de validacion: {val_result['accuracy_score']:.4f} ===> No saving")

            # =================== EARLY STOPPING (COMENTADO) ===================
            # no_improve += 1
            # if no_improve >= patience:
            #     print(f"Early stopping activado después de {i + 1} épocas")
            #     break
    
    # Usar las funciones de visualización
    plot_training_history_detailed(train_losses, val_losses, train_accuracies, val_accuracies)

    return test_loader, val_loader, device


def evaluate_model(data_loader, device, dataset_name="Test"):
    """Evaluar el modelo en cualquier conjunto de datos"""
    from sklearn.metrics import classification_report as rp, confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    
    try:
        # Crear modelo y cargar solo los pesos
        test_model = MyNet().to(device)
        test_model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
        test_model = test_model.to(device)
    except:
        # Si falla, usar el método original
        test_model = torch.load(MODEL_SAVE_PATH, weights_only=False)
        test_model = test_model.to(device)
    
    preds, labels = test_result(test_model, data_loader, device)

    # Calcular accuracy
    accuracy = accuracy_score(labels.flatten(), preds)
    
    print(f"\n{'='*55}")
    print(f"EVALUACIÓN DEL CONJUNTO: {dataset_name.upper()}")
    print(f"{'='*55}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Mostrar reporte de clasificación
    print("\n" + "="*55)
    print(f"Reporte de clasificación - {dataset_name}:")
    print("="*55)
    print(rp(labels.flatten(), preds, target_names=CLASS_NAMES))
    print("="*55)

    # Calcular y mostrar matriz de confusión
    cm = confusion_matrix(labels.flatten(), preds)
    
    print("\nMatriz de confusión:")
    print("="*55)
    print(cm)

    # Visualizar matriz de confusión con porcentajes
    plt.figure(figsize=(10, 8))
    
    # Calcular porcentajes por fila (para cada clase real)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.imshow(cm_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    plt.title(f'Matriz de Confusión - {dataset_name}', fontsize=16, fontweight='bold')
    cbar = plt.colorbar()
    cbar.set_label('Porcentaje (%)', rotation=270, labelpad=20)

    # Agregar números y porcentajes manualmente
    thresh = cm_percent.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Mostrar número absoluto y porcentaje
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            plt.text(j, i, text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm_percent[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    plt.ylabel('Valores Reales', fontsize=14)
    plt.xlabel('Predicciones', fontsize=14)
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.tight_layout()
    plt.show()
    
    return preds, labels


if __name__ == "__main__":
    # Entrenar modelo
    test_loader, val_loader, device = train_model()
    
    print("\n" + "-"*55)
    print("INICIANDO EVALUACIÓN COMPLETA")
    print("-"*55)
    
    # Evaluar conjunto de validación (20% del train)
    val_predictions, val_labels = evaluate_model(val_loader, device, "Validación")
    
    # Evaluar conjunto de test independiente
    test_predictions, test_labels = evaluate_model(test_loader, device, "Test Independiente")
    
    # Resumen final
    print("\n" + "="*55)
    print("RESUMEN FINAL DE EVALUACIÓN")
    print("="*55)
    print(f"Validación (20% train): {len(val_predictions)} muestras")
    print(f"Test independiente: {len(test_predictions)} muestras")
    print("Entrenamiento completado!")
    print("="*55)