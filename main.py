# main.py
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Importar módulos locales
from config import *
from dataset import ClassificationDataset, data_split, load_data
from metrics import Metrics
from model import MyNet, LabelSmoothingLoss
from training import train_one_epoch, test_result


def create_dataloaders():
    """Create training, validation and test dataloaders"""
    # Load data
    train_df, test_df = load_data(TRAIN_FILES[1], TEST_FILES[1])
    
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
    train_loader, val_loader, test_loader = create_dataloaders()
    model, optimizer, scheduler, criterion, train_metrics, val_metrics = setup_training()
    
    device = torch.device("cuda")
    model = model.to(device)
    
    # Enable gradients
    for param in model.parameters():
        param.requires_grad = True
    
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
    
    return test_loader, device


def evaluate_model(test_loader, device):
    """Evaluate the best model on test set"""
    from sklearn.metrics import classification_report as rp
    
    # Intentar cargar primero los pesos solamente (más seguro)
    weights_path = MODEL_SAVE_PATH.replace('.pt', '_weights.pt')
    
    try:
        # Crear modelo y cargar solo los pesos
        test_model = MyNet().cuda()
        test_model.load_state_dict(torch.load(weights_path, weights_only=True))
        test_model = test_model.to(device)
        print("Modelo cargado usando state_dict (seguro)")
    except:
        # Si falla, usar el método original
        test_model = torch.load(MODEL_SAVE_PATH, weights_only=False)
        test_model = test_model.to(device)
        print("Modelo cargado usando método legacy")
    
    preds, labels = test_result(test_model, test_loader, device)
    
    # Mostrar reporte de clasificación
    print("\n" + "="*50)
    print("Reporte de clasificación:")
    print("="*50)
    print(rp(labels.flatten(), preds))
    print("="*50)
    
    return preds, labels


if __name__ == "__main__":
    # Train the model
    test_loader, device = train_model()
    
    # Evaluate on test set
    predictions, true_labels = evaluate_model(test_loader, device)
    
    print("Entrenamiento completado!")
    print(f"Test predictions shape: {len(predictions)}")
    print(f"True labels shape: {true_labels.shape}")