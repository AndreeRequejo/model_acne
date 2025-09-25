# config.py
import torchvision
from torchvision.models import EfficientNet_V2_M_Weights

# PATHS AND FILES
TRAIN_FILES = [
    "ACNE04/NNEW_trainval_0.txt", #0
    "ACNE04/NNEW_trainval_1.txt", #1
    "ACNE04/NNEW_trainval_2.txt", #2
    "ACNE04/NNEW_trainval_3.txt", #3
    "ACNE04/NNEW_trainval_4.txt", #4
    "ACNE04/NORMALIZED_NNEW_trainval_0.txt", #5
    "ACNE04/NORMALIZED_NNEW_trainval_1.txt", #6
    "ACNE04/NORMALIZED_NNEW_trainval_2.txt", #7
    "ACNE04/NORMALIZED_NNEW_trainval_3.txt", #8
    "ACNE04/NORMALIZED_NNEW_trainval_4.txt"  #9
]

TEST_FILES = [
    "ACNE04/NNEW_test_0.txt", #0
    "ACNE04/NNEW_test_1.txt", #1
    "ACNE04/NNEW_test_2.txt", #2
    "ACNE04/NNEW_test_3.txt", #3
    "ACNE04/NNEW_test_4.txt", #4
    "ACNE04/NORMALIZED_NNEW_test_0.txt", #5
    "ACNE04/NORMALIZED_NNEW_test_1.txt", #6
    "ACNE04/NORMALIZED_NNEW_test_2.txt", #7
    "ACNE04/NORMALIZED_NNEW_test_3.txt", #8
    "ACNE04/NORMALIZED_NNEW_test_4.txt"  #9
]

IMAGE_PATH = "ACNE04/JPEGImages"

# TRAINING PARAMETERS
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
SMOOTHING = 0.12

# IMAGE TRANSFORMS
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
IMAGE_SIZE = (224, 224)

# DATA TRANSFORMS
TRAIN_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMAGE_SIZE),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomRotation(degrees=15),
    torchvision.transforms.ElasticTransform(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)
])

TEST_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMAGE_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)
])

# MODEL SAVE PATH
MODEL_SAVE_PATH = "./acne_model.pt"

# CLASS NAMES (ajusta según tu dataset)
CLASS_NAMES = [
    "Leve",      # Clase 0
    "Moderado",  # Clase 1
    "Severo",    # Clase 2
    #"Muy Severo" # Clase 3
]