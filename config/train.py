import torch
import utils.utils as utils

# Constants
NUM_CLASSES = 21
NUM_EPOCHS = 100
BATCH_SIZE=16
MODELS = utils.get_models()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

