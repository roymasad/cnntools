# app wide config variables and hyper parameters

import torch
from enum import Enum

# locations of the data training and test sets(different and less pix in test)
train_dir = "train" # must have classication subfolders with images in them (ie cats, dogs..)
test_dir = "test"   # final random auto pick testing folder (it is not used for epoch validation)

# default name for loading/saving models
default_trained_model_file_path = 'trained_model'

# how many generations 
NUM_EPOCHS = 50 

#64 seems like a good value for pix of cats and dog speed/quality
IMAGE_HEIGHT = 64      # MUST BE THE SAME, W = H 
IMAGE_WIDTH = 64
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

#max feature maps
MAX_FILTERS = 256

ARCHITECTURE = "mps" #'cpu', 'cuda' nvidia, 'mps' macbook m1
device = torch.device(ARCHITECTURE) 

# how many dataleader batches at ones
BATCH_SIZE = 32   #8 to 16 for cpu, 32+ for gpu, memory intesive (more better accuray/mem)
NUM_WORKERS = 0   #PS more than zero can cause a runtine crash, test on your machine

class ImageClass(Enum):
  CAT = 0
  DOG = 1

# ANSI escape code for text color
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # Reset color to default

