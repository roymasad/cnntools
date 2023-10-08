# file with shared data structures

from torchvision import transforms, datasets 
import torch
import torch.nn as nn    
from .config import *

# CNN image classifier, 3 convoluted layers with a pool layer inside each
# -convoluted layer -> learning filters that create feature maps
# -pooling layers to reduce spatial size dependence and improve important salient features
# -fully connected laters process pooled laters to create classifaction label
# kernel->weight

# PS, for speed you can enable stride (like a step function, speed up but lose data)
# PS , maybe increasing kerning size from conv layer to another, (detect different features)
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #layer configurations (padding kept at 1 to prevent loss of information)
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),           #first layer, 3 in channels, rgb, 64 output featuremaps
          nn.BatchNorm2d(64),                       #same as feature nb in each layer, improve training
          nn.ReLU(),                                #activation function
          nn.MaxPool2d(2))                          #pooling, effectively downsampling, 512 becomes 256
        
        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, int(MAX_FILTERS/2), 3, padding=1),         #input nb is the output of previous layers (sequential)
          nn.BatchNorm2d(int(MAX_FILTERS/2)),
          nn.ReLU(),
          nn.MaxPool2d(2))                          #divides feature maps by 2 & output width
        
        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(int(MAX_FILTERS/2), MAX_FILTERS, 3, padding=1),        # last nb of layers. kernel not increasing also?
          nn.BatchNorm2d(MAX_FILTERS),
          nn.ReLU(),
          nn.MaxPool2d(2))                          
        
        self.classifier = nn.Sequential(                    #fully connected linear layers
          nn.Flatten(),                                     #flaten to 0 or 1 and give result
          # in_features = last filter  * output width * output width (feature maps)
          nn.Linear(in_features=MAX_FILTERS*int(IMAGE_WIDTH/8)*int(IMAGE_WIDTH/8), out_features=1024),
          nn.ReLU(),
          nn.Linear(in_features=1024, out_features=2)
          )   #prediction (0..1), linear binary classifier RESULT. Bias implied
        self.dropout = nn.Dropout2d(p=0.5)         #drop out probability
    
    #forward passes on tensors
    def forward(self, x: torch.Tensor):                     
        
        x = self.conv_layer_1(x)              #pass X tensor from one layer to another
        x = self.conv_layer_2(x)
        x = self.dropout(x)                   #helps against overfitting, better regularization
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x
    

# Letterboxing used in the tensor transform to resize + add image padding if need be to keep picture proportions
# without this a resize will strech and squash the images (of different source sizes) which is less ideale
# This version is to be used in transforming datasets (image date isnt tensors yet, image has width,height)
class PadToSizeForDatasets:

    def __call__(self, image):

        # default padding is 0 if W = H
        left = 0
        right = 0
        top = 0
        bottom = 0

        # get image W and H
        width, height = image.size

        pad_difference = abs(height - width) 

        # add padding only on the side that might need to, to make final image uniform in W and H
        if width < height:
            right = pad_difference // 2
            left = pad_difference // 2

        if width > height:
            top = pad_difference // 2
            bottom = pad_difference // 2

        image = transforms.functional.pad(image, (left, top, right, bottom), fill=0)  # Fill=0 for black padding
        image = transforms.functional.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=2, antialias=None)  # Interpolation=2 for bicubic
        
        return image
 
# Slightly modifed version to be used for PyVision
# Image data is already tensors, image has shape[] for width and height
class PadToSizeForTorchvision:

  def __call__(self, image):

      # default padding is 0 if W = H
      left = 0
      right = 0
      top = 0
      bottom = 0

      # get image W and H
      width, height = image.shape[2], image.shape[1]

      pad_difference = abs(height - width) 

      # add padding only on the side that might need to, to make final image uniform in W and H
      if width < height:
          right = pad_difference // 2
          left = pad_difference // 2

      if width > height:
          top = pad_difference // 2
          bottom = pad_difference // 2

      image = transforms.functional.pad(image, (left, top, right, bottom), fill=0)  # Fill=0 for black padding
      image = transforms.functional.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=2, antialias=None)  # Interpolation=2 for bicubic
      
      return image

# dataset transforms for training (have more random variations), the order matters
data_transform_training = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # random flip in case the data set has an orientation bias
    PadToSizeForDatasets(),
    transforms.TrivialAugmentWide(),  #random modifications (random resize, crop)    
    transforms.ToTensor()
])

# dataset transform for validation (just a resize to fit the training)
data_transform_testing = transforms.Compose([
    PadToSizeForDatasets(),
    transforms.ToTensor()
])

# torch vision loading version of the transform (no tensor creation)
torchvision_image_transform = transforms.Compose([
  PadToSizeForTorchvision()
  #transforms.Resize(IMAGE_SIZE, antialias=None) #resizess but stretches the proportions
])
