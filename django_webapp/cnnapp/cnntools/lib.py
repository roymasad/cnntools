# file with the library functions for setting up and using pytorch tensors

import torch
import torch.nn as nn                             # neural network, for training a CNN, input,hidden, out
from torchinfo import summary
from tqdm.auto import tqdm                        # console progress bar
import torchvision
from torch.utils.data import DataLoader  
from .structs import * 
from .config import *


def load_training_data(dirpath):
    """
    Load and transform training datasets , then turn them into tensors and dataloaders
    @dirpath: parent training folder path
    """

    #first step load training data, generic with transforms 
    train_data = datasets.ImageFolder(root=dirpath, 
                                      transform=data_transform_training, #data_transform_testing for validation
                                      target_transform=None)
    
    #second step loading, turn trains into Dataloaders
    
    dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True) #shake off bias, True for training, False for validation

    return dataloader

# Train model  
def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer):
  
  print ("\nTraining")
  
  # start training
  model.train()
  
  # loss and accuracy
  loss_train = 0
  accuracy = 0
  
  # loop in batches
  for (input, labels) in dataloader: 
    
    input = input.to(device) # perform calculations on that device
    labels = labels.to(device)
    
    # run forward pass (take x and run it in the __forward function of the classifier)
    model_result = model(input)
    
    # accumulate loss
    loss = loss_fn(model_result, labels)
    loss_train += loss.item()
    
    # set grandient to zero before starting this epoch, important
    optimizer.zero_grad()
    
    # mandatory back propagation calculation
    loss.backward()
    
    # mandatory optimize parameters based on the back propagation
    optimizer.step()
    
    # predict class (get the index of the highest cell in a certain direction (most probable) )
    label_pred_class = torch.argmax(torch.softmax(model_result, dim=1), dim=1) #dimension 1 is cols
    # Calculate and accumulate accuracy metric across all batches
    accuracy += (label_pred_class == labels).sum().item()/len(model_result)
    
  loss_train = loss_train / len(dataloader)
  accuracy = accuracy / len(dataloader)
  return loss_train, accuracy
    
# main function to train model
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 1):
  
  results = { "train_loss": [],
              "train_acc": []
  }
  # tqdm is an auto console progress bar
  for epoch in tqdm(range(epochs)):
    
    train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
    
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
    )

    # saving only the last loss/acc to return (ignoring graph data)
    results["train_loss"] = train_loss
    results["train_acc"] = train_acc
    
  return results

# make an image prediction based on a trained model
# the model file loaded hyperparameters should match the hyperparameters saved in the file (features, size)
def predict(image_path, model):
  
  print(f"Image selected for prediction :\n {image_path}")
  
  image = torchvision.io.read_image(str(image_path)).type(torch.float32)
  image = image /255 # normalize from 255 pixel values to 0 to 1 #important for accuracy (but trained data is not normalized?)
     
  #resize and letterbox padding    
  image_transformed = torchvision_image_transform(image)
  
  # single forward pass evaluation
  model.eval() #read only, done change anything in the model wires, just use it
  
  with torch.inference_mode(): #dont update parameters/gradients, newer thant torch.nograd()
    
    image_transformed_batch = image_transformed.unsqueeze(dim=0)
    
    prediction = model(image_transformed_batch.to(device))
    
    prediction_probability =  torch.softmax(prediction, dim=1)
    
    prediction_label = torch.argmax(prediction_probability, dim=1)
    
    #summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT])
  
  return prediction_label[0]
