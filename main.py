# Binary image classification with Pytorch 2

# main pyton file to running and testing (cli for training, prediction)
# used to classify cats and dogs image datasets 

# this program uses a CCN Convolutional Network with layers, fit for 2d grids like images and videos

#
# Useful tutorial
# https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212

import random
import torch
from torch import nn
from timeit import default_timer as timer
from datetime import datetime
import os.path

# this project's files
from config import *
from cnntools.utilities import *
from cnntools.structs import * 
from cnntools.lib import *

def main_loop():
    """
    The main loop of the program. It creates a console menu loop that allows the user to interact with the program. The user can choose to test a pre-trained model, train a new model, or quit the program. The function takes no parameters and does not return anything.

    Within the loop, the function displays a menu with three options: test a pre-trained model, train a new model, or quit. The user can enter their choice by selecting the corresponding number. If the user does not enter any input, the default option is selected.

    If the user chooses to test a pre-trained model, the function prompts the user to select a trained file from the local folder. It then loads the selected model and allows the user to choose between auto-selecting test images or providing a custom image. If the user chooses auto-select, the function randomly selects 10 images (5 cats and 5 dogs) from the test folder and predicts their labels using the model. It then displays the images and their predicted labels. If the user chooses to provide a custom image, they can enter the filename of a JPEG image in the local folder. The function predicts the label of the image using the model and displays the image and its predicted label.

    If the user chooses to train a new model, the function prompts the user to enter the filename of the model. The function then loads and transforms the training data from the train folder and creates dataloaders for training and testing data. It defines a loss function and an optimizer, and trains the model for a specified number of epochs. It measures the training time and saves the trained model with a unique filename that includes various hyperparameters and metadata.

    If the user chooses to quit, the function prints a farewell message and exits the program.
   """
    # create console menu loop (you can reuse the functions in the same session, then exit)
    while True:

        print(f"\nKitties & Dogs Classification CNN with pyTorch {torch.__version__}\n")

        print(GREEN+"1-Test pre trained model\n2-Train new model\n3-Quit\n(Default 1)\n"+RESET)
        choice_input = input("# ")
        if choice_input == "":
            choice_input = "1"

        # auto test image classifactions on a saved model
        if choice_input == "1":
            
            print(GREEN+f"-Select index of training file in local folder: (Default 0)\n"+RESET)
            
            trained_filenames = list_files_folder(".", "*.pth") 
            
            if trained_filenames.__len__() == 0:
                print(RED+"\nNo saved training files (*.pth) found in local folder\n"+RESET)
                quit()
            
            for index, name in enumerate(trained_filenames):
                print(str(index) + ": " + name)
            
            filename_input = 0
            
            # lock the cli til a valid answer is selected
            while True:
                try:
                    filename_input = input("# ")
                    if filename_input == "":
                        filename_input = 0
                    filename_input = int(filename_input)    #throw exeption is time is not valid
                    if (filename_input < 0) or (filename_input > trained_filenames.__len__()-1): # not within list
                        print(RED+"\nInvalid selection\n"+RESET)
                        continue
                    break # only exit if answer is acceptable
                except ValueError:
                    print(RED+"\nInvalid selection\n"+RESET)
            
            # get an instance of the model class
            model = ImageClassifier().to(device)

            #load previously saved trained model (if you dont want to train), ps load / save the same network setup!
            model.load_state_dict(torch.load(trained_filenames[filename_input], map_location=device))
                
            print(GREEN+f"1-Auto pick 10 test images (cat+dog)\n2-Choose custom pic\n(Default 1)\n"+RESET)
            selection_mode = "auto"
            choice = input("# ")
            if choice == "1":
                selection_mode = "auto"
            elif choice == "2":
                selection_mode = "manual"
            else:
                selection_mode = "auto"     
            
            # auto select 10 test pics and their predicted labels
            if selection_mode == "auto":
            
                cats_dir = test_dir + "/cats/"
                dogs_dir = test_dir + "/dogs/"

                cat_filenames = list_files_folder(cats_dir, "*.jpg") 
                dog_filenames = list_files_folder(dogs_dir, "*.jpg") 
                
                for step in range(0,5):
                
                    # pick a random cat and dog pic !
                    catfile = cats_dir + cat_filenames[int(random.randrange(0,cat_filenames.__len__()-1))]
                    dogfile = dogs_dir + dog_filenames[int(random.randrange(0,dog_filenames.__len__()-1))]
                
                    #try a random kitty
                    print("Random Picking from cats folder:")
                    # predict!
                    result = predict(catfile, model)
                    if result == ImageClass.CAT.value: 
                        name = "Kitty?"
                    else: 
                        name = "Doggy?"
                    print(GREEN+"Its a "+name+RESET)
                    open_jpeg_file(catfile, cli=False, title = name)

                    #try a random doggy
                    print("Random Picking from dogs folder:")
                    #predict !
                    result = predict(dogfile, model)
                    if result == ImageClass.CAT.value: 
                        name = "Kitty?"
                    else: 
                        name = "Doggy?"
                    print(GREEN+"Its a "+name+RESET)
                    open_jpeg_file(dogfile, cli=False, title = name)
                    
                    print(GREEN+"\nDONE TESTING\n"+RESET)
            
            if selection_mode == "manual":
                
                # transform_image(dogfile) # if you want to see how it looks transformed in training
                print("\nEnter the filename of the jpg image (local folder default)\n")
                filename = input("# ")
                
                if filename == "":
                    print(RED+"\nInvalid selection\n"+RESET)
                    quit()
                    
                if os.path.isfile(filename) == False:
                    print(RED+"\nInvalid file\n"+RESET)
                    quit()
                if filename.endswith(".jpg") == False:
                    print(RED+"\nNot a JPEG file\n"+RESET)
                    quit()
                
                # predict!
                result = predict(filename, model)
                if result == ImageClass.CAT.value: 
                    name = "Kitty?"
                else: 
                    name = "Doggy?"
                print(GREEN+"It's a "+name+RESET)
                open_jpeg_file(filename, cli=False, title = name)
                    
        # train and save new image classification model    
        elif choice_input == "2":
            
            print(GREEN+f"-Filename of model: ({default_trained_model_file_path}.pth)\n"+RESET)
            filename_input = input("# ")
            if filename_input == "":
                filename_input = default_trained_model_file_path

            cats_dir = train_dir + "/cats/"
            dogs_dir = train_dir + "/dogs/"

            cat_filenames = list_files_folder(cats_dir, "*.jpg") 
            dog_filenames = list_files_folder(dogs_dir, "*.jpg") 
            
            nb_files = cat_filenames.__len__() + dog_filenames.__len__()

            # get an instance of the model class
            model = ImageClassifier().to(device)
            
            # load and transform datesets and then turn them into dataloaders for training and testing data
            training_data = load_training_data(train_dir)
            
            # loss function and optimizer
            loss_fn = nn.CrossEntropyLoss() #binary cross entropy function might be better(cat or dog, this or that)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001) #Adam optimizer, faster than SGD?

            # track time
            start_time = timer()

            # train with the selected params
            model_results = train(model=model,
                                train_dataloader=training_data,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                epochs=NUM_EPOCHS)

            end_time = timer()
            print(f"Total training time: {end_time-start_time:.2f} seconds")

            loss = model_results["train_loss"]
            accuracy = model_results["train_acc"]

            # save result with hyper tags
            str_date_time = datetime.now().strftime("%M.%H_%d-%m-%Y")
            meta_tags = f"_{ARCHITECTURE}_{NUM_EPOCHS}epoch_{IMAGE_HEIGHT}pixel_{MAX_FILTERS}filters_{BATCH_SIZE}batch_{end_time-start_time:.0f}sec_{loss:.3f}loss_{accuracy:.3f}acc_{nb_files}files_{str_date_time}"
            finalname = filename_input + meta_tags
            
            torch.save(model.state_dict(), finalname + ".pth")
            
            print(GREEN + f"\nDONE: result saved to :\n {finalname}.pth\n")
            
        elif choice_input == "3":
            
            print("Ciao data scientist!")
            quit(0) # quit the program
        else:
            print(RED+"\nInvalid selection\n"+RESET)
        
if __name__ == "__main__":
    main_loop()
    
    







