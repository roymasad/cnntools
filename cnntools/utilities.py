# file with utility functions

import os
from PIL import Image                               # image manipulation
import glob                                         # system folder path operations
from pathlib import Path                            # console path manipulations
import climage                                      # print images to console
import numpy                                        # funky fast math and array lib
import matplotlib.pyplot as plt                     # graphics ploting and image viewing
from torchvision import transforms, datasets        # AI tensor stuff
from torch.utils.data import DataLoader             # Parallel dataset loader
from cnntools.structs import * 

def list_files_folder(path, ext):
    """
    Returns a list of all files in a folder path
    @path (str): location of the folder
    
    Returns:
    Array of filenames
    """
         
    names = []
    list = glob.glob(path+"/"+ext)
    
    # Sort files by modification time in descending order (newest first)
    list.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for name in list:
        names.append(Path(name).name)

    return names

def open_jpeg_file(filepath, cli = True, title = ""):
    """
    Open a jpeg image file unmodified
    @cli: if True opens it in terminal
    """
    
    img = Image.open(filepath)

    # print(f"Jpeg name {filepath}")
    # print(f"Height: {img.height}")
    # print(f"Width: {img.width}")

    # Print in as ASCII in the console :)
    if cli == True:
        output = climage.convert(filepath, is_unicode=True, is_256color=True, width=60)
        print(output)
    # Show it as a popup GUI
    else :
        img_arr = numpy.asarray(img)
        show_figure(img_arr, title)

def transform_image(filepath):
    """
    Transform image size and turn it into a Tensor then show it
    @filepath (str): image filename path
    """

    img = Image.open(filepath)

    # permute the original data args to a format/order compatible with pytorch
    # original -> color, height, width
    # permuted -> height, width, color
    transformed_img = data_transform_training(img).permute(1,2,0) 
    show_figure(transformed_img, "Transformed for Training")


# Utility function to plot/show image/figure
def show_figure(data, title):
    plt.figure(figsize=(6,6))
    plt.imshow(data)
    plt.title(title)
    plt.axis(False)
    plt.show()
    

