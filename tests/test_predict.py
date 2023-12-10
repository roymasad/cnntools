import pytest
from main import ImageClassifier, predict
from cnntools.structs import ImageClass
import torch
from config import device  # Add this line

def test_predict():
    """
    Test that the predict function correctly classifies a known cat image and a known dog image.
    """
    print("Running test: Correctly predict cat and dog.")
    
    
    # Load a pre-trained model
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load('trained_model_mps_50epoch_64pixel_256filters_32batch_6354sec_0.272loss_0.877acc_24007files_10.13_08-10-2023.pth', map_location=device))

    # Test the predict function with a known cat image
    cat_image = 'test/cats/cat.12000.jpg'
    result = predict(cat_image, model)
    assert result == ImageClass.CAT.value, "The model failed to classify a known cat image"

    # Test the predict function with a known dog image
    dog_image = 'test/dogs/dog.12000.jpg'
    result = predict(dog_image, model)
    assert result == ImageClass.DOG.value, "The model failed to classify a known dog image"