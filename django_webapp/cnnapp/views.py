from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadForm
from .models import UploadedImage
from .cnntools.config import *
from .cnntools.structs import *
from .cnntools.lib import *

# Create your views here.

def index(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            key = form.cleaned_data['key']
            image = form.cleaned_data['image']
            if key == 'pass123' and image.name.endswith('.jpg'):
                uploaded_image = UploadedImage(image = image)
                uploaded_image.save()
                
                # predict
                message = "It's a Kitty !"
                
                # get an instance of the model class
                model = ImageClassifier().to(device)

                #load previously saved trained model (if you dont want to train), ps load / save the same network setup!
                model.load_state_dict(torch.load("trained_model_mps_50epoch_64pixel_256filters_32batch_6354sec_0.272loss_0.877acc_24007files_10.13_08-10-2023.pth",
                                                 map_location=device))
                
                #predict
                result = predict("images/"+image.name, model)
                if result == 0: 
                    message = "Kitty?"
                else: 
                    message = "Doggy?"
                
                return render(request, 'success.html',{ 'message' : message})
    else:
        form = UploadForm
    return render(request, 'index.html', {'form':form})
            