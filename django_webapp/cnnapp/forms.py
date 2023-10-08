from django import forms

class UploadForm(forms.Form):
    key = forms.CharField(max_length=20)
    image = forms.ImageField()