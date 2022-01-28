# Dog_Breed_Classifier_App

The purpose of this project is to create a machine learning model and productionalize it to an app! This was a lot of fun =)

## Table of contents
* [Results](#results)
* [Acknowledgements](#acknowledgements)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Files](#files)

# Results

https://aiduate-dog-breed-classifier.herokuapp.com/

Test accuracy: 81.2201%

<img width="1186" alt="Untitled 3" src="https://user-images.githubusercontent.com/33467922/151484120-abdaf753-5483-470b-b54e-45000f2f2b05.png">


## To run the web app locally:

1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/

# Acknowledgements
* Udacity

# Technologies
* Python
* Html

# Libraries

```
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt  
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import ImageFile                  
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
```

# Files
- dog_app.ipynb - analysis and training/saving models

 - templates

    - master.html  - main page of web app

    - go.html  - classification result page of web app

- run.py  - Flask file that runs app

- OnePicPerDog.py - After training model and saving model, one image per breed from dog_image was kept to retrieve from the app for each predicted breed.

- static
    - img/dog.png - user image input to model

- models

   - train_classifier.py - load models
   
   - saved_models/ResNet50_model  - saved models 
   
   - haarcascades/haarcascade_frontalface_alt.xml
   
   - data
    
      - dog_images - not uploaded to github but link provided: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
   
   - extract_bottleneck_features.py - Needed for ResNet50 dog breed classifier model input preprocessing


