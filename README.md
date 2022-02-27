# Dog_Breed_Classifier_App

## Table of contents
* [Definition](#definition)
* [EDA](#eda)
* [Algorithms](#algorithms)
* [Pre-Processing](#pre-processing)
* [App](#app)
* [Results](#results)
* [Conclusion](#conclusion)
* [Improvement](#improvement)
* [Acknowledgements](#acknowledgements)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Files](#files)

# Definition

The objective of this project is to build a web app that will detect human and dog faces in a user provided image and if detected, yield a breed that the image resembles.
In order to accomplish this, pre-trained classifiers are used: HAAR CASCADE for face detection and ResNet50 CNNs for dog detection and ultimate breed classification.

## Problem Statement

Given any image provided from a user, if there is a dog or a human, classify the image by breed most resembled.

# EDA

## Distribution of breed training data:
![train_dist](https://user-images.githubusercontent.com/33467922/155903203-12ec54ea-7056-4565-a53f-ee316e5c4115.png)

## Sample training images:

### Alaskan Malamute:
![Alaskan_malamute_00299](https://user-images.githubusercontent.com/33467922/155903294-f815a53a-a7b8-49b2-9554-24c494597701.jpg)

### Basset Hound:
![Basset_hound_01033](https://user-images.githubusercontent.com/33467922/155903300-b3fccee7-37cb-4036-a40d-943c768e431d.jpg)

## Distribution of breed test data:
![test_dist](https://user-images.githubusercontent.com/33467922/155903199-0f8f4503-e5bf-4ae2-9cdd-ba3109424fa0.png)

## Sample test images

### Alaskan Malamute
![Alaskan_malamute_00309](https://user-images.githubusercontent.com/33467922/155903379-4d17f1a4-f858-4a59-aa02-5c6f97e9a920.jpg)

### Basset Hound
![Basset_hound_01034](https://user-images.githubusercontent.com/33467922/155903407-745d5893-0a72-43b5-a22e-844352c34c25.jpg)

## Top breeds with change in distribution between train and test set
<img width="569" alt="Top change in test dist" src="https://user-images.githubusercontent.com/33467922/155903068-cd378c4f-cba5-4d2e-864f-de0daf2c22fb.png">

# Algorithms

## ResNet-50 model2
The pre-trained ResNet-50 model used to detect dogs in images was "loaded with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image." [1]

Information on the OpenCV Haar Cascade Face Detector model: 
- "Haar-like features are digital image features used in object recognition. ... A Haar-like feature considers adjacent rectangular regions at a specific location in a detection window, sums up the pixel intensities in each region and calculates the difference between these sums." [3]
- Thousands of features regarding edges, lines and rectangles are "grouped into different stages of classifiers and applied one-by-one", the cascade.

## ResNet-50 model

### Transfer learning 

Technique was used from pre-trained ResNet50 model to classify an image of a human or dog into a dog breed category.
"The model uses the the pre-trained ResNet50 model as a fixed feature extractor, where the last convolutional output of ResNet50 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax." [1]

```
ResNet50_model = Sequential()
ResNet50_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet50_model.add(Dense(133, activation='softmax'))
```

<img width="521" alt="ResNet50ModelSummary" src="https://user-images.githubusercontent.com/33467922/155898648-ea3631be-bb6b-4124-b813-aadf46602eeb.png">

## Metrics

" Before training a model, you need to configure the learning process, which is done via the compile method. It receives three arguments:
1. an optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the Optimizer class.
2. a loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function. See: objectives.
3. a list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric (only accuracy is supported at this point), or a custom metric function." [5]

```ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])```

Saved the model with the best validation loss to be used in algorithm and in the web app.
    - Accuracy is used due to Keras documentation suggestion: "For any classification problem you will want to set this to metrics=['accuracy']" [5]
    
** Performance on final breed classification model is done with accuracy to mirror guidance of ipython notebook but since there is class imbalance seen in EDA above, a better performance metric to evaluate performance would be f measure as it would be less bias towards higher distributed classes and balances between precision and recall.

# Pre-Processing

- Before using the face detector, images must be converted to grayscale
- "Keras ResNet50 CNN models require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape: (nb_samples, rows, columns, channels), where nb_samples corresponds to the total number of images (or samples), and rows, columns, and channels correspond to the number of rows, columns, and channels for each image, respectively." [1]
    - "The path_to_tensor function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN. The function first loads the image and resizes it to a square image that is  pixels. Next, the image is converted to an array, which is then resized to a 4D tensor. In this case, since we are working with color images, each image has three channels." (1,224,224,3) [1]
    -  "The paths_to_tensor function takes a numpy array of string-valued image paths as input and returns a 4D tensor" (nb_samples,224,224,3). "Here, nb_samples is the number of samples, or number of images, in the supplied array of image paths. It is best to think of nb_samples as the number of 3D tensors (where each 3D tensor corresponds to a different image) in our dataset" [1]
    -  "Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing." [1]
    -  "First, the RGB image is converted to BGR by reordering the channels. All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as  and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image." [1] (This is implemented by preprocess_input function linked in acknowledgements section [4])

- For the transfer learning portion, We created the bottleneck features corresponding to our own additional train, test, and validation sets (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

```
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']
```

# App
- Load saved models (except for haar face cascade) to execute on user provided images
- Added form to receive user input image
- Save user image and display after prediction
- Save one picture per dog breed to load after prediction in addition to user image
    - Use predicted breed to dynamically locate corresponding dog picture to display

# Results

https://aiduate-dog-breed-classifier.herokuapp.com/

Test accuracy: 81.2201%
- Accuracy is not the best performance metric here because there is class imbalance between some of the breeds. (See EDA charts for visual of this)

<img width="1186" alt="Untitled 3" src="https://user-images.githubusercontent.com/33467922/151484120-abdaf753-5483-470b-b54e-45000f2f2b05.png">


## To run the web app locally:

1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/

# Improvement

I hope to improve the project by: 
- Adding more training data to the breed classifier + more breeds (my dog at home pictured in the app screenshot is an American Bully and is not a class currently)
- Study and explore different CNN framework that may yield better performance
- Improve Heroku hosted app by adding logic to catch error display in app error message for images where human or dog is not detected instead of Heroku app error
- Evaluate performance more accurately

# Acknowledgements
1. Udacity
2. https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html
3. https://en.wikipedia.org/wiki/Haar-like_feature
4. https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
5. https://faroit.com/keras-docs/1.0.6/getting-started/sequential-model-guide/

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


