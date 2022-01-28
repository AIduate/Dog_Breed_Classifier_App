import cv2                
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from models.extract_bottleneck_features import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model#, Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense         
#from tensorflow.keras.callbacks import ModelCheckpoint       
#from tensorflow.keras import models     
from tqdm import tqdm
#from sklearn.datasets import load_files       
import tensorflow.keras.utils as utils
from glob import glob

    
def path_to_tensor(img_path):
    '''
    Method to convert to 4d tensor
    '''
    # loads RGB image as PIL.Image.Image type
    cv2.imwrite('static//img//dog.png', img_path)
    img = image.load_img('static//img//dog.png', target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    '''
    Method to run path to tensor to transform the image input
    '''
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    '''
    Method to run preprocessing and prediction on image
    '''
    # returns prediction vector for image located at img_path
    ResNet50_model2 = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model2.predict(img))



def face_detector(img_path):
    '''
    Method to return true if human face detected by pre-trained model
    '''
    face_cascade =  cv2.CascadeClassifier('models//haarcascades//haarcascade_frontalface_alt.xml')
    img = img_path#cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



def dog_detector(img_path):
    '''
    Method to return true if dog detected by pre-trained model
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# define function to load train, test, and validation datasets
# def load_dataset(path):
#     data = load_files(path)
#     dog_files = np.array(data['filenames'])
#     dog_targets = utils.to_categorical(np.array(data['target']), 133)
#     return dog_files, dog_targets


# def train_dog_model():
#     '''
#     Method to train resnet50 model for dog classification
#     '''
#     bottleneck_features = np.load('models//bottleneck_features//DogResnet50Data.npz')
#     train_ResNet50 = bottleneck_features['train']
#     valid_ResNet50 = bottleneck_features['valid']
#     test_ResNet50 = bottleneck_features['test']

#     ResNet50_model = Sequential()
#     ResNet50_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
#     ResNet50_model.add(Dense(133, activation='softmax'))

#     #ResNet50_model.summary()

#     ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#     checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
#                                verbose=1, save_best_only=True)

#     # load train, test, and validation datasets
#     train_files, train_targets = load_dataset('models/data/dog_images/train')
#     valid_files, valid_targets = load_dataset('models/data/dog_images/valid')
#     test_files, test_targets = load_dataset('models/data/dog_images/test')

#     ResNet50_model.fit(train_ResNet50, train_targets, 
#           validation_data=(valid_ResNet50, valid_targets),
#           epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
#     ResNet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
#     return ResNet50_model


def Resnet50_predict_breed(img_path):
    '''
    Method to execute classifier and return dog breed string given image input
    '''
    ResNet50_model = load_model("models/saved_models/ResNet50_model")#train_dog_model()
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet50_model.predict(bottleneck_feature)
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("models/data/dog_images/train/*/"))]
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]#.split('.')[1]

def dog_breed(img_path):
    '''
    Method to take in image and return a message with breed prediction or error message
    '''

    if face_detector(img_path): 

        results = Resnet50_predict_breed(img_path)
        #return 'Human detected that resembles a ' + results.replace('_', ' ')
        return results

    elif dog_detector(img_path):

        results = Resnet50_predict_breed(img_path)
        return results
        #return 'Predicted dog breed: ' + results#.replace('_', ' ')

    else:

        return 'Error, no human or dog detected'

if __name__ == '__main__':
    main()