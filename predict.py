# import libraries
import argparse
import numpy as np
import json
import tensorflow as tf 
import tensorflow_hub as hub
from PIL import Image

# Parser Object Creating
parser = argparse.ArgumentParser(description = "Predict Flower Classification")

# Declaring Arguments
parser.add_argument('--image_path', default = './test_images/cautleya_spicata.jpg',  help='Path of Image that you want to predict')
parser.add_argument('--model', default = 'saved_model.h5',help='Path of Model')
parser.add_argument('--topk',type = int , default = 5 ,help='Top K Reseluts needed')
parser.add_argument('--classes', default = 'label_map.json' ,help='clases names')

# Retuen The Values 
args = parser.parse_args()

img = args.image_path
model = args.model
topk = args.topk
classes = args.classes

#     loading model
model_load = tf.keras.models.load_model(model, custom_objects = {'KerasLayer':hub.KerasLayer})

# for printing the arguments
def predict_image():
    print(type(img))
    print(img, model , topk, classes)

with open( classes, 'r' ) as file:
    class_name = json.load(file)
    
# Image pre-processing    
def process_image(image): 
    img = tf.convert_to_tensor(image)
    img_resize = tf.image.resize(img, (224, 224))
    img_normal = img_resize / 255
    np_img = img_normal.numpy()
    return np_img

class_names = dict()
for key in class_name:
    class_names[str(int(key)-1)] = class_name[key]
# 
def predict(img_path, model, top_k ):
    
    img = Image.open(img_path)
    img_test = np.asarray(img)
    img_transform = process_image(img_test)
    
    img_redim = np.expand_dims(img_transform ,axis = 0)
    prediction = model_load.predict(img_redim)
    prediction = prediction.tolist()
    
    values, indices = tf.math.top_k(prediction,top_k)
    probs = values.numpy().tolist()[0]
    label = indices.numpy().tolist()[0]
    
    labele_names = [class_names[str(idd)] for idd in label]
    print(labele_names, '\n', probs)
    
    
if __name__ == '__main__':
    predict(img, model_load,topk)