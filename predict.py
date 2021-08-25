import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import argparse
import numpy as np


with open('label_map.json', 'r') as f:
    class_names = json.load(f)
    

image_size = 224

def process_image(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,(image_size,image_size))
    image /=255
    return image.numpy()

saved_keras_model_filepath = 'train_model.h5'
reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath,custom_objects={'KerasLayer':hub.KerasLayer})
print(reloaded_keras_model.summary())

def predict_class(image_path,model,top_k,class_names):
    top_k = int(top_k)
    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)
    prob_predict = reloaded_keras_model.predict(np.expand_dims(processed_test_image,axis =0))
    probs = prob_predict[0].tolist()
    values, indices= tf.math.top_k(probs, k=top_k)
    top_k_probs = values.numpy().tolist()
    top_k_class = indices.numpy().tolist()
    top_k_classes = [i+1 for i in top_k_class]
    top_k_classes = [str(i) for i in top_k_classes]
    top_k_class_labels = [class_names[label] for label in top_k_classes]
    print("These are the category names\n" , class_names)
    class_prob_dict = dict(zip(top_k_class_labels, top_k_probs))       
    print("\nTop "+str(top_k)+" classes along with associated probabilities :\n",class_prob_dict)
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("saved_model",help="Model Path", default="")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()
    
    predict_class(args.image_path, args.saved_model, args.top_k,class_names)
    
    


    

