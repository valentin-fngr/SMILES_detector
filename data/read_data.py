import os 
import numpy as np 
import cv2
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

def build_dataset():

    DIRECTORY = "SMILEs"
    
    data = [] 
    labels = []

    folders = ["negatives", "positives"] 

    for folder in folders: 
        if folder == "negatives": 
            label = 0 
        else: 
            label = 1
        folder_path = os.path.join(DIRECTORY, folder) 
        folder_image_path = os.path.join(folder_path, folder+"7")
        for f_name in os.listdir(folder_image_path): 
            img_path = os.path.join(folder_image_path, f_name)
            # convert to PIL format 
            try: 
                img = tf.keras.preprocessing.image.load_img(img_path) 
                img_array = tf.keras.preprocessing.image.img_to_array(img) 
                data.append(img_array)
                labels.append(label) 
            except: 
                pass

        print(f"DONE handling {folder} labeled data")
        print("-"*50)
        print()

    print("Dataset info : ")
    print(f"data size : {len(data)}") 
    print(f"image size : {data[0].shape}") 
    print("\n")


    # serializing data 
    print("Serializing Data ...  \n")
    with open("serialized_data", "wb") as f: 
        pickle.dump({
            "data" : data, 
            "label" : labels
        }, f) 
    
    print("Data Serialized \n ")


    return data, labels