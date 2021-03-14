import numpy as np  
import pickle
from data.read_data import build_dataset
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

target_label = ["Not Smiling", "Smiling"]

# load data 
try: 
    with open("data/serialized_data", "rb") as f: 
        print("Serialized dataset already exists \n") 
        dataset = pickle.load(f) 
        data, labels = np.array(dataset["data"]), np.array(dataset["labels"])
        print("Successfuly load the serialized data \n ")
except: 
    print("Building the dataset ... \n") 
    data, labels = build_dataset()

# one example 

ex_img = data[0] 
plt.imshow(ex_img) 
plt.show()

# split data 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
print(f"X_train size : {X_train.shape}")
print(f"X_test size : {X_test.shape} \n")  

print("Looking at class imbalance")
# bar plot 
height = [list(labels).count(0), list(labels).count(1)] 
for i, h in enumerate(height): 
    print(f"{i}th classe : {h} occurencies  ")
plt.bar(target_label, height) 
plt.show()

# on-hot-encoding 

y_train = tf.keras.utils.to_categorical(y_train) 
y_test = tf.keras.utils.to_categorical(y_test)