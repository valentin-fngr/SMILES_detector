import numpy as np  
import pickle
from data.read_data import build_dataset
from miniVGG import MiniVGGNet
from utils.utils import stepwise_scheduler
import matplotlib.pyplot as plt
import os
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

# GPU device is now visible  
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


GPU = tf.config.list_physical_devices('GPU')
print(f"Available GPUs {GPU}")

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

# split data 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, shuffle=True)
print(f"X_train size : {X_train.shape}")
print(f"X_test size : {X_test.shape} \n")  
print(y_train[:50])

print("Looking at class imbalance")
# bar plot 
classes_weight = []

height = [list(labels).count(0), list(labels).count(1)] 
for i, h in enumerate(height): 
    print(f"{i}th classe : {h} occurencies  ")
    classes_weight.append(h / len(labels))
    print(f"class {i} Frequency : {classes_weight[i] * 100} % ")

# plt.bar(target_label, height) 
# plt.show()

# target reshaping 

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

# model instantiation
width, height, depth = X_train.shape[1:]
model = MiniVGGNet.build(width, height, depth, 2)
print("Model summary : \n")
model.summary()

# callbacks 
epochs = 50
batch_size = 32

checkpoint_filepath = "/training/model.h5"
log_dir  = "logs/fit/MiniVGGNet_50ep_bs32_0.01decay_noweights"
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.LearningRateScheduler(stepwise_scheduler, verbose=1), 
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor="val_loss", verbose=1, save_best_only=True
        ), 
    ]

# compile 
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) 
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

# fit 
h = model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=epochs,
    verbose=1, 
    validation_data=(X_test, y_test), 
    shuffle=True, 
    callbacks=callbacks
)

try: 
    model.load_weights(checkpoint_filepath)
    print("Loaded weights ! ")
except Exception as e: 
    print("-"*50) 
    print("Something went wrong ! ")
    print(e)
    print("-"*50)

# check tensorflow for learning curves .. 
print("Classification reports : \n")
predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=["Not Smiling", "Smiling"]))