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

ex_img = data[0] 
plt.imshow(ex_img) 
plt.show()

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

plt.bar(target_label, height) 
plt.show()

# one-hot-encoding 

y_train = tf.keras.utils.to_categorical(y_train) 
y_test = tf.keras.utils.to_categorical(y_test)

# model instantiation
width, height, depth = X_train.shape[1:]
model = MiniVGGNet.build(width, height, depth, 2)
print("Model summary : \n")
model.summary()

# callbacks 
epochs = 50
batch_size = 32

checkpoint_filepath = "./training/checkpoints"
log_dir  = "logs/fit/MiniVGGNet_50ep_bs32_decay"
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(stepwise_scheduler), 
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor="val_loss", verbose=1, save_best_only=True
        ), 
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

# compile 
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) 
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

# fit 
# model.fit(
#     x=X_train, 
#     y=y_train, 
#     batch_size=batch_size, 
#     epochs=epochs
#     versbose=1, 
#     validation_data=(X_test, y_test), 
#     shuffle=True, 
#     callbacks=callbacks
# )