import tensorflow as tf
from keras import layers

import matplotlib.pyplot as plt

"""
    The main difference between loading image dataset with keras
    instead of scikitLearn is that in keras, each of them is represented
    as a 28x28 array rather than an array of size 784.
    +
    In keras, pixel intensities are represented as intergers from 0 to 255
    instead of floats from 0.0 to 255.0
"""

# Load fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Load train and test data
#               60_000 samples                 10_000 samples
(x_fashion_train, y_fashion_train), (x_fashion_test, y_fashion_test) = fashion_mnist.load_data()

# 55_000 samples
x_train = x_fashion_train / 255.0
y_train = y_fashion_train
x_test = x_fashion_test / 255.0

# Fashion MNIST Data Labels glossaire
# The first column is the label
class_names = [
    "T-shirt/top", # 0 
    "Trouser",     # 1
    "Pullover",    # 2
    "Dress",       # 3
    "Coat",        # 4
    "Sandal",      # 5
    "Shirt",       # 6
    "Sneaker",     # 7
    "Bag",         # 8
    "Ankle boot"   # 9
]

# Building our neural network
model = tf.keras.models.Sequential([

    # For preprocessing
    layers.Flatten(input_shape=[28,28]), # Transform 28x28 image data into 1D array because default is 2D

    # Creating hidden layer - Manages its own weight matrix with connections weight between neurons. 
    # It manages a Bias vector too * once per neuron

    # ReLU to make sure that everything is greater than 0 ( No negative numbers )
    
    layers.Dense(300, activation='relu'), # Create a Hidden Layer of 300 neurons
    layers.Dense(100, activation='relu'), # Create a Hidden Layer of 100 neurons

    # The Output Layer - Last Layer
    layers.Dense(10, activation='softmax') # Normalize the Output with SoftMax
])

"""
    First Hidden Layer - 
    * 235500 params ( Alot of flexibility ) because * 
    235200 = 784 * 300 (Connections Weights ) 
    - 
    235500 = 235200 + 300 ( Bias )
"""
# print(model.summary())

# Compiling the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

# Start training the model
# model.fit(
#     x_train,
#     y_train,
#     1,          # Batch Size / 1 Batch size = all the 55_000 samples, 32 batch size = 1719 samples
#     epochs=30,  # Number of time it will go through all the samples
#     validation_split=0.1 # Take the last 10% of samples and use them to validate
# )

print(x_train)
print("Last 3 ")
print(x_train[:3])

x_toPredict = x_test[:1]
# print(y_fashion_test[:3])
# y_proba = model.predict(x_toPredict) 
# print(y_proba.round(2))


# Evaluate model with test data
# model.evaluate(
#     x_test,
#     y_fashion_test,
#     batch_size=1
# )

# weights, biases = model.layers[1].get_weights() # First Hidden Layer

# print(f"Weights : {weights.shape} - Biases : {biases.shape}")

# Display The Clothe image

# plt.imshow(x_train[:3])

# The first column is the label
# plt.title(class_names[y_train[:3]])

# plt.show()