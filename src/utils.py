import tensorflow as tf
from keras import layers

def create_model():
    # Building NN
    model = tf.keras.models.Sequential([
        layers.Flatten(input_shape=[28,28]), 
        layers.Dense(300, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compiling Model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"]
    )
    return model


def train_model(model, x_train, y_train):
    model.fit(
        x_train,
        y_train,
        batch_size=1,
        epochs=30,
        validation_split=0.1 
    )
    model.save("latest.tf")