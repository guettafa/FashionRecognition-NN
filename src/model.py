import tensorflow as tf
import matplotlib.pyplot as plt
from utils import create_model, train_model

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_fashion_train, y_fashion_train), (x_fashion_test, y_fashion_test) = fashion_mnist.load_data()

x_train = x_fashion_train / 255.0
y_train = y_fashion_train

x_test = x_fashion_test / 255.0

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

model = create_model()

model.evaluate(
    x_test,
    y_fashion_test,
    batch_size=1
)

train_model(
    model, 
    x_train, 
    y_train
)

# Load trained model
# model = tf.keras.models.load_model("./latest.tf")

x_toPredict = x_test[:1]
y_prob = model.predict(x_toPredict)
print(y_prob.round(2))

plt.imshow(x_test[0])
plt.show()

