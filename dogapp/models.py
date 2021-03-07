# models.py - define model architectures.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


class DogCNN(keras.Model):
    def __init__(self):
        super(DogCNN, self).__init__(name="cnn")
        self.gap = GlobalAveragePooling2D(input_shape='2048')
        self.fc1 = Dense(133, activation='softmax')

    def call(self, inputs, training=False):
        x = self.gap(inputs)
        return self.fc1(x)
    
    def compile(self):
        super(DogCNN, self).compile()
        self.optimizer = Adam()
        self.loss_fn = CategoricalCrossentropy()
        self.accuracy_fn = Accuracy()

    def summary(self, input_shape):
        x_in = Input(shape=input_shape, name='X')
        summary = Model(inputs=x_in, outputs=self.call(x_in), name=self.name)
        return summary
