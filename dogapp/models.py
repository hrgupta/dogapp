# models.py - define model architectures.
import keras
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy
from keras.models import Model
from keras.optimizers import Adam


class DogCNN(keras.Model):
    def __init__(self):
        super(DogCNN, self).__init__(name="cnn")
        self.gap = GlobalAveragePooling2D()
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
