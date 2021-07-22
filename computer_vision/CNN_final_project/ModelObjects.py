
import keras
from keras.layers import Input, Dropout, Flatten
from keras.layers.core import Dense

class VGG16Scratch(object):
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest
        self.saved_name = "vgg16_scratch.h5"
        self.weights = None
        self.batch_size = 128
        self.num_classes = 11
        self.epochs = 10
        self.input_size = (48,48)

    def setup(self):
        self.model = keras.applications.vgg16.VGG16(weights=self.weights, include_top=False, classes=self.num_classes, input_shape=(48,48,3))
        model_input = keras.Input(shape=(self.input_size[0], self.input_size[1], 3), name='image_input')
        model_output = self.model(model_input)
        flattened_layer = Flatten(name='flatten')(model_output)
        dense_layer1 = Dense(self.input_size[0], activation='relu')(flattened_layer)
        dropped = Dropout(0.25, name="dropout1")(dense_layer1)
        dense_layer2 = Dense(self.input_size[0], activation='relu')(dropped)
        dense_layer3 = Dense(self.num_classes, activation='softmax')(dense_layer2)
        self.model = keras.models.Model(inputs=model_input, outputs=dense_layer3)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                        metrics=['accuracy'])

    def train(self):
        self.model.fit(self.Xtrain, self.Ytrain,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    validation_data=(self.Xtest, self.Ytest),
                    callbacks=None)
        score = self.model.evaluate(self.Xtest, self.Ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        self.model.save(self.saved_name)

class VGG16Pretrained(VGG16Scratch):
    # inherit the same model VGG16 model as scratch make sure you specify 'imagenet'
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest):
        VGG16Scratch.__init__(self, Xtrain, Ytrain, Xtest, Ytest)
        self.saved_name = 'vgg16_pretrained.h5'
        self.weights = 'imagenet'


class Sequential(object):
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest
        self.saved_name = "sequential.h5"
        self.batch_size = 128
        self.num_classes = 11
        self.epochs = 10
        self.input_size = (48,48)

    def setup(self):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(self.input_size[0], kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(self.input_size[0], self.input_size[1], 3)))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(1000, activation='relu'))
        self.model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                        metrics=['accuracy'])

    def train(self):
        self.model.fit(self.Xtrain, self.Ytrain,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    validation_data=(self.Xtest, self.Ytest),
                    callbacks=None)
        score = self.model.evaluate(self.Xtest, self.Ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        self.model.save(self.saved_name)
