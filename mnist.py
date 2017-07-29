# Larger CNN for the MNIST Dataset
import numpy
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
test = pd.read_csv('C:/Users/ElMoghazy/Desktop/test.csv')
training = pd.read_csv('C:/Users/ElMoghazy/Desktop/train.csv')
X_test = test
train = ["pixel" +str(x) for x in range(784)]
X_train = training[train]
y_train = training['label']
X_test = X_test.as_matrix()
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
num_classes = 10


# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(96, (11, 11), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3 ,3)))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3 ,3)))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))


    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, epochs=15, batch_size=200)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
result = model.predict(X_test)
result = numpy.argmax(result,axis = 1)
result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("C:/Users/ElMoghazy/Desktop/f.csv",index=False)