# Reimplementation of CIFAR project using Convolutional Neural Networks based on the 4th assignment
# in Artificial Intelligence course in Uppsala university when revising knowledge related to CNN

# Flag indicating whether to print verbose messages.  (verbose ~ unnecessary)
from tabnanny import verbose
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
# Load the CIFAR 10 dataset. (50000 32x32 color training examples and 10000 test images)
from keras.datasets import cifar10 
# Numpy-related utilities
from keras.utils import np_utils
from keras import backend as K
from tensorflow.keras import datasets, layers, models, optimizers
# Stop training when a monitored metric has stopped improving
from keras.callbacks import EarlyStopping
# Callback to save the Keras model or model weights at some frequency
from keras.callbacks import ModelCheckpoint

def getModel(data):
    # CNNs implementation 
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fitModel(model, data):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    checkpoint = ModelCheckpoint('best_model.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
    model.fit(data.x_train, data.y_train, epochs = 10, validation_data = (data.x_test, data.y_test), callbacks=[es, checkpoint])
    return model


def runImageClassification(getModel=None, fitModel=None, seed=7):
    print('Preparing data ...')
    data = CIFAR(seed)

    # data.showImages()

    # Create model
    print('Creating model ....')
    model = getModel(data)

    # Fit model
    print('Fitting model ...')
    model = fitModel(model, data)

    # Evaluate on test data
    print('Evalutating model ...')
    score = model.evaluate(data.x_test, data.y_test, verbose = 0)
    print('Test accuracy: ', score[1])


class CIFAR:
    def __init__(self,seed=0):
        # Get and split data
        data = self.__getData(seed)
        self.x_train_raw=data[0][0]
        self.y_train_raw=data[0][1]
        self.x_valid_raw=data[1][0]
        self.y_valid_raw=data[1][1]
        self.x_test_raw=data[2][0]
        self.y_test_raw=data[2][1]
        # Record input/output dimensions
        self.num_classes=10
        self.input_dim=self.x_train_raw.shape[1:]
         # Convert data
        self.y_train = np_utils.to_categorical(self.y_train_raw, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid_raw, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test_raw, self.num_classes)
        self.x_train = self.x_train_raw.astype('float32')
        self.x_valid = self.x_valid_raw.astype('float32')
        self.x_test = self.x_test_raw.astype('float32')
        self.x_train  /= 255
        self.x_valid  /= 255
        self.x_test /= 255
        # Class names
        self.class_names=['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    def __getData (self,seed=0):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return self.__shuffleData(x_train,y_train,x_test,y_test,seed)
    
    def __shuffleData (self,x_train,y_train,x_test,y_test,seed=0):
        # Training and validation sets percentage
        tr_perc=.75
        va_perc=.15
        x=np.concatenate((x_train,x_test))
        y=np.concatenate((y_train,y_test))
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        indices = np.random.permutation(len(x))
        tr=round(len(x)*tr_perc)
        va=round(len(x)*va_perc)
        self.tr_indices=indices[0:tr]
        self.va_indices=indices[tr:(tr+va)]
        self.te_indices=indices[(tr+va):len(x)]
        x_tr=x[self.tr_indices,]
        x_va=x[self.va_indices,]
        x_te=x[self.te_indices,]
        y_tr=y[self.tr_indices,]
        y_va=y[self.va_indices,]
        y_te=y[self.te_indices,]
        return ((x_tr,y_tr),(x_va,y_va),(x_te,y_te))
    # Print 25 random figures from the validation data
    def showImages(self):
        images=self.x_valid_raw
        labels=self.y_valid_raw
        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        plt.figure(figsize=(10,10))
        indices=np.random.randint(0,images.shape[0],25)
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[indices[i]], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays, 
            # which is why we need the extra index
            plt.xlabel(class_names[labels[indices[i]][0]])
        plt.show()
def main():
    runImageClassification(getModel,fitModel)
if __name__== "__main__" :
    main()