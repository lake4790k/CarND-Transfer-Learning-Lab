import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import  Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np

vgg = "https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b432_vgg-100/vgg-100.zip"
resnet = "https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b634_resnet-100/resnet-100.zip"
inception = "https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b498_inception-100/inception-100.zip"

# TODO: import Keras layers you need here

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    nb_classes = len(np.unique(y_train))
    inp_shape = X_train.shape[1:]

    model = Sequential()
    model.add(Flatten(input_shape=inp_shape))
    model.add(Dense(1024, activation='relu', W_regularizer=l2(0.01)))
    #model.add(Dropout(.5))
    model.add(Dense(nb_classes, activation='softmax'))

    adam = Adam()
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TODO: train your model here

    model.fit(X_train, y_train,
              batch_size=128,
              nb_epoch=1000,
              validation_data=(X_val, y_val),
              shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
