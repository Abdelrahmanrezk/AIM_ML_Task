from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
import pickle
# Main libraries
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer  
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
import tensorflow as tf
import keras
from configs import *
from fetch_data import *
from features_extraction import *
from data_shuffling_split import *
from data_preprocess import *


def keras_vectorization(strat_train_set, strat_test_set, num_words):

    texts = list(strat_train_set['text']) + list(strat_test_set['text'])

    keras_tokenizer = Tokenizer(num_words=num_words, oov_token='UNK')
    keras_tokenizer.fit_on_texts(texts)
    print("The number of unique words are: ", len(keras_tokenizer.word_index))


    return keras_tokenizer

def prepare_data(data, keras_tokenizer, features_mode):
    x_train, x_val = Stratified_split_and_shuffle(data, "dialect", split_percentage=.02)
    x_train_text, x_val_text = list(x_train['text']), list(x_val['text'])
    y_train, y_val = x_train['dialect_l_encoded'].values, x_val['dialect_l_encoded'].values

    print("The number of trainin instances: ", len(x_train_text))
    print("The number of validation instances: ",len(x_val_text))
    print("The number of trainin labels : ", len(y_train))
    print("The number of validation labels : ", len(y_val))

    # Extract binary BoW features
    x_train_features = keras_tokenizer.texts_to_matrix(x_train_text, mode=features_mode)
    x_val_features   = keras_tokenizer.texts_to_matrix(x_val_text, mode=features_mode)

    y_train = np.asarray(y_train).astype('float32')
    y_val = np.asarray(y_val).astype('float32')

    x_train_features = x_train_features.astype('float32')
    x_val_features = x_val_features.astype('float32')

    print("The type of training features: ", type(x_train_features[0][0]))
    print("The type of validation features: ", type(x_val_features[0][0]))
    print("The type of training labels: ", type(y_train[0]))
    print("The type of validation labels: ", type(y_val[0]))
    return x_train_features, x_val_features, y_train, y_val


def keras_model(x_train, y_train, epochs, batch_size, lr):

    model = models.Sequential()
    # model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    # model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(18, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
            
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=.1)
    return history, model


def save_mode(model, file_path):

    pickle.dump(model, open(file_path, 'wb'))

    return True

def load_model(file_path):
    '''
    Load the fitted tf-idf file.
    Argument:
        file_path: The file contain the model weights.
    '''
    model = pickle.load(open(file_path, "rb"))
    return model


def graph_1(history, title_):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(title_)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    return history_dict

def graph_2(history, history_dict, title_):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.clf()   # clear figure
    acc_values = history_dict['binary_accuracy']
    val_acc_values = history_dict['val_binary_accuracy']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(title_)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()