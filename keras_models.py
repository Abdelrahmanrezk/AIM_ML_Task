
import time
import os
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Flatten
from tensorflow import keras
import time

TENSOR_DIR = os.path.join(os.curdir, "models", "dl_models", 'tensor_logs/')
MODELS_DIR = os.path.join(os.curdir, "models", "dl_models/") 


def get_run_tensor_logdir(run_hyper_params, tensor_dir=TENSOR_DIR):
    '''
    The function used to create dierction with the time we have run the model in, beside of that,
    concat to this time which hyperparameters we have used in this run, this time along with hyperparameters, 
    will help us compare result from different run with different hyperparamters, 
    as we used the tensorboard server as our vislization tool to help decide which model we can use.
    
    Argument:
    TENSOR_DIR: the tensor logs direction to be our direction for different runs.
    run_hyper_params: which hyper params we have used for this run.
    return
    TENSOR_DIR + run id(which run along with hyperparams to create subdirectory for)
    '''
    
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S_") + run_hyper_params
    return os.path.join(tensor_dir, run_id)


def keras_callbacks(word2vec_type, model_type, learning_rate):
     # Handle the different runs for the model to easily monitor from tensor board
    hyper_params = word2vec_type + "_" + model_type + "_learning_rate=" + str(learning_rate) + "_"
    run_log_dir = get_run_tensor_logdir(hyper_params, TENSOR_DIR)

    cb_tensor_board = keras.callbacks.TensorBoard(run_log_dir)

    # Once there is no progress stop the model and retrive the best weights
    cb_early_stop   = keras.callbacks.EarlyStopping(patience=5,monitor="val_loss",
                                                    restore_best_weights=True)

    # Handle problems that happend and long time training and save model check points
    file_path      = "/run_with_" + hyper_params +  "_model.h5"
    model_save_dir = MODELS_DIR + file_path
    cb_check_point = keras.callbacks.ModelCheckpoint(model_save_dir, monitor="binary_accuracy")

    # create list of callbacks we create
    callbacks = [cb_early_stop, cb_check_point, cb_tensor_board]

    return callbacks


# def lstm_keras_sequential_model_create(hid_num_neurons, max_len=64, number_of_features=100, dropout=.2):
#     '''
#     The function used to build your architecture of sequential model
    
#     Argument:
#         embed_create:  list that contain:
#             max_len       : Which fixed length we need to retrive the text in
#             number_of_features     : The number of features (Laten features we need for each word)
#     return:
#         model: The architecture of the model we have built
#     '''

#     # Create the Sequential model
#     model = keras.models.Sequential()
#     model.add(LSTM(hid_num_neurons, return_sequences=True, input_shape=(max_len, number_of_features)))
#   model.add(Dropout(dropout))
#   model.add(Flatten())
#   model.add(Dense(18, activation="softmax"))
#     return model

def lstm_no_batch_seqential_model_create(hid_num_neurons, max_len=64, number_of_features=100, dropout=.2):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(hid_num_neurons, return_sequences=True, input_shape=(max_len, number_of_features)))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(18, activation="softmax"))
    return model



def lstm_with_batch_model_create(hid_num_neurons, max_len=64, number_of_features=100, dropout=.2):

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(max_len, number_of_features)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LSTM(hid_num_neurons, return_sequences=True, input_shape=(max_len, number_of_features)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(18, activation="softmax"))
    return model

def seqential_model_compile(model, optimizer):
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=['accuracy'])
    return model



# def keras_embed_sequential_model_compile(model, optimizer_):
#     '''
#     The architecture of the model we have built need to be compiled to define which loss function the model will use,
#     beside of which optimization algorithm to update the weights to minimize the loss function. 
#     other optional parameters we can pass like binary_accuracy.
    
#     Argument:
#         model          : The model we have built
#         optimizer_     : Which optimizer we tend to use
#     return:
#         model : The architecture of the model we have built and compiled 
#     '''
#     model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer_, 
#                   metrics=['accuracy'])
#     return model


