from gensim.models import Word2Vec
import numpy as np

from configs import *
from fetch_data import *
from features_extraction import *
from data_shuffling_split import *
from data_preprocess import *




def prepare_data(data):
    x_train, x_val = Stratified_split_and_shuffle(data, "dialect", split_percentage=.02)
    x_train_text, x_val_text = list(x_train['text']), list(x_val['text'])
    y_train, y_val = x_train['dialect_l_encoded'].values, x_val['dialect_l_encoded'].values

    print("The number of trainin instances: ", len(x_train_text))
    print("The number of validation instances: ",len(x_val_text))
    print("The number of trainin labels : ", len(y_train))
    print("The number of validation labels : ", len(y_val))

    return x_train_text, x_val_text, y_train, y_val



def load_model(file_path):
    '''
    Load the fitted tf-idf file.
    Argument:
        file_path: The file contain the model weights.
    '''
    model = pickle.load(open(file_path, "rb"))
    return model


def save_mode(model, file_path):

    pickle.dump(model, open(file_path, 'wb'))

    return True




def load_word2vec_model(model_path):
    word_to_vec_model = Word2Vec.load(model_path)
    return word_to_vec_model



def ML_text_to_matrix_using_word2vec(word_to_vec_model, text_list, number_of_features, max_len_str):
    '''
    The function used to build our word2vec matrix for the training and testing data.
    Argument:
            List of string each of them is list of words
            the word_to_vec_model model
            number of features you apply to word2vec model
            number of words of greatest string in your reviews
    return:
        embedding matrix that can apply to machine learning algorithms
    '''
    embedding_matrix            = np.zeros((len(text_list), number_of_features*max_len_str), dtype=np.float16) # largest sentence and 5 fetures
    print("The shape of matrix", embedding_matrix.shape)
#loop over each review
    i = 0
    for index,text in enumerate(text_list):
        if (i+1) % 30000:
            print("We have processed: ", i+1)
        i +=1
# list of each reviw which will be appended to embedding matrix
        one_sentence_list       = [] 
        for word in text:
            try:
                word                = word_to_vec_model[word]
                one_sentence_list.extend(word)
            except:
                pass
            
# make padding for small strings
        zero_pad                = np.zeros(number_of_features*max_len_str-len(one_sentence_list))
        zero_pad                = list(zero_pad)
    
# apply the padding
        one_sentence_list.extend(zero_pad)
        embedding_matrix[index] = one_sentence_list
    return embedding_matrix