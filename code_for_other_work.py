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
        if (i+1) % 30000 == 0:
            print("We have processed: ", i+1)
        i +=1
# list of each reviw which will be appended to embedding matrix
        one_sentence_list       = [] 
        for word in text:
            try:
                print("1111111111111111111111111111111111")
                word                = word_to_vec_model.wv[word]
                one_sentence_list.extend(word)
            except:
                pass

# make padding for small strings
        zero_pad                = np.zeros(number_of_features*max_len_str-len(one_sentence_list))
        zero_pad                = list(zero_pad)
# apply the padding
        one_sentence_list.extend(zero_pad)

        embedding_matrix[index] = one_sentence_list

    X_train_embed_matrix = pad_sequences(X_train_embed_matrix, maxlen=max_len_str, padding='post')
    return embedding_matrix


def pad_trunc(data, maxlen):
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
        
    for sample in data:
        if len(sample) > maxlen:
            temp =sample[:maxlen]
        elif len(sample) < maxlen:
            temp=sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data



def grid_search(model, parameters, X_train, y_train):
    grid_s_model = GridSearchCV(model, parameters, cv=3, verbose=1)
    grid_s_model.fit(X_train, y_train)
    return grid_s_model


def grid_search_result(grid_s_model):
    results = grid_s_model.cv_results_
    for score, params in zip(results['mean_test_score'], results['params']):
        print(score, params)
    print("="*50)
    print("The best score is: ", grid_s_model.best_score_)
    print("The best paramters for that score is: ", grid_s_model.best_params_)
    return True





def bow_tf_idf_model(param_distribs, X_train_tfidf, y_train, keras_cls_hyperparams):
    '''
    The funcation used to train keras model aling with sklearn
    Arguments:
        param_distribs         : Search for best hyperparamters
        X_train_tfidf          : The data to train one
        y_train                : desired output for each instance input
        keras_cls_hyperparams  : keras hyper paramters to use 
    '''
    batch_size, epochs, validation_split = keras_cls_hyperparams
    
    keras_cls = keras.wrappers.scikit_learn.KerasClassifier(build_model)

    rnd_search_cv = RandomizedSearchCV(keras_cls, param_distribs, n_iter=10, cv=3)

    rnd_search_cv.fit(X_train_tfidf, y_train, epochs=50, validation_split=.03)
    print("="*50)
    print(rnd_search_cv.best_params_['learning_rate'])
    model, history = Keras_sgd_classifer(rnd_search_cv.best_params_['learning_rate'], 
                                     epochs, batch_size, validation_split, 10, X_train_tfidf, y_train, "BOW Tf-idf")

    return rnd_search_cv, model, history




def bow_tfidf_prepare_data(row_to_load, vocab_size, text_col):
    '''
    The function used to build the index matrix to pass latter to our model
    Argument:
        row_to_load : Number of rows we need from this csv file
        vocab_size  : Number of uniqe vocabs we used
        text_col    : Which column you need as we deigned for text classifcation just retrive column text
    Return:
        the trained and test
    '''
    df_file = get_number_of_rows_from_csv(row_to_load)

    comment_text, target = fetch_text_and_target(df_file, text_col, target_col)
    del df_file


    tokenizer            = Tokenizer(num_words=vocab_size, oov_token="UNK")
    tfidf_txt            = text_to_tfidf(tokenizer, comment_text)
 

    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(tfidf_txt, target, 
                                                                test_size=0.02, random_state=42)


    print(X_train_tfidf.shape)
    print(y_train.shape)

    print(X_test_tfidf.shape)
    print(y_test.shape)
    
    return X_train_tfidf, y_train, X_test_tfidf, y_test


def build_model(n_hidden=0, n_neurons=0, learning_rate=3e-3, input_shape=[vocab_size]):
    '''
    This is based on keras wrapper to run sklearn with keras to search for best hyper params
    '''
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.SGD(lr=learning_rate),
                 metrics=['accuracy'])
    return model

