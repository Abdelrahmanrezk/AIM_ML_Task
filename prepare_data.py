

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

