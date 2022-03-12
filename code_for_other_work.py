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
