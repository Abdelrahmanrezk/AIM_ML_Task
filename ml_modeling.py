

# Main libraries 
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from datetime import datetime
import numpy as np
import os


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


def model_fit(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def f1_score_result(model, x, y):
    
    predict                     = model.predict(x)
    micro_f1 = f1_score(y, predict, average='micro')

    print("===================== Validate Result =====================")
    print("F1 score is: ", micro_f1)

    return np.round(micro_f1, 3)




def ml_classifer_pipeline(model, X_train, y_train, X_val, y_val, used_word2vec_path, model_path_to_save):
    start                                       = datetime.now()
    model = model_fit(model, X_train, y_train)
    micro_f1 = f1_score_result(model, X_val, y_val)
    model_name = type(model).__name__
    model_path_to_save = os.path.join(model_path_to_save, used_word2vec_path)
    model_path_to_save = model_path_to_save + model_name + "_" + "_f1_" + str(micro_f1) + "_ml.sav" 
    _ = save_mode(model, model_path_to_save)
    print ("It takes to run: ", datetime.now() - start)
    return model

def voting_models():
    svc_clf_model = LinearSVC(C=0.5,  verbose=1)
    lg_clf_model  = LogisticRegression(penalty='l2', C=1, multi_class='multinomial', solver='lbfgs', verbose=1)
    dec_tree_clf_model  = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
    estimators = [("svc_clf_model", svc_clf_model), ("lg_clf_model", lg_clf_model), ("dec_tree_clf_model", dec_tree_clf_model)]
    return estimators



# def soft_vot():

# def ml_voting_classifer(estimators, voting_type="hard", X_train, y_train, X_val, y_val, 
#                 used_word2vec_path, model_path_to_save):
    

#     if voting_type =="hard":
#         model = VotingClassifier(estimators=estimators, voting="hard")
#         _ = ml_classifer_pipeline(model, X_train, y_train, X_val, y_val, used_word2vec_path, model_path_to_save)
#     else:
#         model = VotingClassifier(estimators=estimators, voting="soft")
#         model = model_fit(model, X_train, y_train)


#     model = model_fit(model, X_train, y_train)
# def classifier(X_train, y_train):

#     svm_clf = svm.LinearSVC(C=0.1)
#     vec_clf.fit(X_train, y_train)
#     joblib.dump(vec_clf, 'saved_model/svmClassifier.pkl', compress=3)

#     return vec_clf


# def keras_sgd(X_train, y_train):
#     model = keras.models.Sequential()
#     model.add(keras.layers.Dense(18, activation='softmax'))
#     model.compile(loss="sparse_categorical_crossentropy",
#              optimizer="sgd",
#              metrics="accuracy")
#     history = model.fit(X_train, y_train, batch_size=128, epochs=30, validation_split=.02)

#     return model
# def voting_classifier(estimators, vote_type):

#     if vote_type == "hard":

#     # else:
