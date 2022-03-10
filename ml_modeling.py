from sklearn.svm import SVC
from gensim.models import Word2Vec
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
import keras
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

def save_mode(model, file_path):

    pickle.dump(model, open(file_path, 'wb'))

    return True


def f1_score_display(model, x, y):
    
    predict                     = model.predict(x)
    print("===================== Validate Result =====================")
    print("F1 score is: ", f1_score(y, predict, average='micro'))

    return True

def model_fit(model, X_train, y_train, model_path_to_save):
    model.fit(X_train, y_train)
    save_mode(model, model_path_to_save)
    return model

def classifier(X_train, y_train):

    vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    svm_clf = svm.LinearSVC(C=0.1)
    vec_clf.fit(X_train, y_train)
    joblib.dump(vec_clf, 'saved_model/svmClassifier.pkl', compress=3)

    return vec_clf


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
