from sklearn.svm import SVC
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import GridSearchCV


def grid_search(model, parameters, X_train, y_train):
    grid_s_model = GridSearchCV(model, parameters, cv=3, verbose=3, n_jobs=-1)
    grid_s_model.fit(X_train, y_train)
    return grid_s_model

def grid_search_result(grid_s_model):
    results = grid_s_model.cv_results_
    for score, params in zip(results['mean_test_score'], results['params']):
        print(score, params)
    print("The best score is: ", grid_s_model.best_score_)
    print("The best paramters for that score is: ", grid_s_model.best_params_)
    return True