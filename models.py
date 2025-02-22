from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV


def svm(x_train, y_train, x_val, x_test):

    svm_model = SVC(gamma='scale', random_state=42)

    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best model:", best_model)

    y_pred_train = best_model.predict(x_train)
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    return best_model, y_pred_train, y_pred_val, y_pred_test

def knn(x_train, y_train, x_val, x_test, min_neighbors=3):

    knn_model = KNeighborsClassifier(weights = 'uniform')

    param_grid = {
        'n_neighbors': range(min_neighbors, 15, 2),  
        'p': [1, 2, 3]  
    }

    grid_search = GridSearchCV(knn_model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best model:", best_model)

    y_pred_train = best_model.predict(x_train)
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    return best_model, y_pred_train, y_pred_val, y_pred_test

def logistic_regression(x_train, y_train, x_val, x_test):

    lr_model = LogisticRegression(max_iter=10000, solver='liblinear')    

    param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

    grid_search = GridSearchCV(lr_model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best model:", best_model)

    y_pred_train = best_model.predict(x_train)
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    return best_model, y_pred_train, y_pred_val, y_pred_test



def decision_tree(x_train, y_train, x_val, x_test):
    
    dt_model = DecisionTreeClassifier()
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': (2, 21),
        'min_samples_split': (2, 20), 
        'min_samples_leaf': (2, 20)
    }

    bayes_search = BayesSearchCV(dt_model, param_grid, cv=5)
    bayes_search.fit(x_train, y_train)

    best_model = bayes_search.best_estimator_

    print("Best model:", best_model)

    y_pred_train = best_model.predict(x_train)
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    return best_model, y_pred_train, y_pred_val, y_pred_test

def random_forest(x_train, y_train, x_val, x_test):
    
    rf_model = RandomForestClassifier(random_state=42)

    param_space = {
        # 'n_estimators': (10, 200),  
        # 'max_depth': (2, 20),  
        # 'min_samples_split': (2, 20), 
        # 'min_samples_leaf': (1, 20)
        'n_estimators': (50, 200),  
        'max_depth': (5, 15),  
        'min_samples_split': (2, 20), 
        'min_samples_leaf': (1, 20)    
    }

    bayes_search = BayesSearchCV(
        rf_model,
        param_space,
        n_iter=30,  
        random_state=42,
        cv = 5
    )
    
    bayes_search.fit(x_train, y_train)

    best_model = bayes_search.best_estimator_
    print("Best model:", best_model)

    y_pred_train = best_model.predict(x_train)
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    return best_model, y_pred_train, y_pred_val, y_pred_test
