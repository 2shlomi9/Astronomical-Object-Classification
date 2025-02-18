from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def svm(x_train, y_train, x_test):
    # מודל SVM עם ברירת מחדל
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(x_train, y_train)

    # חיזוי תוצאות
    y_pred_train = svm_model.predict(x_train)
    y_pred_test = svm_model.predict(x_test)

    print("Best model: SVC with default parameters")

    return svm_model, y_pred_train, y_pred_test


def knn(x_train, y_train, x_test):
    # מודל KNN עם ברירת מחדל
    knn_model = KNeighborsClassifier()
    knn_model.fit(x_train, y_train)

    # חיזוי תוצאות
    y_pred_train = knn_model.predict(x_train)
    y_pred_test = knn_model.predict(x_test)

    print("Best model: KNN with default parameters")

    return knn_model, y_pred_train, y_pred_test


def logistic_regression(x_train, y_train, x_test):
    # מודל לוגיסטי עם ברירת מחדל
    lr_model = LogisticRegression(max_iter=10000, solver='liblinear')    
    lr_model.fit(x_train, y_train)

    # חיזוי תוצאות
    y_pred_train = lr_model.predict(x_train)
    y_pred_test = lr_model.predict(x_test)

    print("Best model: Logistic Regression with default parameters")

    return lr_model, y_pred_train, y_pred_test


def decision_tree(x_train, y_train, x_test):
    # מודל עץ החלטה עם ברירת מחדל
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)

    # חיזוי תוצאות
    y_pred_train = dt_model.predict(x_train)
    y_pred_test = dt_model.predict(x_test)

    print("Best model: Decision Tree with default parameters")

    return dt_model, y_pred_train, y_pred_test


def random_forest(x_train, y_train, x_test):
    # מודל Random Forest עם ברירת מחדל
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(x_train, y_train)

    # חיזוי תוצאות
    y_pred_train = rf_model.predict(x_train)
    y_pred_test = rf_model.predict(x_test)

    print("Best model: Random Forest with default parameters")

    return rf_model, y_pred_train, y_pred_test