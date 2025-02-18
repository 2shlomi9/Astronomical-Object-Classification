from utils import evaluate_model, prepare_data, apply_pca
import pandas as pd
from models import svm, knn, logistic_regression, decision_tree, random_forest
from visualization import EDA_graphs,  plot_model_comparison, plot_svm, plot_knn , plot_lr, plot_dt, plot_rf
from sklearn.metrics import accuracy_score

data_path = "Dataset/Skyserver.csv"

data = pd.read_csv(data_path)

# ------------------ Check And Clean Data -------------------
print("First 5 rows in the data:\n",data.head())  
print("\nFeature type & if there are NULL values:\n",data.info())  

print("\nDATA describtion:\n",data.describe())   

print("\nNumber of duplicated values:\n",data.duplicated().sum())  

print("\nNumber of unique values for each feature:\n",data.nunique())

print("\nDrop feature with 1 unique value: ('objid', 'rerun')\n")

data.drop(columns=['objid', 'rerun'])

# EDA_graphs(data)

# -------------------- Prepare Data --------------------
x_train, x_test, y_train, y_test, x_val, y_val, classes, feature_names = prepare_data(data)

# -------------------- Define Models --------------------
models = {
    "SVM": (svm, plot_svm),
    "KNN": (knn, plot_knn),
    "Logistic Regression": (logistic_regression, plot_lr),
    "Decision Tree": (decision_tree, plot_dt),
    "Random Forest": (random_forest, plot_rf)
}

results = {"Without PCA": {}, "With PCA": {}}

for model_name, (model_func, plot_func) in models.items():
    print(f"\n===== Running {model_name} =====")
    model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
    print("Train Result:")
    evaluate_model(y_train, y_pred_train, classes)
    print("Validation Result:")
    evaluate_model(y_val, y_pred_val, classes)
    print("Test Result:")
    evaluate_model(y_test, y_pred_test, classes)
    plot_func(x_train, y_train, model, feature_names)  
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    results["Without PCA"][model_name] = (acc_train, acc_test)

# -------------------- Apply PCA --------------------
x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, dim=9) # Reduce to 9 dimenation

for model_name, (model_func, plot_func) in models.items():
    print(f"\n===== Running {model_name} With PCA =====")
    model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
    print("Train Result With PCA:")
    evaluate_model(y_train, y_pred_train, classes)
    print("Validation Result With PCA:")
    evaluate_model(y_val, y_pred_val, classes)
    print("Test Result With PCA:")
    evaluate_model(y_test, y_pred_test, classes)
    results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# -------------------- Plot Model Comparisons --------------------
plot_model_comparison(results, models)






# ---------------------- Delete Less Important Features (According to the results from above) ----------------------
print('===== Delete Less Important Features (According to the results from above) =====')

data = pd.read_csv(data_path)
data.drop(columns=['fiberid', 'camcol', 'field', 'run', 'ra', 'objid', 'rerun'])

# -------------------- Prepare Data --------------------
x_train, x_test, y_train, y_test, x_val, y_val, classes, feature_names = prepare_data(data)

# -------------------- Repeat Process --------------------

for model_name, (model_func, plot_func) in models.items():
    print(f"\n===== Running {model_name} Without PCA =====")
    model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_test)
    print("Train Result:")
    evaluate_model(y_train, y_pred_train, classes)
    print("Validation Result:")
    evaluate_model(y_val, y_pred_val, classes)
    print("Test Result:")
    evaluate_model(y_test, y_pred_test, classes)
    plot_func(x_train, y_train, model, feature_names)  
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    results["Without PCA"][model_name] = (acc_train, acc_test)

# -------------------- Apply PCA --------------------
x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, dim=5) # Reduce to 5 dimenation

for model_name, (model_func, plot_func) in models.items():
    print(f"\n===== Running {model_name} With PCA =====")
    model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
    print("Train Result With PCA:")
    evaluate_model(y_train, y_pred_train, classes)
    print("Validation Result With PCA:")
    evaluate_model(y_val, y_pred_val, classes)
    print("Test Result With PCA:")
    evaluate_model(y_test, y_pred_test, classes)
    results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# -------------------- Plot Model Comparisons --------------------
plot_model_comparison(results, models)