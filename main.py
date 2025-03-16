from utils import evaluate_model, prepare_data, apply_pca
import pandas as pd
from models import svm, knn, logistic_regression, decision_tree, random_forest
from visualization import EDA_graphs,  plot_model_comparison, plot_svm, plot_knn , plot_lr, plot_dt, plot_rf
from sklearn.metrics import accuracy_score


data_path = "Dataset/Skyserver.csv"

data = pd.read_csv(data_path)

print('\nFirst five rows:')
print(data.head())  

print('\nData type:')
print(data.info())  

print('\nCheck for null values:')
print(data.isnull().sum())  

print('\nCheck for duplicates rows:')
print(data.duplicated().sum())  

print('\nNumber of unique values for each feature:')
print(data.nunique())
data = data.drop(columns=['objid', 'rerun']) # remove feature with 1 unique value

print('\nData statistics:')
print(data.describe())   

# -------------------- Show Graphs --------------------
EDA_graphs(data, type='histogram') # Plots the distribution of a numeric variable

EDA_graphs(data, type='DensityPlot') # Plots the density distribution of a continuous variable

EDA_graphs(data, type='boxplot') # The data shows varying levels of dispersion across features

EDA_graphs(data, type='heatmap') # represents a correlation matrix between various variables

EDA_graphs(data, type='scatterplot') # Used to compare pairs of features (u, g, r, i, z)

EDA_graphs(data, type='classes') # Bar Plot of Class Distribution

data = data.drop(columns=['specobjid']) # drop data with Correlation 1.0 with another feature (heatmap show that specobjid, plate with 1.0 Correlation)


# ---------------------------- Split & Prepare Data ----------------------------
x_train, x_test, y_train, y_test, x_val, y_val, classes, feature_names = prepare_data(data)

# ---------------------------- Define Models ----------------------------
models = {
    "SVM": (svm, plot_svm),
    "KNN": (knn, plot_knn),
    "Logistic Regression": (logistic_regression, plot_lr),
    "Decision Tree": (decision_tree, plot_dt),
    "Random Forest": (random_forest, plot_rf)
}

results = {"Without PCA": {}, "With PCA": {}}

# ---------------------------- Running Models ----------------------------

# ------- Running SVM -------
print("\n===== Running SVM =====")
model_name = "SVM"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running KNN -------
print("\n===== Running KNN =====")
model_name = "KNN"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running KNN (min_neighbors = 7) -------
print("\n===== Running KNN (min_neighbors = 7) =====")
model_name = "KNN"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test, min_neighbors=7)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Logistic Regression -------
print("\n===== Running Logistic Regression =====")
model_name = "Logistic Regression"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Decision Tree -------
print("\n===== Running Decision Tree =====")
model_name = "Decision Tree"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Random Forest -------
print("\n===== Running Random Forest =====")
model_name = "Random Forest"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))


# ---------------------------- Apply PCA & Run Models ----------------------------
x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, dim=9) # Reduce to 9 dimenation
feature_names = ['vec1', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']

# ------- Running SVM With PCA -------
print("\n===== Running SVM With PCA =====")
model_name = "SVM"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running KNN With PCA -------
print("\n===== Running KNN With PCA =====")
model_name = "KNN"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Logistic Regression With PCA -------
print("\n===== Running Logistic Regression With PCA =====")
model_name = "Logistic Regression"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Decision Tree With PCA -------
print("\n===== Running Decision Tree With PCA =====")
model_name = "Decision Tree"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Random Forest With PCA -------
print("\n===== Running Random Forest With PCA =====")
model_name = "Random Forest"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# -------------------- Plot Model Comparisons --------------------
plot_model_comparison(results, models)


# ---------------------------- Remove Not important features and repeat the process ----------------------------
data = pd.read_csv(data_path)
data = data.drop(columns=['fiberid', 'camcol', 'field', 'run', 'ra', 'plate', 'mjd', 'dec', 'specobjid', 'objid', 'rerun'])

x_train, x_test, y_train, y_test, x_val, y_val, classes, feature_names = prepare_data(data)

# ---------------------------- Running Models ----------------------------

# ------- Running SVM -------
print("\n===== Running SVM =====")
model_name = "SVM"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running KNN -------
print("\n===== Running KNN =====")
model_name = "KNN"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Logistic Regression -------
print("\n===== Running Logistic Regression =====")
model_name = "Logistic Regression"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Decision Tree -------
print("\n===== Running Decision Tree =====")
model_name = "Decision Tree"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Random Forest -------
print("\n===== Running Random Forest =====")
model_name = "Random Forest"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train, y_train, x_val, x_test)
print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train, y_train, model, feature_names)
results["Without PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ---------------------------- Apply PCA & Run Models ----------------------------
x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, dim=3) # Reduce to 3 dimenation
feature_names = ['vec1', 'vec2', 'vec3']

# ------- Running SVM With PCA -------
print("\n===== Running SVM With PCA =====")
model_name = "SVM"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running KNN With PCA -------
print("\n===== Running KNN With PCA =====")
model_name = "KNN"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Logistic Regression With PCA -------
print("\n===== Running Logistic Regression With PCA =====")
model_name = "Logistic Regression"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Decision Tree With PCA -------
print("\n===== Running Decision Tree With PCA =====")
model_name = "Decision Tree"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# ------- Running Random Forest With PCA -------
print("\n===== Running Random Forest With PCA =====")
model_name = "Random Forest"
model_func, plot_func = models[model_name]
model, y_pred_train, y_pred_val, y_pred_test = model_func(x_train_pca, y_train, x_val_pca, x_test_pca)
print("Train Result With PCA:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result With PCA:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result With PCA:")
evaluate_model(y_test, y_pred_test, classes)
plot_func(x_train_pca, y_train, model, feature_names)
results["With PCA"][model_name] = (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test))

# -------------------- Plot Model Comparisons --------------------
plot_model_comparison(results, models)