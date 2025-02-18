from utils import evaluate_model, prepare_data, apply_pca
import pandas as pd
from models import svm, knn, logistic_regression, decision_tree, random_forest
from visualization import EDA_graphs,  plot_svm, plot_knn , plot_lr, plot_dt, plot_rf

data_path = "Dataset/Skyserver.csv"

data = pd.read_csv(data_path)

# -------------------- Show Graphs --------------------
# EDA_graphs(data)

# -------------------- Split and Prepare Data --------------------
# data = data.drop(columns=['objid', 'rerun'])
# data = data.drop(columns=['fiberid', 'camcol', 'field', 'run', 'ra', 'objid', 'rerun'])
data = data.drop(columns=['fiberid', 'camcol', 'field', 'run', 'ra', 'objid', 'rerun', 'plate', 'mjd', 'specobjid', 'dec'])
# data = data.drop(columns=['fiberid', 'camcol', 'field', 'run', 'ra', 'objid', 'rerun', 'plate', 'specobjid', 'dec'])
x_train, x_test, y_train, y_test, x_val, y_val, classes, feature_names = prepare_data(data)

# -------------------- Apply PCA ----------------------
# x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, dim=9)
# x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, dim=5)
x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, dim=3)
# # -------------------- Run SVM --------------------
# svm_model, y_pred_train, y_pred_val, y_pred_test = svm(x_train, y_train, x_val, x_test)

# print("Train Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test Result:")
# evaluate_model(y_test, y_pred_test, classes)

# plot_svm(x_train, y_train, svm_model, feature_names)

# svm_model, y_pred_train, y_pred_val, y_pred_test = svm(x_train_pca, y_train, x_val_pca, x_test_pca)

# print("Train With PCA Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation With PCA Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test With PCA Result:")
# evaluate_model(y_test, y_pred_test, classes)
# -------------------- Run KNN --------------------

# knn_model, y_pred_train, y_pred_val, y_pred_test = knn(x_train, y_train, x_val, x_test)

# print("Train Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test Result:")
# evaluate_model(y_test, y_pred_test, classes)
# plot_knn(x_train, y_train, knn_model, feature_names)

# knn_model, y_pred_train, y_pred_val, y_pred_test = knn(x_train_pca, y_train, x_val_pca, x_test_pca)
# print("Train With PCA Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation With PCA Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test With PCA Result:")
# evaluate_model(y_test, y_pred_test, classes)

# # -------------------- Run Logistic Regression --------------------
# lr_model, y_pred_train, y_pred_val, y_pred_test = logistic_regression(x_train, y_train, x_val, x_test)

# print("Train Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test Result:")
# evaluate_model(y_test, y_pred_test, classes)
# plot_lr(x_train, y_train, lr_model, feature_names)

# lr_model, y_pred_train, y_pred_val, y_pred_test = logistic_regression(x_train_pca, y_train, x_val_pca, x_test_pca)

# print("Train With PCA Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation With PCA Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test With PCA Result:")
# evaluate_model(y_test, y_pred_test, classes)

# # -------------------- Run Desicion Tree --------------------
dt_model, y_pred_train, y_pred_val, y_pred_test = decision_tree(x_train, y_train, x_val, x_test)

print("Train Result:")
evaluate_model(y_train, y_pred_train, classes)
print("Validation Result:")
evaluate_model(y_val, y_pred_val, classes)
print("Test Result:")
evaluate_model(y_test, y_pred_test, classes)
plot_dt(x_train, y_train, dt_model, feature_names)

# dt_model, y_pred_train, y_pred_val, y_pred_test = decision_tree(x_train_pca, y_train, x_val_pca, x_test_pca)

# print("Train With PCA Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation With PCA Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test With PCA Result:")
# evaluate_model(y_test, y_pred_test, classes)

# -------------------- Run Random Forst --------------------
# rf_model, y_pred_train, y_pred_val, y_pred_test = random_forest(x_train, y_train, x_val, x_test)

# print("Train Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test Result:")
# evaluate_model(y_test, y_pred_test, classes)
# plot_rf(x_train, y_train, rf_model, feature_names)

# rf_model, y_pred_train, y_pred_val, y_pred_test = random_forest(x_train_pca, y_train, x_val_pca, x_test_pca)

# print("Train With PCA Result:")
# evaluate_model(y_train, y_pred_train, classes)
# print("Validation With PCA Result:")
# evaluate_model(y_val, y_pred_val, classes)
# print("Test With PCA Result:")
# evaluate_model(y_test, y_pred_test, classes)

