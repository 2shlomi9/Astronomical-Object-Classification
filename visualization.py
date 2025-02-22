import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.inspection import permutation_importance

def plot_model_comparison(results, models):
    labels = list(models.keys())

    acc_train_no_pca = [results["Without PCA"][m][0] for m in labels]
    acc_test_no_pca = [results["Without PCA"][m][1] for m in labels]
    acc_train_pca = [results["With PCA"][m][0] for m in labels]
    acc_test_pca = [results["With PCA"][m][1] for m in labels]

    x = np.arange(len(labels))  
    width = 0.2  

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, acc_train_no_pca, width, label="Train (No PCA)")
    ax.bar(x, acc_test_no_pca, width, label="Test (No PCA)")
    ax.bar(x + width, acc_train_pca, width, label="Train (With PCA)")
    ax.bar(x + 2 * width, acc_test_pca, width, label="Test (With PCA)")

    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison (With/Without PCA)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(axis="y")

    plt.show()


def plot_overfitting_analysis(results, models):
    labels = list(models.keys())

    acc_train_no_pca = [results["Without PCA"][m][0] for m in labels]
    acc_test_no_pca = [results["Without PCA"][m][1] for m in labels]
    acc_train_pca = [results["With PCA"][m][0] for m in labels]
    acc_test_pca = [results["With PCA"][m][1] for m in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(acc_train_no_pca, acc_test_no_pca)
    for i, txt in enumerate(labels):
        axes[0].annotate(txt, (acc_train_no_pca[i], acc_test_no_pca[i]))

    axes[0].plot([0, 1], [0, 1], 'r--')
    axes[0].set_xlabel("Train Accuracy")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Overfitting Analysis (No PCA)")

    axes[1].scatter(acc_train_pca, acc_test_pca)
    for i, txt in enumerate(labels):
        axes[1].annotate(txt, (acc_train_pca[i], acc_test_pca[i]))

    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_xlabel("Train Accuracy")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Overfitting Analysis (With PCA)")

    plt.tight_layout()
    plt.show()


def EDA_graphs(data, type='histogram'):

    col=[ 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'field', 'redshift', 'plate', 'mjd', 'fiberid', "specobjid"]

    if type == 'histogram':  # Plots the distribution of a numeric variable
        plt.figure(figsize=(15, 15))  
        for i in range(len(col)):  
            plt.subplot(4,4,i+1)
            sns.histplot(data,x=col[i],hue="class",element="step")
            plt.xlabel(col[i])
        plt.show()

    if type == 'DensityPlot':  # Plots the density distribution of a continuous variable
        plt.figure(figsize=(15, 15))  
        for i in range(len(col)):
            plt.subplot(4, 4, i+1)
            sns.kdeplot(data=data, x=col[i])  
        plt.tight_layout()
        plt.show()

    if type == 'boxplot':  # The data shows varying levels of dispersion across features
        plt.figure(figsize=(15, 15))  
        for i in  range(len(col)):  
            plt.subplot(4,4,i+1)
            sns.boxplot(data,x=col[i])
            plt.xlabel(col[i])
        plt.show()

    if type == 'heatmap':  # represents a correlation matrix between various variables
        plt.figure(figsize=(10, 6))  
        sns.heatmap(data[col].corr(), annot=True, annot_kws={'size':6}, fmt='.2f')  
        plt.show()

    if type == 'scatterplot':  # Used to compare pairs of features (u, g, r, i, z)
        sns.pairplot(data=data[['u', 'g', 'r', 'i', 'z','class']],hue='class')
        plt.show()

    if type == 'classes':  # Bar Plot of Class Distribution
        data['class'].value_counts().plot(kind='bar')  
        plt.show()

# ---------------- Plot Models Resualts Functions ---------------

def plot_svm(x_train, y_train, svm_model, feature_names):
    result = permutation_importance(svm_model, x_train, y_train, n_repeats=10, random_state=42)
    importance = result.importances_mean
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for SVM')
    plt.show()

def plot_knn(x_train, y_train, knn_model, feature_names):
    result = permutation_importance(knn_model, x_train, y_train, n_repeats=10, random_state=42)
    importance = result.importances_mean
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for KNN')
    plt.show()

def plot_lr(x_train, y_train, lr_model, feature_names):
    result = permutation_importance(lr_model, x_train, y_train, n_repeats=10, random_state=42)
    importance = result.importances_mean
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Logistic Regression')
    plt.show()

def plot_dt(x_train, y_train, dt_model, feature_names):
    result = permutation_importance(dt_model, x_train, y_train, n_repeats=10, random_state=42)
    importance = dt_model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Decision Tree')
    plt.show()


def plot_rf(x_train, y_train, rf_model, feature_names):
    result = permutation_importance(rf_model, x_train, y_train, n_repeats=10, random_state=42)
    importance = rf_model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Random Forest')
    plt.show()