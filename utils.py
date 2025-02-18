from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def evaluate_model(y, y_pred, class_names):
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred, target_names=class_names))
    print("\nConfusion Matrix\n", confusion_matrix(y, y_pred))

def prepare_data(data):
    
    classes = ['STAR', 'GALAXY', 'QSO']

    data.dropna(inplace=True)  
    data['class'] = data['class'].map({'STAR':0 ,'GALAXY':1 ,'QSO':2})

    train_data, test_data = train_test_split(
    data, 
    test_size=0.2, 
    random_state=42, 
    stratify=data['class']  
    )
    
    y_train = train_data['class'].values
    y_test = test_data['class'].values
    
    train_data = train_data.drop(columns=['class'])
    test_data = test_data.drop(columns=['class'])

    train_data = train_data.values
    test_data = test_data.values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_data)
    x_test = scaler.fit_transform(test_data)

    feature_names = [col for col in data.drop(columns=['class']).columns]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, x_val, y_val, classes, feature_names

def apply_pca(x_train, x_val, x_test, dim=2):

    pca = PCA(n_components=dim)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val) 
    x_test_pca = pca.transform(x_test) 

    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    print("Final Variance Ratio:", sum(pca.explained_variance_ratio_))

    return x_train_pca, x_val_pca, x_test_pca





