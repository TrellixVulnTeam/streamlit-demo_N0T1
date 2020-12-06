import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

st.title("Classifier Comparison")

st.write("""
### Which one is best?
Use the side bar to explore different classifiers and change their arrirubutes.
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))


classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of the dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = {}
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"], random_state=42)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# Plot
pca = PCA(2)
X_projected =pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig, ax = plt.subplots()
ax.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

st.pyplot(fig)