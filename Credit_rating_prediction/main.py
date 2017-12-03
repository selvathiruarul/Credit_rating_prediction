from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

BASE_DIR = os.path.dirname(__file__)
FILE_NAME = 'resources/creditData.csv'
df = pd.read_csv(FILE_NAME)
headers = ['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-support', 'AUC']

def data_exploration():

    print(df.columns.values)
    print(df.describe())
    df['class'].hist()
    plt.savefig(BASE_DIR+'/resources/class_exploration.png')
    plt.clf()
    df.hist()
    plt.savefig(BASE_DIR+'/resources/data_exploration.png')
    plt.clf()


def data_preprocessing():

    df.dropna()
    le = preprocessing.LabelEncoder()
    data = df
    for col in df.columns.values:
        le.fit(df[col])
        data[col] = le.transform(df[col])
    data.hist()
    plt.savefig(BASE_DIR+'/resources/data_processed.png')
    plt.clf()
    return data


def predict_data(data):

    x_train, x_test, y_train, y_test = train_test_split(data.drop(['class'], axis=1), data['class'], random_state=2,
                                                        train_size=0.7, test_size=0.3)

    # Logistic Regression
    classifier_logit = LogisticRegression(random_state=2)
    classifier_logit.fit(x_train, y_train)
    y_pred = classifier_logit.predict(x_test)
    result = []
    result.append(['Logistic Regression', classifier_logit.score(x_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Linear SVC
    classifier_svc = LinearSVC()
    classifier_svc.fit(x_train, y_train)
    y_pred = classifier_svc.predict(x_test)
    result.append(['Linear SVC', classifier_svc.score(x_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Decision Tree
    classifier_decision = DecisionTreeClassifier()
    classifier_decision.fit(x_train, y_train)
    y_pred = classifier_decision.predict(x_test)
    result.append(
        ['Decision Tree classifier', classifier_decision.score(x_test, y_test), precision_score(y_test, y_pred),
         recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Random Forest
    classifier_rf = RandomForestClassifier()
    classifier_rf.fit(x_train, y_train)
    y_pred = classifier_rf.predict(x_test)
    result.append(['Random Forest', classifier_rf.score(x_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Multilayer
    classifier_mlp = MLPClassifier()
    classifier_mlp.fit(x_train, y_train)
    result.append(['Multilayer Perceptron', classifier_mlp.score(x_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])
    print(tabulate(result, headers))

    n_features = df.shape[1]
    plt.barh(range(n_features - 1), classifier_rf.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.savefig(BASE_DIR+'/resources/top_features.png')
    plt.clf()

if __name__ == '__main__':
    data_exploration()
    data_processed = data_preprocessing()
    predict_data(data_processed)
