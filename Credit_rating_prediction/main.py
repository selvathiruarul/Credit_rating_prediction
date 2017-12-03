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


def data_exploration():

    print(df.columns.values)
    print(df.describe())
    df['class'].hist()
    plt.savefig(BASE_DIR+'/resources/class_exploration.png')
    df.hist()
    plt.savefig(BASE_DIR+'/resources/data_exploration.png')



def read_data(file_name):
    df.dropna()


    le = preprocessing.LabelEncoder()
    data = df
    headers = ['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-support', 'AUC']


    for col in df.columns.values:
        le.fit(df[col])
        data[col] = le.transform(df[col])
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['class'], axis=1), data['class'], random_state=2,
                                                        train_size=0.7, test_size=0.3)

    # Logistic Regression
    classifier_logit = LogisticRegression(random_state=2)
    classifier_logit.fit(X_train, y_train)
    y_pred = classifier_logit.predict(X_test)
    result = []
    result.append(['Logistic Regression', classifier_logit.score(X_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Linear SVC
    classifier_svc = LinearSVC()
    classifier_svc.fit(X_train, y_train)
    y_pred = classifier_svc.predict(X_test)
    result.append(['Linear SVC', classifier_svc.score(X_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Decision Tree
    classifier_decision = DecisionTreeClassifier()
    classifier_decision.fit(X_train, y_train)
    y_pred = classifier_decision.predict(X_test)
    result.append(
        ['Decision Tree classifier', classifier_decision.score(X_test, y_test), precision_score(y_test, y_pred),
         recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Random Forest
    classifier_rf = RandomForestClassifier()
    classifier_rf.fit(X_train, y_train)
    y_pred = classifier_rf.predict(X_test)
    result.append(['Random Forest', classifier_rf.score(X_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])

    # Multilayer
    classifier_mlp = MLPClassifier()
    classifier_mlp.fit(X_train, y_train)
    result.append(['Multilayer Perceptron', classifier_mlp.score(X_test, y_test), precision_score(y_test, y_pred),
                   recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)])
    print(tabulate(result, headers))

    n_features = df.shape[1]
    plt.barh(range(n_features - 1), classifier_rf.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()


if __name__ == '__main__':
    data_exploration()
    # read_data(os.path.join(BASE_DIR, FILE_NAME))
