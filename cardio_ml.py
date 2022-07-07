import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def get_data(path):
    return pd.read_csv(path, sep=';')


def age_to_years(dataset):
    dataset['age'] = dataset['age'] / 365
    dataset['age'] = dataset['age'].astype(np.int64)


def gender_mod(dataset):
    dataset.loc[dataset['gender'] == 1, 'gender'] = 'female'
    dataset.loc[dataset['gender'] == 2, 'gender'] = 'male'


def data_selection(dataset):
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    return X, y


def gender_encoding(X):
    le = LabelEncoder()
    X[:, 1] = le.fit_transform(X[:, 1])


def model_splitting(X, y, test):
    return train_test_split(X, y, test_size=test, random_state=42, shuffle=True)


def scaler():
    return StandardScaler()


def scaling_fit(X_train, X_test):
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test


# Creating classification models
def gbc_model(est, X_train, y_train):
    return GradientBoostingClassifier(loss='log_loss', n_estimators=est).fit(X_train, y_train)


def log_reg(X_train, y_train):
    return LogisticRegression().fit(X_train, y_train)


def near_neighbours(X_train, y_train):
    return KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2, n_jobs=-1).fit(X_train, y_train)


def support_vm(X_train, y_train):
    svm_linear = SVC(kernel='linear', degree=5).fit(X_train, y_train)
    svm_rbf = SVC(kernel='rbf', degree=5).fit(X_train, y_train)
    return svm_linear, svm_rbf


def bayes(X_train, y_train):
    return GaussianNB().fit(X_train, y_train)


def dec_tree(X_train, y_train):
    return DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)


def radn_tree(X_train, y_train):
    return RandomForestClassifier(n_estimators=100, criterion='entropy').fit(X_train, y_train)


def xgbmodel(X_train, y_train):
    return XGBClassifier().fit(X_train, y_train)


def scores(X_test, y_test, gbc, lgr, knn, svm_linear, svm_rbf, nb, dt, rt, xgb):
    # Accuracy Score, Precision score, F1_Score, Recall Score
    gbc_acc = accuracy_score(y_test, gbc.predict(X_test))
    gbc_prec = precision_score(y_test, gbc.predict(X_test))
    gbc_f1 = f1_score(y_test, gbc.predict(X_test))
    gbc_rec = recall_score(y_test, gbc.predict(X_test))

    lgr_acc = accuracy_score(y_test, lgr.predict(X_test))
    lgr_prec = precision_score(y_test, lgr.predict(X_test))
    lgr_f1 = f1_score(y_test, lgr.predict(X_test))
    lgr_rec = recall_score(y_test, lgr.predict(X_test))

    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    knn_prec = precision_score(y_test, knn.predict(X_test))
    knn_f1 = f1_score(y_test, knn.predict(X_test))
    knn_rec = recall_score(y_test, knn.predict(X_test))

    svm_linear_acc = accuracy_score(y_test, svm_linear.predict(X_test))
    svm_linear_prec = precision_score(y_test, svm_linear.predict(X_test))
    svm_linear_f1 = f1_score(y_test, svm_linear.predict(X_test))
    svm_linear_rec = recall_score(y_test, svm_linear.predict(X_test))

    svm_rbf_acc = accuracy_score(y_test, svm_rbf.predict(X_test))
    svm_rbf_prec = precision_score(y_test, svm_rbf.predict(X_test))
    svm_rbf_f1 = f1_score(y_test, svm_rbf.predict(X_test))
    svm_rbf_rec = recall_score(y_test, svm_rbf.predict(X_test))

    nb_acc = accuracy_score(y_test, nb.predict(X_test))
    nb_prec = precision_score(y_test, nb.predict(X_test))
    nb_f1 = f1_score(y_test, nb.predict(X_test))
    nb_rec = recall_score(y_test, nb.predict(X_test))

    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    dt_prec = precision_score(y_test, dt.predict(X_test))
    dt_f1 = f1_score(y_test, dt.predict(X_test))
    dt_rec = recall_score(y_test, dt.predict(X_test))

    rt_acc = accuracy_score(y_test, rt.predict(X_test))
    rt_prec = precision_score(y_test, rt.predict(X_test))
    rt_f1 = f1_score(y_test, rt.predict(X_test))
    rt_rec = recall_score(y_test, rt.predict(X_test))

    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    xgb_prec = precision_score(y_test, xgb.predict(X_test))
    xgb_f1 = f1_score(y_test, xgb.predict(X_test))
    xgb_rec = recall_score(y_test, xgb.predict(X_test))

    return gbc_acc, gbc_prec, gbc_f1, gbc_rec, lgr_acc, lgr_prec, lgr_f1, lgr_rec, knn_acc, knn_prec, knn_f1, knn_rec, svm_linear_acc, svm_linear_prec, svm_linear_f1, svm_linear_rec, svm_rbf_acc, svm_rbf_prec, svm_rbf_f1, svm_rbf_rec, nb_acc, nb_prec, nb_f1, nb_rec, dt_acc, dt_prec, dt_f1, dt_rec, rt_acc, rt_prec, rt_f1, rt_rec, xgb_acc, xgb_prec, xgb_f1, xgb_rec


def scores_df(gbc_acc, gbc_prec, gbc_f1, gbc_rec, lgr_acc, lgr_prec, lgr_f1, lgr_rec, knn_acc, knn_prec, knn_f1,
              knn_rec, svm_linear_acc, svm_linear_prec, svm_linear_f1, svm_linear_rec, svm_rbf_acc, svm_rbf_prec,
              svm_rbf_f1, svm_rbf_rec, nb_acc, nb_prec, nb_f1, nb_rec, dt_acc, dt_prec, dt_f1, dt_rec, rt_acc, rt_prec,
              rt_f1, rt_rec, xgb_acc, xgb_prec, xgb_f1, xgb_rec):
    return pd.DataFrame(
        {'Models': ['Gradient Boosting Classifier', 'Logistic Regression', 'Nearest Neighbors', 'SVM Linear',
                    'SVM RBF', 'Gaussian Naive Bayes', 'Decision Tree Classifier', 'Random Forest Classifier',
                    'Extreme Gradient Boosting (XGB)'],
         'Accuracy Score': [gbc_acc, lgr_acc, knn_acc,
                            svm_linear_acc, svm_rbf_acc, nb_acc,
                            dt_acc, rt_acc, xgb_acc],
         'Precision Score': [gbc_prec, lgr_prec, knn_prec,
                             svm_linear_prec, svm_rbf_prec, nb_prec,
                             dt_prec, rt_prec, xgb_prec],
         'F1 Score': [gbc_f1, lgr_f1, knn_f1,
                      svm_linear_f1, svm_rbf_f1, nb_f1,
                      dt_f1, rt_f1, xgb_f1],
         'Recall Score': [gbc_rec, lgr_rec, knn_rec,
                          svm_linear_rec, svm_rbf_rec, nb_rec,
                          dt_rec, rt_rec, xgb_rec]})


if __name__ == '__main__':
    # Data operations
    dataset = get_data(path='cardio_train.csv')

    age_to_years(dataset)
    gender_mod(dataset)

    X, y = data_selection(dataset)

    gender_encoding(X)

    # Splitting model into train and test
    X_train, X_test, y_train, y_test = model_splitting(X, y, test=0.3)

    # Creating a scaler based on Standard Deviation
    sc = scaler()

    # Fitting the scaler on the X values
    X_train, X_test = scaling_fit(X_train, X_test)

    # Models
    gbc = gbc_model(est=50, X_train=X_train, y_train=y_train)

    lgr = log_reg(X_train, y_train)

    knn = near_neighbours(X_train, y_train)

    svm_linear, svm_rbf = support_vm(X_train, y_train)

    nb = bayes(X_train, y_train)

    dt = dec_tree(X_train, y_train)

    rt = radn_tree(X_train, y_train)

    xgb = xgbmodel(X_train, y_train)

    gbc_acc, gbc_prec, gbc_f1, gbc_rec, lgr_acc, lgr_prec, lgr_f1, lgr_rec, knn_acc, knn_prec, knn_f1, knn_rec, svm_linear_acc, svm_linear_prec, svm_linear_f1, svm_linear_rec, svm_rbf_acc, svm_rbf_prec, svm_rbf_f1, svm_rbf_rec, nb_acc, nb_prec, nb_f1, nb_rec, dt_acc, dt_prec, dt_f1, dt_rec, rt_acc, rt_prec, rt_f1, rt_rec, xgb_acc, xgb_prec, xgb_f1, xgb_rec = scores(
        X_test, y_test, gbc, lgr, knn, svm_linear, svm_rbf, nb, dt, rt, xgb)

    df = scores_df(gbc_acc, gbc_prec, gbc_f1, gbc_rec, lgr_acc, lgr_prec, lgr_f1, lgr_rec, knn_acc, knn_prec, knn_f1,
              knn_rec, svm_linear_acc, svm_linear_prec, svm_linear_f1, svm_linear_rec, svm_rbf_acc, svm_rbf_prec,
              svm_rbf_f1, svm_rbf_rec, nb_acc, nb_prec, nb_f1, nb_rec, dt_acc, dt_prec, dt_f1, dt_rec, rt_acc, rt_prec,
              rt_f1, rt_rec, xgb_acc, xgb_prec, xgb_f1, xgb_rec)


    df.to_excel('/home/gui/Desktop/estudos/machine_learning/cardio_project/scores.xlsx', sheet_name='Scores')