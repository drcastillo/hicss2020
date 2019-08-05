#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Reference:

#from __future__ import print_function
#from utils.heaton_utils import *

import numpy as np
import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob
#pip install counter
from collections import Counter

import pickle
import sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

import seaborn as sns
from IPython.display import display, HTML
from sklearn.metrics import classification_report
from utils.perturbation import load_models_lendingclub
from IPython.display import display_html, display, HTML
import lime.lime_tabular
import lime

class KerasModelUtil:

    modelwts_extension = "h5"
    json_extension = "json"
    pickle_extension = "p"

    def save(self, model_dir, model_name, model, label_class_map):
        if model_dir.endswith('/') == False:
            model_dir = model_dir + '/'

        # put the file name into specific tokens
        fn_base, sep, tail = model_name.partition('.')
        if not sep:
            sep = "."

        json_fn = model_dir + fn_base + sep + self.json_extension

        wt_ext = tail
        if not wt_ext:
            wt_ext = self.modelwts_extension
        wt_fn = model_dir + fn_base + sep + wt_ext

        pickle_fn = model_dir + fn_base + sep + self.pickle_extension

        pickle.dump(label_class_map, open(pickle_fn, 'wb'))

        # serialize model to JSON
        model_json = model.to_json()

        with open(json_fn, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(wt_fn)

    def load(self, model_dir, model_name, input_shape=(None, 224, 224, 3)):
        # Load the json model first
        if model_dir.endswith('/') == False:
            model_dir = model_dir + '/'

        # put the file name into specific tokens
        fn_base, sep, tail = model_name.partition('.')
        if not sep:
            sep = "."

        json_fn = model_dir + fn_base + sep + self.json_extension
        json_file = open(json_fn, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # form the model from the json and rebuild the layers
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.build(input_shape=input_shape)

        # Load the weights
        wt_ext = tail
        if not wt_ext:
            wt_ext = self.modelwts_extension
        wt_fn = model_dir + fn_base + sep + wt_ext
        loaded_model.load_weights(wt_fn)

        #print("Loaded model from disk")

        # Load the labels and Class ids
        pickle_fn = model_dir + fn_base + sep + self.pickle_extension
        label_classids = pickle.load(open(pickle_fn, "rb"))
        class_label_map = {v: k for k, v in label_classids.items()}
        #print(label_classids)
        #print(classids_labels)

        return loaded_model, class_label_map


##################################################
# Keras callbacks for plotting training model
# accuracy and loss
##################################################
from IPython.display import clear_output
import math
import keras


#Can just import LiveLossPlot & add to model callbacks.
class TrainingPlot(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=False)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="training loss")
        ax1.plot(self.x, self.val_losses, label="validation loss")
        ax1.legend()

        ax2.set_ylim(0, 1.0)
        ax2.plot(self.x, self.acc, label="training accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show()


##################################################
# Utility code for computing a Confusion Matrix
##################################################

import matplotlib.pyplot as plt  #for plotting
import itertools as it


#Note, this code is taken straight from the SKLEARN website, a nice way of viewing confusion matrix.
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Note: class is a listlike parameter. Pass in list of classes, eg: ["No Loan", "Loan"]
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        value = '{0:.2g}'.format(cm[i, j])
        plt.text(j,
                 i,
                 value,
                 fontsize=10,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



##################################################
# Utility code for measuring model performance given dataset size
##################################################
def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylim=None,
                        cv=None,
                        n_jobs=-1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")
    plt.plot(train_sizes,
             train_scores_mean,
             'o-',
             color="r",
             label="Training score")
    plt.plot(train_sizes,
             test_scores_mean,
             'o-',
             color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def display_sklearn_feature_importance(data, set, features, n_features):
    '''
    Parameters:
    data: data object; coomatrix w/ encoded features
    n_features: number of features to visualize
    set: str;
        'lendingclub' - load lending club models
        'uci' - load uci models
    Returns:
    Graph of basic feature importance measurements

    '''
    if 'uci' in set:
        rfc, gbc, logit, keras_ann, sk_ann = load_models_uci()
    else:
        rfc, gbc, logit, keras_ann, sk_ann = load_models_lendingclub()
    feature_importance = pd.DataFrame({
        "feature":
        features,
        "RF_Feature_Importance":
        np.round(rfc.feature_importances_, 4),
        "GBC_Feature_Importance":
        np.round(gbc.feature_importances_, 4),
        "Logit_Coeff":
        np.round(logit.coef_[0], 4),
        "Max_Feature_Val":
        pd.DataFrame(data.toarray(), columns=features).max(),
    })

    n = n_features
    feature_importance['coeff_max'] = feature_importance[
        'Logit_Coeff'] * feature_importance['Max_Feature_Val']
    temp = feature_importance.nlargest(n, 'RF_Feature_Importance')
    sns.barplot(temp['RF_Feature_Importance'], temp['feature'])
    plt.title('Random Forest - Feature Importance Top {}'.format(n_features))
    plt.show()

    temp = feature_importance.nlargest(n, 'GBC_Feature_Importance')
    sns.barplot(temp['GBC_Feature_Importance'], temp['feature'])
    plt.title('Gradient Boosted Classifier - Feature Importance Top {}'.format(
        n_features))
    plt.show()

    #We want to show the total possible feature impact here. Take the max of each feature in the training set by the logit coeff.
    lookup = pd.DataFrame(data.toarray(), columns=features).max()
    temp = feature_importance.nlargest(int(n / 2), 'coeff_max')
    temp1 = feature_importance.nsmallest(int(n / 2), 'coeff_max')
    temp = pd.concat([temp, temp1])
    sns.barplot(temp['coeff_max'], temp['feature'])
    plt.title('Logistic Regression - Coefficients Top&Bottom {}'.format(
        int(n_features / 2)))
    plt.show()


def get_best_score(x, y):
    try:
        return sklearn.metrics.accuracy_score(x, y.predict(encoded_test))
    except:
        return sklearn.metrics.accuracy_score(x, keras_ann.predict_classes(encoded_test.toarray()))


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'),
                 raw=True)

def neg_pos_logit_coefficients(model, features):
    logistic_regress_coeff = pd.DataFrame({
        "features": features,
        "Coef": model.coef_[0]
    })

    neg_coef = round(logistic_regress_coeff[
        logistic_regress_coeff['Coef'] < 0].sort_values('Coef', ascending=True),2).head(15)
    pos_coef = round(logistic_regress_coeff[
        logistic_regress_coeff['Coef'] > 0].sort_values('Coef', ascending=False),2).head(15)
    display_side_by_side(neg_coef, pos_coef)
