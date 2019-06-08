#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Reference:

from __future__ import print_function
from utils.heaton_utils import *

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
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import seaborn as sns
from utils.load_objects import *


def load_models_lendingclub():
    from sklearn.externals import joblib
    from keras.models import load_model
    rf_file = "models/LendingClub/random_forest.pkl"
    gbc_file = "models/LendingClub/GBC.pkl"
    logit_file = "models/LendingClub/Logit.pkl"
    sklearn_nn_file = "models/LendingClub/SklearnNeuralNet.pkl"
    keras_ann_file = "models/LendingClub/ann_deepexplain.h5"

    rfc = joblib.load(rf_file)
    gbc = joblib.load(gbc_file)
    logit = joblib.load(logit_file)
    keras_ann = load_model(keras_ann_file)
    sk_ann = joblib.load(sklearn_nn_file)
    return rfc, gbc, logit, keras_ann, sk_ann

def load_models_uci():
    from sklearn.externals import joblib
    from keras.models import load_model
    sklearn_nn_file = 'models/UCI_Census/SklearnNeuralNet.pkl'
    keras_ann_file = 'models/UCI_Census/ann_deepexplain.h5'
    rf_file = "models/UCI_Census/random_forest.pkl"
    gbc_file = "models/UCI_Census/GBC.pkl"
    logit_file = "models/UCI_Census/Logit.pkl"

    rfc = joblib.load(rf_file)
    gbc = joblib.load(gbc_file)
    logit = joblib.load(logit_file)
    keras_ann = load_model(keras_ann_file)
    sk_ann = joblib.load(sklearn_nn_file)
    return rfc, gbc, logit, keras_ann, sk_ann

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
    plt.savefig('images/sklearn_feature_importance/RandomForest_{}features.png'.format(n_features), eps = 500)
    plt.show()

    temp = feature_importance.nlargest(n, 'GBC_Feature_Importance')
    sns.barplot(temp['GBC_Feature_Importance'], temp['feature'])
    plt.title('Gradient Boosted Classifier - Feature Importance Top {}'.format(
        n_features))
    plt.savefig('images/sklearn_feature_importance/GradientBoosting_{}features.png'.format(n_features), eps = 500)
    plt.show()


    #We want to show the total possible feature impact here. Take the max of each feature in the training set by the logit coeff.
    lookup = pd.DataFrame(data.toarray(), columns=features).max()
    temp = feature_importance.nlargest(int(n / 2), 'coeff_max')
    temp1 = feature_importance.nsmallest(int(n / 2), 'coeff_max')
    temp = pd.concat([temp, temp1])
    sns.barplot(temp['coeff_max'], temp['feature'])
    plt.title('Logistic Regression - Coefficients Top&Bottom {}'.format(
        int(n_features / 2)))
    plt.savefig('images/sklearn_feature_importance/Logit_{}features.png'.format(n_features), eps = 500)
    plt.show()


def get_shap_values(model):
    if type(model) == keras.engine.training.Model:
        f = lambda x: model.predict(x)[:, 1]
    else:
        f = lambda x: model.predict_proba(x)[:, 1]
    med = X_train_shap.median().values.reshape((1, X_train_shap.shape[1]))
    explainer = shap.KernelExplainer(f, med)
    shap_values = explainer.shap_values(X_test_shap, samples =500)
    return shap_values




class ExplainShap():

    def __init__(self, train, test, model_dict, feature_names, class_names = ['Bad Loan', 'Good Loan']):
        '''
        Parameters:
            shap_values: dict;
                locally generated shap values. Stored as local variable = shap_values
            train: df;
                X_train_shap is the df that needs to be passed in here.
            test: df;
                X_test_shap is the df that needs to be passed in here.
            model_dict: dict;
                pass in dict == models. This stores model names and shap Values
            feature_names: list;
                This stores a list of all features in the necessary order (Unravels the OHE during preprocessing)
            class_names: list;
                list of both classes

        Example:
            plot = ExplainShap(X_train_shap, X_test_shap, models, features, class_names = ['Bad Loan', 'Good Loan'])
            plot.shap_local_graph(model=keras_ann, observation=1)
        '''
        self.train = train
        self.test = test
        self.model_dict = model_dict
        self.feature_names = feature_names
        self.class_names = class_names

    def shap_local_graph(self, model, observation):
        '''
        Parameters:
            model: object,
                Random Forest: rfc
                Gradient Boosted Classifier: gbc
                Logistic Regression: logit
                Keras Neural Network = keras_ann
                Sklearn Neural Network = sk_ann
            observation: int

        Returns:
            Local Shap Explanation

        '''
        import shap
        #Keras doesn't have a model.predict_proba function. It outputs the probabilities via predict method
        if type(model) ==keras.engine.sequential.Sequential:
            f = lambda x: model.predict(x)[:, 1]
        else:
            f = lambda x: model.predict_proba(x)[:, 1]
        #We use the median as a proxy for computational efficiency. Rather than getting the expected value over the whole
        #training distribution, we get E(f(x)) over the median of the training set, e.g., model.predict(median(xi))
        med = self.train.median().values.reshape((1, self.train.shape[1]))
        explainer = shap.KernelExplainer(f, med)
        print("{} Shap Values".format(self.model_dict[str(type(model))][0]))

        return shap.force_plot(
            explainer.
            expected_value,  #Expected value is the base value - E(mean(f(x)))
            self.model_dict[str(type(model))][1][observation],
            feature_names=self.feature_names)


    #plot_cmap=['#808080', '#0000FF']
    def shap_many_graph(self, model):
        '''
        Parameters:
        model: object,
            Random Forest: rfc
            Gradient Boosted Classifier: gbc
            Logistic Regression: logit
            Keras Neural Network = keras_ann
            Sklearn Neural Network = sk_ann
        Returns:
        Global Shap Explanations over test set

        '''
        import shap
        #Keras doesn't have a model.predict_proba function. It outputs the probabilities via predict method
        if type(model) == keras.engine.sequential.Sequential:
            f = lambda x: model.predict(x)[:, 1]
        else:
            f = lambda x: model.predict_proba(x)[:, 1]
        med = self.train.median().values.reshape((1, self.train.shape[1]))
        explainer = shap.KernelExplainer(f, med)
        return shap.force_plot(explainer.expected_value, self.model_dict[str(type(model))][1],
                               self.test)

    def shap_summary_graph(self, model):
        '''
        Parameters:
            model: object,
                Random Forest: rfc
                Gradient Boosted Classifier: gbc
                Logistic Regression: logit
                Keras Neural Network = keras_ann
                Sklearn Neural Network = sk_ann
        Returns:
        Global Shap Explanations over test set - Summary
        '''
        import shap
        if type(model) == keras.engine.sequential.Sequential:
            f = lambda x: model.predict(x)[:, 1]
        else:
            f = lambda x: model.predict_proba(x)[:, 1]
        med = self.train.median().values.reshape((1, self.train.shape[1]))
        explainer = shap.KernelExplainer(f, med)
        print("{} Shap Values".format(self.model_dict[str(type(model))][0]))
        return shap.summary_plot(self.model_dict[str(type(model))][1],
                                 self.test,
                                 class_names=self.class_names,
                                 plot_type="dot")


class Perturb():
    def __init__(self, X, y, data_str):
        '''
        Parameters:
            X: df;
                pass in X_test_holdout dataframe
            y: df;
                pass in y_test_holdout dataframe
            data_str: str;
                'uci' if using census data, 'lending' if using lending club data
        Example:
            p = Perturb(X_test_holdout, y_test_holdout, data_str= 'uci')
            p.manual_perturb(column='age',scalar=1.1)
        '''
        self.X = X
        self.y = y
        self.data = data_str

        self.a = [str(i) + '%' for i in range(0, 201) if i % 5 == 0]
        self.b = [(i / 100) for i in range(0, 201) if i % 5 == 0]
        self.pertu = dict(zip(self.a, self.b))
        self.pert = [i for i in self.pertu.values()]

        from utils.load_objects import load_models_lendingclub, load_models_uci
        if 'uci' in self.data:
            self.rfc, self.gbc, self.logit, self.keras_ann, self.sk_ann = load_models_uci()
        elif 'lending' in self.data:
            self.rfc, self.gbc, self.logit, self.keras_ann, self.sk_ann = load_models_lendingclub()

    def perturb_graph(self, model, mode, column):
        '''
        Parameters:
            model: object,
                Random Forest: rfc
                Gradient Boosted Classifier: gbc
                Logistic Regression: logit
            column: pass dataframe column, e.g, age, fnlwgt etc. Can pass a list of columns, e.g., ['age', 'fnlwgt']
            title: str; pass title of graph

        Returns:
            Forecasting Graph based on perturbed features.

        '''
        import collections
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        clone = self.X.copy()
        column = column


        preds = []
        mean = []
        num_of_1s = []
        for i in self.pert:
            clone[column] = self.X[column] * (i)
            mean.append(str(column) + ':' + str(np.round(clone[column].mean())))

            preds.append(
                sklearn.metrics.accuracy_score(self.y, model.predict(clone))* 100)

            num_of_1s.append((collections.Counter(model.predict(clone))[1] / self.y.shape[0]) *100)

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))
        if 'accuracy' in mode:
            sns.lineplot(x=self.pert, y=preds, ax=ax1)
            ax1.set_ylabel('Accuracy %', fontsize=15)
            ax1.set_title('Accuracy :{}'.format(
            column.upper()),
                      fontsize=25)
            ax1.set_ylim(0,100)
        elif 'proportion' in mode:
            sns.lineplot(x=self.pert, y=num_of_1s, ax=ax1)
            ax1.set_ylabel('% of Predictions == 1', fontsize=15)
            ax1.set_title('Proportionality of Predictions :{}'.format(
            column.upper()),
                      fontsize=25)
            ax1.set_ylim(0,100)

        ax1.set_xlabel('{} Perturbation'.format(column.upper()), fontsize=15)
        plt.legend(title='Legend',
                   loc='lower right',
                   labels=[
                       'Perturbed - Probability'

                   ])

        if isinstance(column, str):
            for i, txt in enumerate(mean):
                if i % 5 == 0:
                    if 'accuracy' in mode:
                        ax1.annotate(txt, (self.pert[i], preds[i]))
                    if 'proportion' in mode:
                        ax1.annotate(txt, (self.pert[i], num_of_1s[i]))

    def perturb_graph_cons(self, mode, column):

        '''
        Parameters:
            mode: str;
                'accuracy' : Y Axis = Percent of Correct Predictions
                'proportion' : Percentage of Class 1 Predictions / Total length of Y_test
            column: pass dataframe column, e.g, age, fnlwgt etc. Can pass a list of columns, e.g., ['age', 'fnlwgt']
            title: str; pass title of graph

        Returns:
            Perturbed Input Graph. Shows all models simultaneously, as opposed to the above

        '''
        import collections
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        clone = self.X.copy()
        column = column


        rfc_preds = []
        gbc_preds = []
        logit_preds = []

        rfc_1_preds =[]
        gbc_1_preds =[]
        logit_1_preds =[]

        for i in self.pert:
            clone[column] = self.X[column] * (i)

            rfc_preds.append(sklearn.metrics.accuracy_score(self.y, self.rfc.predict(clone))*100)
            gbc_preds.append(sklearn.metrics.accuracy_score(self.y, self.gbc.predict(clone))*100)
            logit_preds.append(sklearn.metrics.accuracy_score(self.y, self.logit.predict(clone))*100)

            rfc_1_preds.append((collections.Counter(self.rfc.predict(clone))[1] / self.y.shape[0]) *100)
            gbc_1_preds.append((collections.Counter(self.gbc.predict(clone))[1] / self.y.shape[0]) *100)
            logit_1_preds.append((collections.Counter(self.logit.predict(clone))[1] / self.y.shape[0]) *100)

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))
        if 'accuracy' in mode:
            sns.lineplot(x=self.pert, y=rfc_preds, ax=ax1)
            sns.lineplot(x=self.pert, y=gbc_preds, ax=ax1)
            sns.lineplot(x=self.pert, y=logit_preds, ax=ax1)
            ax1.set_ylabel('Accuracy %', fontsize=15)
            ax1.set_title('Accuracy : {}'.format(
            column.upper()),
                      fontsize=25)
            ax1.set_ylim(0,100)

        elif 'proportion' in mode:
            sns.lineplot(x=self.pert, y=rfc_1_preds, ax=ax1)
            sns.lineplot(x=self.pert, y=gbc_1_preds, ax=ax1)
            sns.lineplot(x=self.pert, y=logit_1_preds, ax=ax1)
            ax1.set_ylabel('% of Predictions == 1', fontsize=15)
            ax1.set_title('Proportionality of Predictions :{}'.format(
            column.upper()),
                      fontsize=25)
            ax1.set_ylim(0,100)


        ax1.set_xlabel('{} Perturbation'.format(column), fontsize=15)
        plt.legend(title='Model',
                   loc='upper left',
                   labels=[
                       'Random Forest', 'Gradient Boosted Classifier',
                       'Logistic Regression'
                   ])

    def manual_perturb(self, column, scalar):
        '''
        Parameters
            X: X test DataFrame
            y: y test Dataframe
            set: str;
                'lendingclub' - load lending club models
                'uci' - load uci models
            column: str; feature of interest
            scalar: float; multiplier
        Returns:
            To String
        '''
        import collections
        import sklearn
        temp = self.X.copy()
        temp[column] = temp[column] * scalar
        print("Perturbing Feature: {} by {}".format(column,scalar))
        print('-' * 75)
        print("\033[1m Random Forest \033[0m")
        bef_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.rfc.predict(self.X))*100,4)
        print("\tBefore Perturbation, Accuracy: {}%".format(bef_acc))
        aft_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.rfc.predict(temp))*100,4)
        print("\tAfter Perturbation, Accuracy: {}%".format(aft_acc))
        print("\tNumber of '1' Predictions, Before Perturbation: {}".format(collections.Counter(self.rfc.predict(self.X))[1]))
        print("\tNumber of '1' Predictions, After Perturbation: {}".format(collections.Counter(self.rfc.predict(temp))[1]))


        print("\n\033[1m Gradient Boosted Classifier\033[0m")
        bef_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.gbc.predict(self.X))*100,4)
        print("\tBefore Perturbation, Accuracy: {}%".format(bef_acc))
        aft_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.gbc.predict(temp))*100,4)
        print("\tAfter Perturbation, Accuracy: {}%".format(aft_acc))
        print("\tNumber of '1' Predictions, Before Perturbation: {}".format(collections.Counter(self.gbc.predict(self.X))[1]))
        print("\tNumber of '1' Predictions, After Perturbation: {}".format(collections.Counter(self.gbc.predict(temp))[1]))

        print("\n\033[1m Logistic Regression\033[0m")
        bef_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.logit.predict(self.X))*100,4)
        print("\tBefore Perturbation, Accuracy: {}%".format(bef_acc))
        aft_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.logit.predict(temp))*100,4)
        print("\tAfter Perturbation, Accuracy: {}%".format(aft_acc))
        print("\tNumber of '1' Predictions, Before Perturbation: {}".format(collections.Counter(self.logit.predict(self.X))[1]))
        print("\tNumber of '1' Predictions, After Perturbation: {}".format(collections.Counter(self.logit.predict(temp))[1]))

        print("\n\033[1m Neural Net\033[0m")
        bef_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.keras_ann.predict_classes(self.X))*100,4)
        print("\tBefore Perturbation, Accuracy: {}%".format(bef_acc))
        aft_acc = np.round(sklearn.metrics.accuracy_score(self.y, self.keras_ann.predict_classes(temp))*100,4)
        print("\tAfter Perturbation, Accuracy: {}%".format(aft_acc))
        print("\tNumber of '1' Predictions, Before Perturbation: {}".format(collections.Counter(self.keras_ann.predict_classes(self.X))[1]))
        print("\tNumber of '1' Predictions, After Perturbation: {}\n".format(collections.Counter(self.keras_ann.predict_classes(temp))[1]))
        print('-' * 75)
def display_abs_shapvalues(shapvalues, features, num_features):
    rfc, gbc, logit, keras_ann, sk_ann = load_models_lendingclub()
    rfc_shapvalues_abs = pd.DataFrame(shapvalues[str(type(rfc))], columns = features).abs().sum()
    logit_shapvalues_abs = pd.DataFrame(shapvalues[str(type(logit))], columns = features).abs().sum()
    gbc_shapvalues_abs = pd.DataFrame(shapvalues[str(type(gbc))], columns = features).abs().sum()
    keras_ann_shapvalues_abs = pd.DataFrame(shapvalues[str(type(keras_ann))], columns = features).abs().sum()


    combined_shap = pd.DataFrame(rfc_shapvalues_abs, columns= ['rfc'])
    combined_shap['logit'] = logit_shapvalues_abs
    combined_shap['gbc'] = gbc_shapvalues_abs
    combined_shap['nn'] = keras_ann_shapvalues_abs

    temp = combined_shap.nlargest(num_features, 'rfc')
    sns.barplot(temp['rfc'], temp.index)
    plt.title('Random Forest - Absolute Shap Values TOP {}'.format(num_features))
    plt.show()

    temp = combined_shap.nlargest(num_features, 'logit')
    sns.barplot(temp['logit'], temp.index)
    plt.title('Logistic Regression - Absolute Shap Values TOP {}'.format(num_features))
    plt.show()

    temp = combined_shap.nlargest(num_features, 'gbc')
    sns.barplot(temp['gbc'], temp.index)
    plt.title('Gradient Boosted Classifier - Absolute Shap Values TOP {}'.format(num_features))
    plt.show()

    temp = combined_shap.nlargest(num_features, 'nn')
    sns.barplot(temp['nn'], temp.index)
    plt.title('Neural Network - Absolute Shap Values TOP {}'.format(num_features))
    plt.show()

def display_shapvalues(shapvalues, features, n):
    rfc, gbc, logit, keras_ann, sk_ann = load_models_lendingclub()
    rfc_shapvalues = pd.DataFrame(shapvalues[str(type(rfc))], columns = features).sum()
    logit_shapvalues = pd.DataFrame(shapvalues[str(type(logit))], columns = features).sum()
    gbc_shapvalues = pd.DataFrame(shapvalues[str(type(gbc))], columns = features).sum()
    keras_ann_shapvalues = pd.DataFrame(shapvalues[str(type(keras_ann))], columns = features).sum()

    combined_shap = pd.DataFrame(rfc_shapvalues, columns= ['rfc'])
    combined_shap['logit'] = logit_shapvalues
    combined_shap['gbc'] = gbc_shapvalues
    combined_shap['nn'] = keras_ann_shapvalues

    temp = combined_shap.nlargest(int(n / 2), 'rfc')
    temp1 = combined_shap.nsmallest(int(n / 2), 'rfc')
    temp = pd.concat([temp, temp1])
    sns.barplot(temp['rfc'], temp.index)
    plt.title('Random Forest Shap Values - Top&Bottom {}'.format(int(n / 2)))
    plt.show()

    temp = combined_shap.nlargest(int(n / 2), 'logit')
    temp1 = combined_shap.nsmallest(int(n / 2), 'logit')
    temp = pd.concat([temp, temp1])
    sns.barplot(temp['logit'], temp.index)
    plt.title('Logistic Regression Shap Values - Top&Bottom {}'.format(int(n / 2)))
    plt.show()

    temp = combined_shap.nlargest(int(n / 2), 'gbc')
    temp1 = combined_shap.nsmallest(int(n / 2), 'gbc')
    temp = pd.concat([temp, temp1])
    sns.barplot(temp['gbc'], temp.index)
    plt.title('Gradient Boosting Shap Values - Top&Bottom {}'.format(int(n / 2)))
    plt.show()

    temp = combined_shap.nlargest(int(n / 2), 'nn')
    temp1 = combined_shap.nsmallest(int(n / 2), 'nn')
    temp = pd.concat([temp, temp1])
    sns.barplot(temp['nn'], temp.index)
    plt.title('Neural Network Shap Values - Top&Bottom {}'.format(int(n / 2)))
    plt.show()
