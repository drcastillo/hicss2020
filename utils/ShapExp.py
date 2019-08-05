
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from collections import Counter

import pickle
import sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

import seaborn as sns
from IPython.display import display, HTML
from sklearn.metrics import classification_report
from utils.perturbation import load_models_lendingclub
from IPython.display import display_html, display, HTML
import keras


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
                Multilayer Perceptron = keras_ann
                Sklearn Neural Network = sk_ann
            observation: int (Range: 0:4000)

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
            Multilayer Perceptron = keras_ann
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
                Multilayer Perceptron = keras_ann
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




def display_abs_shapvalues(features, num_features):
    try:
        combined_shap = pd.read_csv('obj/lendingclub/shap/All_Abs_Sum_ShapValues.csv', index_col = 0)
        models_str = ['Random Forest', 'Logistic Regression','Gradient Boosted Classifier', 'Multilayer Perceptron', 'Sklearn Neural Network']
        for i,j in zip(combined_shap.columns, models_str):
            temp = combined_shap.nlargest(num_features, i)
            sns.barplot(temp[i], temp.index)
            plt.title('{} - Absolute Shap Values TOP {}'.format(j, num_features))
            plt.savefig('images/shap_summed_values/{} - Absolute Shap Values TOP {}'.format(j, num_features), eps = 1000, bbox_inches = "tight")
            plt.show()
    except:
        print("Could not find csv containing shap values. This function references csv files @ obj/lendingclub/shap/All_Abs_Sum_ShapValues.csv")

def display_shapvalues(features, n):
    try:
        combined_shap = pd.read_csv('obj/lendingclub/shap/All_Sum_ShapValues.csv', index_col = 0)
        models_str = ['Random Forest', 'Logistic Regression','Gradient Boosted Classifier', 'Multilayer Perceptron', 'Sklearn Neural Network']
        for i,j in zip(combined_shap.columns, models_str):
            temp = combined_shap.nlargest(int(n / 2), i)
            temp1 = combined_shap.nsmallest(int(n / 2), i)
            temp = pd.concat([temp, temp1])
            sns.barplot(temp[i], temp.index)
            plt.title('{} Shap Values - Top&Bottom {}'.format(j, int(n / 2)))
            plt.savefig('images/shap_summed_values/{} - Summed Shap Values Top&Bottom {}'.format(j, int(n / 2)), eps = 1000, bbox_inches = "tight")
            plt.show()
    except:
        print("Could not find csv containing shap values. This function references csv files @ obj/lendingclub/shap/All_Abs_Sum_ShapValues.csv")

def get_shap_values(model):
    if type(model) == keras.engine.training.Model:
        f = lambda x: model.predict(x)[:, 1]
    else:
        f = lambda x: model.predict_proba(x)[:, 1]
    med = X_train_shap.median().values.reshape((1, X_train_shap.shape[1]))
    explainer = shap.KernelExplainer(f, med)
    shap_values = explainer.shap_values(X_test_shap, samples =500)
    return shap_values
