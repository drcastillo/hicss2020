#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Reference:


from __future__ import print_function
from heaton_utils import *

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



    def load(self,
             model_dir,
             model_name,
             input_shape=(None, 224, 224, 3)):
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

        plt.show();


##################################################
# Utility code for computing a Confusion Matrix
##################################################

import matplotlib.pyplot as plt #for plotting
import itertools as it

#Note, this code is taken straight from the SKLEARN website, a nice way of viewing confusion matrix.
def plot_confusion_matrix(cm, classes,
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
        plt.text(j, i, value,
                 fontsize=10,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####################################################################################################
# Utilities for Fetching data files from dir and performing various data cleaning
####################################################################################################
def list_dir(verbose = True):
    '''
    function to list the contents of current working directory
    Return:
    list of branches
    dict of current working directory {idx : branch}
    '''
    idx = []
    contents = []
    cwd = os.getcwd()
    tree = os.listdir(cwd)
    for i,j in enumerate(tree):
        idx.append(i)
        contents.append(j)
    if verbose:
        print("Working Dir: {}".format(cwd))
        print("Returning Contents of Working Directory..")
    return contents, dict(zip(idx, contents))
#helper
def fetch_data_path (folder = 'data'):
    '''
    function to string concat holistic path to data files w/ user input
    Parameters:
        folder: str - name of data folder
    Return:
        string of concatenated path to data file
    '''
    cwd = os.getcwd()
    path = cwd + "\\" + folder
    print("Choose a file from data directory:")
    for idx, pat in enumerate(os.listdir(path)):
        print("{}) {}".format(idx, pat))
    i = input("Enter Number: ")
    try:
        if 0 <= int(i) <= len(os.listdir(path)):
            dataPath = os.listdir(path)[int(i)]
            print("Path to Data Stored: {}". format(path + "\\" + dataPath))
            return path + "\\" + dataPath
    except:
        print("Invalid Selection")
    return None

def load_data_australian(path = 'data', drop_outliers = True, columns = 16, outlier_columns = None):
    '''

    PARAMETERS:
        path: string - folder where data resides
        drop_outliers: boolean - True if dropping outliers based on Tukey method
        columns: int - number of features in the dataframe. Used for column naming
        outlier_columns: list - list of columns to search for outliers

    RETURN:
        Dataframe Object

    EXAMPLE:
        df = load_data(path = 'data', drop_outliers = True, columns = 15)

    '''
    data_path = fetch_data_path(folder = path) #Using UDF to fetch data
    cols = [("A" + str(i)) for i in range(1,columns)] #Arbitrary Column Naming
    df = pd.read_csv(data_path, header = None, delimiter= " ", names= cols) #Create DF. Reading in a .dat file with tab delimeter

    if drop_outliers:
        Outliers_to_drop = detect_outliers(df,1,outlier_columns)
        df = df.drop(df.index[Outliers_to_drop]) #Remove outliers based on Tukey method
    return df

def load_data_UCI(path = 'data\\adult.csv', clean = True):
    '''
    Fetches and cleans UCI data from path
    '''
    data = pd.read_csv(path)
    if clean:
        data = data[data.occupation != '?']
        # create numerical columns representing the categorical data
        data['target_above'] = np.where(data.income == '<=50K', 0, 1)
        data['workclass_num'] = data.workclass.map({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
        data['marital_num'] = data['marital.status'].map({'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3, 'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})
        data['race_num'] = data.race.map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
        data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
        data['relative_num'] = data.relationship.map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
        #data['education_num'] = data.education.map({'1st-4th':0, 'Preschool':0, '5th-6th':0, '12th':0, '9th ':0, '7th-8th':0,
        #'10th':0, '11th':0, 'HS-grad':1, 'Some-college':2, 'Assoc-voc':3, 'Assoc-acdm':3, 'Bachelors':4, 'Masters':5, 'Doctorate':6, 'Prof-school':6, })
        data[['Male', 'Female']] = pd.get_dummies(data.sex_num)
        del data['sex_num']
        data[['relationship_no', 'relationship_yes']] = pd.get_dummies(data.relative_num)
        del data['relative_num']

        df = data[['workclass_num', 'education.num', 'marital_num', 'race_num', 'Male', 'Female', 'relationship_no', 'relationship_yes', 'capital.gain', 'capital.loss', 'target_above']]
        return df
    else:
        return data

#Only use for Australian Data
def load_data_gridsearch(path = 'C:\\Users\\jdine\\Documents\\1.MachineLearning\\hicss2020-master\\data\\australian.dat',
                         columns = 16, outlier_columns = None, scale_columns = None):
    '''
    Function to use when utilizing Hyperas for Keras GridSearchCV
    PARAMETERS:
    path: str; path to data file. Default path defined for Jake's PC

    drop_outliers: bool; If True, drop rows with outliers present

    columns: int; number of columns including index columns

    outlier_columns: list; Pass list of columns you want to search/remove outliers for

    normalize_columns: list; Pass list of columns you want to normalize. We use MinMaxScaler because no features follow Gaussian Distribution
    Returns:
        Dataframe Object
    Underloaded Method for loading data
    '''
    cols = [("A" + str(i)) for i in range(1,columns)]
    df = pd.read_csv(path, header = None, delimiter= " ", names= cols)

    if outlier_columns:
        Outliers_to_drop = detect_outliers(df,1,outlier_columns)
        df = df.drop(df.index[Outliers_to_drop])

    if scale_columns:
        scaler = MinMaxScaler()
        x= df[scale_columns]
        df[scale_columns] = scaler.fit_transform(x)

    return df

#This works the same way as Sklearn's train/test split function with addition of a binary flag
#that allows for one hot encoding of the response variable. I
def split_data(df, keras = False, testSize = 0.2, randomState = None):
    '''
    PARAMETERS:
		df: dataframe object
        keras: True if Using MLP. One hot encoding of response variable
        testSize = split between train and test set
        randomState = seeding

    RETURN:
        4 objects for model building
        xtrain, xtest, ytrain, ytest

    EXAMPLE:
        X_train, X_test, y_train, y_test = split_data(df = df, keras = False, testSize = 0.2, randomState = 123)

    '''
    if not keras:
        X, y = df.iloc[:,:-1], df.iloc[:,-1] #Split Dependent / Independent
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, stratify = y, test_size = testSize,
                                                            shuffle = True, random_state = randomState) #Train Test Split
        return xtrain, xtest, ytrain, ytest
    else:
        if target:
            target = df[target]
        else:
            target = df.columns[-1] #Fetch last columns name
        X, y = to_xy(df, target)

        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = testSize,
                                                            shuffle = True, random_state = randomState) #Train Test Split
        return xtrain, xtest, ytrain, ytest

##################################################
# Utility code for detecting outliers - Tukey Method
##################################################

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.

    Parameters:
        df: Dataframe object
        n: int; specifies the thresholded integer count of outliers per observation
        features: Specify which features to search
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers



##################################################
# Utility code for measuring model performance given dataset size
##################################################
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
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

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
