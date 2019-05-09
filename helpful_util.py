#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Reference:


from __future__ import print_function

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
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.models import load_model
from keras.models import model_from_json



 
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
                 fontsize=14,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
##################################################
# Utility code for Fetching data files from dir        
##################################################
def list_dir(verbose = True):
    '''
    function to list the contents of current working directory
    return:: list of branches
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

def fetch_data_path (folder = 'data'):
    '''
    function to string concat holistic path to data files w/ user input
    arguments:
        pass in string of data folder
    return :: string of concatenated path to data file
    
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
	
##################################################
# Utility code for detecting outliers - Tukey Method
##################################################	

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
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
    
