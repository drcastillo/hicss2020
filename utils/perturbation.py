import collections
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display_html, display, HTML

from keras.models import load_model
import pickle
import keras
import pandas as pd

def save_obj(obj, name):
    with open('obj/lendingclub/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/lendingclub/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_models_lendingclub():
    rf_file = "models/LendingClub/random_forest.pkl"
    gbc_file = "models/LendingClub/GBC.pkl"
    logit_file = "models/LendingClub/Logit.pkl"
    ann_file = "models/LendingClub/SklearnNeuralNet.pkl"
    path_to_ann = 'models/LendingClub/ann_deepexplain.h5'

    rfc = pickle.load(open(rf_file, 'rb'))
    logit = pickle.load(open(logit_file, 'rb'))
    gbc = pickle.load(open(gbc_file, 'rb'))
    sk_ann = pickle.load(open(ann_file, 'rb'))
    keras_ann = load_model(path_to_ann)
    return rfc, gbc, logit, keras_ann, sk_ann

def load_models_uci():
    sklearn_nn_file = 'models/UCI_Census/SklearnNeuralNet.pkl'
    keras_ann_file = 'models/UCI_Census/ann_deepexplain.h5'
    rf_file = "models/UCI_Census/random_forest.pkl"
    gbc_file = "models/UCI_Census/GBC.pkl"
    logit_file = "models/UCI_Census/Logit.pkl"

    rfc = pickle.load(open(rf_file, 'rb'))
    logit = pickle.load(open(logit_file, 'rb'))
    gbc = pickle.load(open(gbc_file, 'rb'))
    sk_ann = pickle.load(open(ann_file, 'rb'))
    keras_ann = load_model(path_to_ann)
    return rfc, gbc, logit, keras_ann, sk_ann

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

        self.a = [str(i) + '%' for i in range(50, 151) if i %10 == 0]
        self.b = [(i / 100) for i in range(50, 151) if i %10 == 0]
        self.pertu = dict(zip(self.a, self.b))
        self.pert = [i for i in self.pertu.values()]

        if 'uci' in self.data:
            self.rfc, self.gbc, self.logit, self.keras_ann, self.sk_ann = load_models_uci()
        elif 'lending' in self.data:
            self.rfc, self.gbc, self.logit, self.keras_ann, self.sk_ann = load_models_lendingclub()

    def perturb_graph_int(self, model, mode, column):
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
        clone = self.X.copy()
        column = column


        preds = []
        mean = []
        num_of_1s = []
        for i in self.pert:
            clone[column] = self.X[column] * (i)
            mean.append(str(column) + ':' + str(np.round(clone[column].mean())))

            if type(model) ==keras.engine.sequential.Sequential:
                preds.append(sklearn.metrics.accuracy_score(self.y, model.predict_classes(clone))* 100)
                num_of_1s.append((collections.Counter(model.predict_classes(clone))[1] / self.y.shape[0]) *100)
            else:
                preds.append(sklearn.metrics.accuracy_score(self.y, model.predict(clone))* 100)
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


        if isinstance(column, str):
            for i, txt in enumerate(mean):
                if i % 5 == 0:
                    if 'accuracy' in mode:
                        ax1.annotate(txt, (self.pert[i], preds[i]))
                    if 'proportion' in mode:
                        ax1.annotate(txt, (self.pert[i], num_of_1s[i]))

    def perturb_graph_cons_int(self, mode, column):

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
        clone = self.X.copy()
        column = column

        rfc_preds = []
        gbc_preds = []
        logit_preds = []
        keras_ann_preds = []
        sk_ann_preds = []

        rfc_1_preds =[]
        gbc_1_preds =[]
        logit_1_preds =[]
        keras_ann_1_preds = []
        sk_ann_1_preds = []
        for i in self.pert:
            clone[column] = self.X[column] * (i)

            rfc_preds.append(sklearn.metrics.accuracy_score(self.y, self.rfc.predict(clone))*100)
            gbc_preds.append(sklearn.metrics.accuracy_score(self.y, self.gbc.predict(clone))*100)
            logit_preds.append(sklearn.metrics.accuracy_score(self.y, self.logit.predict(clone))*100)
            keras_ann_preds.append(sklearn.metrics.accuracy_score(self.y, self.keras_ann.predict_classes(clone))* 100)
            sk_ann_preds.append(sklearn.metrics.accuracy_score(self.y, self.sk_ann.predict(clone))*100)

            rfc_1_preds.append((collections.Counter(self.rfc.predict(clone))[1] / self.y.shape[0]) *100)
            gbc_1_preds.append((collections.Counter(self.gbc.predict(clone))[1] / self.y.shape[0]) *100)
            logit_1_preds.append((collections.Counter(self.logit.predict(clone))[1] / self.y.shape[0]) *100)
            keras_ann_1_preds.append((collections.Counter(self.keras_ann.predict_classes(clone))[1] / self.y.shape[0]) *100)
            sk_ann_1_preds.append((collections.Counter(self.sk_ann.predict(clone))[1] / self.y.shape[0]) *100)

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))
        if 'accuracy' in mode:
            model_preds = {'rfc' : rfc_preds, 'gbc' : gbc_preds, 'logit' : logit_preds, 'keras_ann' : keras_ann_preds, 'sklearn_ann' : sk_ann_preds}
            for i in model_preds.keys():
                sns.lineplot(x=self.pert, y=model_preds[i], ax=ax1)
                ax1.set_ylabel('Accuracy %', fontsize=15)
                ax1.set_title('Accuracy : {}'.format(
                column.upper()),
                          fontsize=25)
                ax1.set_ylim(0,100)

        elif 'proportion' in mode:
            model_1_preds = {'rfc' : rfc_1_preds, 'gbc' : gbc_1_preds, 'logit' : logit_1_preds, 'keras_ann' : keras_ann_1_preds, 'sklearn_ann' : sk_ann_1_preds}
            for i in model_1_preds.keys():
                sns.lineplot(x=self.pert, y=model_1_preds[i], ax=ax1)
                ax1.set_ylabel('% of Predictions == 1', fontsize=15)
                ax1.set_title('Proportionality of Predictions :{}'.format(
                column.upper()),
                          fontsize=25)
                ax1.set_ylim(0,100)


        ax1.set_xlabel('{} Perturbation'.format(column), fontsize=15)
        plt.legend(title='Model',
                   loc='lower right',
                   labels=[
                       'Random Forest', 'Gradient Boosted Classifier',
                       'Logistic Regression', 'Multilayer Perceptron', 'Sklearn Neural Network'
                   ])
        if 'accuracy' in mode:
            temp = pd.DataFrame(model_preds).T
        if 'proportion' in mode:
            temp = pd.DataFrame(model_1_preds).T
        temp.columns = self.a
        print('\t\tTable Showing {} by {} perturbance percentage ***100 % is equivalent to the baseline {}%'.format(mode, column, mode))
        display(HTML(temp.to_html()))
        print('-' * 125)

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
        temp = self.X.copy()
        temp[column] = temp[column] * scalar
        print("Perturbing Feature: {} by {}".format(column,scalar))
        print('-' * 75)
        models_str = ['Random Forest', 'Gradient Boosted Classifier', 'Logistic Regression', 'Sklearn Neural Network',
                 'Multilayer Perceptron']
        models = [self.rfc, self.gbc, self.logit, self.sk_ann, self.keras_ann]
        for i,j in zip(models_str, models):
            print("\033[1m {} \033[0m".format(i))
            try:
                bef_acc = np.round(sklearn.metrics.accuracy_score(self.y, j.predict(self.X))*100,4)
                print("\tBefore Perturbation, Accuracy: {}%".format(bef_acc))
                aft_acc = np.round(sklearn.metrics.accuracy_score(self.y, j.predict(temp))*100,4)
                print("\tAfter Perturbation, Accuracy: {}%".format(aft_acc))
                print("\tNumber of '1' Predictions, Before Perturbation: {}".format(collections.Counter(j.predict(self.X))[1]))
                print("\tNumber of '1' Predictions, After Perturbation: {}".format(collections.Counter(j.predict(temp))[1]))
                print('-' * 75)
            except:
                bef_acc = np.round(sklearn.metrics.accuracy_score(self.y, j.predict_classes(self.X))*100,4)
                print("\tBefore Perturbation, Accuracy: {}%".format(bef_acc))
                aft_acc = np.round(sklearn.metrics.accuracy_score(self.y, j.predict_classes(temp))*100,4)
                print("\tAfter Perturbation, Accuracy: {}%".format(aft_acc))
                print("\tNumber of '1' Predictions, Before Perturbation: {}".format(collections.Counter(j.predict_classes(self.X))[1]))
                print("\tNumber of '1' Predictions, After Perturbation: {}\n".format(collections.Counter(j.predict_classes(temp))[1]))
    def categorical_perturb_loangrade(self,column, grouping):
        #print("Perturbing column: {}".format(column))
        b = [(i / 100) for i in range(0, 101) if i %20 == 0]
        model_str = ['Random Forest', 'Gradient Boosted Classifier', 'Logistic Regression', 'Sklearn Neural Network',
                 'Multilayer Perceptron']
        models = [self.rfc, self.gbc, self.logit, self.sk_ann, self.keras_ann]
        col = column
        cols = [i for i in self.X.columns if grouping in i]
        scores_dict1_logit = {}
        for m,n in zip(models, model_str):
            cols = [x for x in cols if x != col]
            scores_dict = {}
            #print("Model: {}".format(n))
            for i in b:
                #print("Sampling {}".format(i))
                test = self.X.copy()
                scores = []
                counter = 0
                while counter < 10:
                    idx_0s = test.index[test[col] == 0].tolist() #find all 0 indices
                    idx_1s = test.index[test[col] == 1].tolist() #find all 1 indices
                    idx = np.random.choice(idx_0s, int(len(idx_1s) *i), replace = True)#random select n indices from 0 indices
                    test[col].iloc[idx,] = 1 #change n indices from 0 to 1
                    test[cols].iloc[idx,] = 0 #change n indices from 1 to 0
                    try:
                        scores.append(collections.Counter(m.predict(test))[1])
                    except:
                        scores.append(collections.Counter(m.predict_classes(test))[1])
                    counter +=1
                    test = self.X.copy()
                scores_dict[i] = scores
            scores_dict1_logit[n] = scores_dict

        test_dict_df = pd.DataFrame()
        for i in scores_dict1_logit.keys():
            test_dict_df[i] = pd.DataFrame(scores_dict1_logit[i]).mean().values
        test_dict_df = test_dict_df.T
        #test_dict_df = test_dict_df / 2000
        test_dict_df.columns = [(i / 100) for i in range(0, 101) if i %20 == 0]
        return test_dict_df
    def categorical_perturb_loangrade_overloaded(self,column, grouping, sub_column, subgrouping):
        test = self.X.copy()

        b = [(i / 100) for i in range(0, 101) if i %20 == 0] #Step by sample size: 0:100, 10

        model_str = ['Random Forest', 'Gradient Boosted Classifier', 'Logistic Regression', 'Sklearn Neural Network',
                 'Multilayer Perceptron']
        models = [self.rfc, self.gbc, self.logit, self.sk_ann, self.keras_ann]

        col = column #Get Target Column
        cols = [i for i in self.X.columns if grouping in i] #Extract grouping cols

        sub_col = sub_column #Get Target SubColumn
        sub_cols = [i for i in self.X.columns if subgrouping in i] #Extract Sub-grouping cols

        #print("Perturbing column: {} \nPerturbing Subcolumn : {}".format(column, sub_column))
        #print('{} LoanGradeA'.format(len(test.index[test[col] == 1].tolist())))
        #print('{} Non-LoanGradeA'.format(len(test.index[test[col] == 0].tolist())))
        #print('{} LoanSubGradeA'.format(len(test.index[test[sub_column] == 1].tolist())))
        #print('{} Non-LoanSubGradeA'.format(len(test.index[test[sub_column] == 0].tolist())))

        #print('-' * 50)

        scores_dict1_logit = {}
        for m,n in zip(models, model_str):
            nontarget_cols = [x for x in cols if x != col] #Get all nontarget columns
            nontarget_subcols = [x for x in sub_cols if x != sub_column]

            scores_dict = {}
            #print("Model: {}".format(n))
            for i in b:
                test = self.X.copy() #Refresh copy each subloop for each model
                scores = [] #Empty
                counter = 0 #Reset Counter for each subloop
                #print("Changing Non-LoanGradeA's to Loan Grade A at {}".format(i))
                while counter < 10:
                    #Working on overarching Group
                    idx_0s = test.index[test[col] == 0].tolist() #find all 0 indices
                    idx_1s = test.index[test[col] == 1].tolist() #find all 1 indices
                    idx = np.random.choice(idx_0s, int(len(idx_1s) *i), replace = True) #random select n indices from 0 indices (Sample of zero indices at i*size of 1's)


                    test[col].iloc[idx,] = 1 #change n indices from 0 to 1
                    #print('{} LoanGradeA'.format(test[col].sum()))

                    for l in nontarget_cols:
                        test[l].iloc[idx,] = 0 #change n indices from 1 to 0 if not target column
                    #print('{} Non-LoanGradeA'.format(sum(test[nontarget_cols].sum())))


                    test[sub_col].iloc[idx,] = 1 #change n indices from 0 to 1
                    #print('{} LoanSubGradeA'.format(len(test.index[test[sub_col] == 1].tolist())))

                    for o in nontarget_subcols:
                        test[o].iloc[idx,] = 0 #change n indices from 1 to 0 if not target column
                    #print('{} Non-LoanSubGradeA'.format(sum(test[nontarget_subcols].sum())))




                    #print('-' * 50)

                    try:
                        scores.append(collections.Counter(m.predict(test))[1])
                    except:
                        scores.append(collections.Counter(m.predict_classes(test))[1])
                    counter +=1
                    test = self.X.copy()
                scores_dict[i] = scores
            scores_dict1_logit[n] = scores_dict

        test_dict_df = pd.DataFrame()
        for i in scores_dict1_logit.keys():
            test_dict_df[i] = pd.DataFrame(scores_dict1_logit[i]).mean().values
        test_dict_df = test_dict_df.T
        #test_dict_df = test_dict_df / 2000
        test_dict_df.columns = [(i / 100) for i in range(0, 101) if i %20 == 0]
        return test_dict_df
    def categorical_perturb_loangrade_overloaded_v2(self,column, grouping, subgrouping):
        test = self.X.copy()

        b = [(i / 100) for i in range(0, 101) if i %20 == 0] #Step by sample size: 0:100, 10

        model_str = ['Random Forest', 'Gradient Boosted Classifier', 'Logistic Regression', 'Sklearn Neural Network',
                 'Multilayer Perceptron']
        models = [self.rfc, self.gbc, self.logit, self.sk_ann, self.keras_ann]

        col = column #Get Target Column
        cols = [i for i in self.X.columns if grouping in i] #Extract grouping cols

        sub_cols = [i for i in self.X.columns if subgrouping in i] #Extract Sub-grouping cols


        scores_dict1_logit = {}
        for m,n in zip(models, model_str):
            nontarget_cols = [x for x in cols if x != col] #Get all nontarget columns
            scores_dict = {}
            for i in b:
                test = self.X.copy() #Refresh copy each subloop for each model
                scores = [] #Empty
                counter = 0 #Reset Counter for each subloop
                while counter < 1:
                    #Working on overarching Group
                    idx_0s = test.index[test[col] == 0].tolist() #find all 0 indices
                    idx_1s = test.index[test[col] == 1].tolist() #find all 1 indices
                    idx = np.random.choice(idx_0s, int(len(idx_1s) *i), replace = False) #random select n indices from 0 indices (Sample of zero indices at i*size of 1's)


                    test[col].iloc[idx,] = 1 #change n indices from 0 to 1
                    for l in nontarget_cols:
                        test[l].iloc[idx,] = 0 #change n indices from 1 to 0 if not target column

                    dict_test = {}
                    for i in idx:
                        dict_test[i] = subgrouping + 'A' + sub_cols[np.where(np.array(test[sub_cols].iloc[i,] == 1))[0][0]][-1] #New Column flag for SubLoan

                    for i in dict_test.keys(): #Loop through indices of random sample
                        sub_column = dict_test[i] #Extract new Sub Solumn per iteration
                        nontarget_subcols = [x for x in sub_cols if x != sub_column] #Find all new non-sub columns to zero out
                        test[sub_column].iloc[i,] = 1 #Turn flag at index i new subcolumn on
                        for o in nontarget_subcols: #Turn all other subcolumns off
                            test[o].iloc[i,] = 0


                    try:
                        scores.append(collections.Counter(m.predict(test))[1])
                    except:
                        scores.append(collections.Counter(m.predict_classes(test))[1])
                    counter +=1
                    test = self.X.copy()
                scores_dict[i] = scores
            scores_dict1_logit[n] = scores_dict

        test_dict_df = pd.DataFrame()
        for i in scores_dict1_logit.keys():
            test_dict_df[i] = pd.DataFrame(scores_dict1_logit[i]).mean().values
        test_dict_df = test_dict_df.T
        #test_dict_df = test_dict_df / 2000
        test_dict_df.columns = [(i / 100) for i in range(0, 101) if i %20 == 0]
        return test_dict_df
