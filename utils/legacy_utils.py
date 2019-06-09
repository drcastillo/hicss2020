from utils.helpful_util import *

####################################################################################################
# Utilities for Fetching data files from dir
####################################################################################################
def list_dir(verbose=True):
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
    for i, j in enumerate(tree):
        idx.append(i)
        contents.append(j)
    if verbose:
        print("Working Dir: {}".format(cwd))
        print("Returning Contents of Working Directory..")
    return contents, dict(zip(idx, contents))


def fetch_data_path(folder='data'):
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
            print("Path to Data Stored: {}".format(path + "\\" + dataPath))
            return path + "\\" + dataPath
    except:
        print("Invalid Selection")
    return None

####################################################################################################
# Utilities for Loading various data sets into memory w/ some cleaning
####################################################################################################



#This works the same way as Sklearn's train/test split function with addition of a binary flag
#that allows for one hot encoding of the response variable. I
def split_data(df, keras=False, testSize=0.2, target = None, randomState=None):
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
        X, y = df.iloc[:, :-1], df.iloc[:, -1]  #Split Dependent / Independent
        xtrain, xtest, ytrain, ytest = train_test_split(
            X,
            y,
            stratify=y,
            test_size=testSize,
            shuffle=True,
            random_state=randomState)  #Train Test Split
        return xtrain, xtest, ytrain, ytest
    else:
        if target:
            target = df[target]
        else:
            target = df.columns[-1]  #Fetch last columns name
        X, y = to_xy(df, target)

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=testSize, shuffle=True,
            random_state=randomState)  #Train Test Split
        return xtrain, xtest, ytrain, ytest


##################################################
# Utility code for detecting outliers - Tukey Method
##################################################


def detect_outliers(df, n, features):
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
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) |
                              (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

'''
from gam.gam.gam import GAM

local shap values for all models are stored in the obj folder
change attributions_path based on which model you want to evaluate
    obj/lendingclub/gam/rfc_local.csv
    obj/lendingclub/gam/keras_ann_local.csv
    obj/lendingclub/gam/gbc_local.csv
    obj/lendingclub/gam/logit_local.csv
#gam_keras =  GAM(attributions_path="obj/lendingclub/gam/keras_ann_local.csv", k=2)
#gam_keras_5clusters = GAM(
    attributions_path=pd.DataFrame(shap_values[str(type(model))],
                         columns=features), k=2)
#gam_logit = GAM(attributions_path="obj/lendingclub/gam/logit_local.csv", k=2)
#gam_rfc = GAM(attributions_path="obj/lendingclub/gam/rfc_local.csv", k=2)
#gam_gbc = GAM(attributions_path="obj/lendingclub/gam/gbc_local.csv", k=2)

#gam_keras.generate()
#gam_keras_5clusters.generate()
#gam_logit.generate()
#gam_rfc.generate()
#gam_gbc.generate()

#save_obj(gam_keras, '/gam/gam_keras')
#save_obj(gam_keras_5clusters, '/gam/gam_keras_5clusters')

#save_obj(gam_logit, '/gam/gam_logit')
#save_obj(gam_rfc, '/gam/gam_rfc')
#save_obj(gam_gbc, '/gam/gam_gbc')

'''
print('Commented Out Cell')

'''
Put Shap values in the format GAM needs.
e.g., [shap values x features]
GAM auto drops header
'''


def shap_to_csv(model, file_name):
    local = pd.DataFrame(shap_values[str(type(model))],
                         columns=features)  #Sample 1000 rows of shap values
    local.to_csv('obj/lendingclub/gam/{}_local.csv'.format(file_name),
                 index=False)


#shap_to_csv(gbc, 'gbc')
#shap_to_csv(logit, 'logit')
#shap_to_csv(keras_ann, 'keras_ann')
#shap_to_csv(rfc, 'rfc')

#shap_to_csv(gbc, 'gbc_200')
#shap_to_csv(logit, 'logit_200')
#shap_to_csv(keras_ann, 'keras_ann_200')
#shap_to_csv(rfc, 'rfc_200')

#deep_df = pd.DataFrame(attributions_sv, columns=features)
#deep_df = deep_df.iloc[:, -15:].melt(var_name='groups', value_name='vals')
#ax = sns.stripplot(x="vals", y="groups", data=deep_df, jitter = .5)
#plt.title('Shap Attributions')
#plt.show()

'''
Code to generate various attributions. Leave commented out. Objects stored locally.

#Code largely taken fron DeepExplain Git
#https://github.com/marcoancona/DeepExplain/blob/master/examples/mint_cnn_keras.ipynb
from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from keras.models import Model
import tensorflow as tf

with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
#    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:
    # 1. Get the input tensor to the original model
    input_tensor = keras_ann.layers[0].input

    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
    fModel = Model(inputs=input_tensor, outputs = keras_ann.layers[-2].output)
    target_tensor = fModel(input_tensor)

    xs = np.array(X_train_shap)
    ys = tf.keras.utils.to_categorical(y_train,2)

    attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
    attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
    #attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
    #attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys) #Deeplift. Incompatible w/ model arch.
    attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
    #attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

    # Compare Gradient * Input with approximate Shapley Values
    # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
    # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
    attributions_sv     = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=500)

save_obj(attributions_gradin, 'attributions_gradin')
save_obj(attributions_sal, 'attributions_sal')
save_obj(attributions_elrp, 'attributions_elrp')
save_obj(attributions_sv, 'attributions_sv')
'''

#save_obj(keras_ig_df, 'keras_ig_df')
'''
#Leave Commented out. Code to generate integrated gradients attributions
from collections import OrderedDict

from IntegratedGradients import IntegratedGradients

ig = IntegratedGradients.integrated_gradients(keras_ann)


def fetch_grads(test_data):
    grads = []
    for i in test_data.toarray():
        grads.append(ig.explain(i, num_steps=1000))
    return pd.DataFrame(grads, columns=features)


keras_ig_df = fetch_grads(encoded_test)
save_obj(keras_ig_df, 'data_objects/keras_ig_df')

'''

'''


#Dataframe Operations to concatenate all Shap Values into CSVs
rfc_shapvalues_abs = pd.DataFrame(shap_values[str(type(rfc))], columns = features).abs().sum().to_csv('obj/lendingclub/shap/rfc_shap_abs.csv')
logit_shapvalues_abs = pd.DataFrame(shap_values[str(type(logit))], columns = features).abs().sum().to_csv('obj/lendingclub/shap/logit_shapvalues_abs.csv')
gbc_shapvalues_abs = pd.DataFrame(shap_values[str(type(gbc))], columns = features).abs().sum().to_csv('obj/lendingclub/shap/gbc_shapvalues_abs.csv')
keras_ann_shapvalues_abs = pd.DataFrame(shap_values[str(type(keras_ann))], columns = features).abs().sum().to_csv('obj/lendingclub/shap/keras_ann_shapvalues_abs.csv')
sk_ann_shapvalues_abs = pd.DataFrame(shap_values[str(type(sk_ann))], columns = features).abs().sum().to_csv('obj/lendingclub/shap/sk_ann_shapvalues_abs.csv')

rfc_shapvalues = pd.DataFrame(shap_values[str(type(rfc))], columns = features).sum().to_csv('obj/lendingclub/shap/rfc_shapvalues_sum.csv')
logit_shapvalues = pd.DataFrame(shap_values[str(type(logit))], columns = features).sum().to_csv('obj/lendingclub/shap/logit_shapvalues_sum.csv')
gbc_shapvalues = pd.DataFrame(shap_values[str(type(gbc))], columns = features).sum().to_csv('obj/lendingclub/shap/gbc_shapvalues_sum.csv')
keras_ann_shapvalues = pd.DataFrame(shap_values[str(type(keras_ann))], columns = features).sum().to_csv('obj/lendingclub/shap/keras_ann_shapvalues_sum.csv')
sk_ann_shapvalues = pd.DataFrame(shap_values[str(type(sk_ann))], columns = features).sum().to_csv('obj/lendingclub/shap/sk_ann_shapvalues_sum.csv')

rfc_shapvalues_abs = pd.DataFrame(shap_values[str(type(rfc))], columns = features).abs().sum()
logit_shapvalues_abs = pd.DataFrame(shap_values[str(type(logit))], columns = features).abs().sum()
gbc_shapvalues_abs = pd.DataFrame(shap_values[str(type(gbc))], columns = features).abs().sum()
keras_ann_shapvalues_abs = pd.DataFrame(shap_values[str(type(keras_ann))], columns = features).abs().sum()
sk_ann_shapvalues_abs = pd.DataFrame(shap_values[str(type(sk_ann))], columns = features).abs().sum()

combined_shap = pd.DataFrame(rfc_shapvalues_abs, columns= ['rfc'])
combined_shap['logit'] = logit_shapvalues_abs
combined_shap['gbc'] = gbc_shapvalues_abs
combined_shap['keras_ann'] = keras_ann_shapvalues_abs
combined_shap['sk_ann'] = sk_ann_shapvalues_abs
combined_shap.to_csv('obj/lendingclub/shap/All_Abs_Sum_ShapValues.csv')


rfc_shapvalues = pd.DataFrame(shap_values[str(type(rfc))], columns = features).sum()
logit_shapvalues = pd.DataFrame(shap_values[str(type(logit))], columns = features).sum()
gbc_shapvalues = pd.DataFrame(shap_values[str(type(gbc))], columns = features).sum()
keras_ann_shapvalues = pd.DataFrame(shap_values[str(type(keras_ann))], columns = features).sum()
sk_ann_shapvalues = pd.DataFrame(shap_values[str(type(sk_ann))], columns = features).sum()


combined_shap = pd.DataFrame(rfc_shapvalues, columns= ['rfc'])
combined_shap['logit'] = logit_shapvalues
combined_shap['gbc'] = gbc_shapvalues
combined_shap['keras_ann'] = keras_ann_shapvalues
combined_shap['sk_ann'] = sk_ann_shapvalues

combined_shap.to_csv('obj/lendingclub/shap/All_Sum_ShapValues.csv')


predProbs = logit.predict_proba(encoded_train.toarray())
X_design = np.hstack(
    [np.ones((encoded_train.toarray().shape[0], 1)),
     encoded_train.toarray()])
V = np.diagflat(np.product(predProbs, axis=1))
covLogit = np.linalg.inv(X_design.T @ V @ X_design)
#print("Covariance matrix: ", covLogit)
# Standard errors
print("Standard errors: ", np.sqrt(np.diag(covLogit)))
'''
