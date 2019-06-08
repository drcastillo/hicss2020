from utils.helpful_util import *

def save_obj(obj, name):
    import pickle
    with open('obj/lendingclub/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    import pickle
    with open('obj/lendingclub/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

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
