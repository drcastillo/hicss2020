#Local Explainability

#Only need to run this once. Pickled all Shap values.
def get_shap_values(model):
    if type(model) == keras.engine.training.Model:
        f = lambda x: model.predict(x)[:, 1]
    else:
        f = lambda x: model.predict_proba(x)[:, 1]
    med = X_train_shap.median().values.reshape((1, X_train_shap.shape[1]))
    explainer = shap.KernelExplainer(f, med)
    shap_values = explainer.shap_values(X_test_shap, samples =500)
    return shap_values


rfc_shap_values = get_shap_values(rfc)
gbc_shap_values = get_shap_values(gbc)
logit_shap_values = get_shap_values(logit)
sk_ann_shap_values = get_shap_values(sk_ann)
keras_ann_shap_values = get_shap_values(keras_ann)

shap_values = {str(type(rfc)) : rfc_shap_values,
               str(type(gbc)) : gbc_shap_values,
               str(type(logit)) : logit_shap_values,
str(type(sk_ann)) : sk_ann_shap_values,
               str(type(keras_ann)) : keras_ann_shap_values}

save_obj(shap_values, 'shap_values')


#def get_base_values(model):
#    #Keras doesn't have a model.predict_proba function. It outputs the probabilities via predict method
#    if type(model) == keras.engine.training.Model:
#        f = lambda x: model.predict(x)[:, 1]
#    else:
#        f = lambda x: model.predict_proba(x)[:, 1]
#    #We use the median as a proxy for computational efficiency. Rather than getting the expected value over the whole
#    #training distribution, we get E(f(x)) over the median of the training set, e.g., model.predict(median(xi))
#    med = X_train_shap.median().values.reshape((1, X_train_shap.shape[1]))
#    explainer = shap.KernelExplainer(f, med)
#    return explainer.expected_value



#rfc_base_value = get_base_values(rfc)
#gbc_base_value = get_base_values(gbc)
#logit_base_value = get_base_values(logit)
#sk_ann_base_value = get_base_values(sk_ann)
#keras_ann_base_value = get_base_values(keras_ann)

#base_values =  {'rfc' : rfc_base_value, 'gbc' : gbc_base_value, 'logit' : logit_base_value,
#'sklearn_ann' : sk_ann_base_value, 'keras_ann' : keras_ann_base_value}

#Removing need for DeepExplain pip install. Stor


#If this cell breaks, need to run:
#pip install -e git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain

#import keras
#from keras.datasets import mnist
#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
#from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
#import tensorflow as tf
#from keras.utils import multi_gpu_model
# Import DeepExplain
#from deepexplain.tensorflow import DeepExplain
#from livelossplot import PlotLossesKeras


#Code largely taken fron DeepExplain Git
#https://github.com/marcoancona/DeepExplain/blob/master/examples/mint_cnn_keras.ipynb

#with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
#    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:
    # 1. Get the input tensor to the original model
#    input_tensor = keras_ann.layers[0].input

    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
#    fModel = Model(inputs=input_tensor, outputs = keras_ann.layers[-2].output)
#    target_tensor = fModel(input_tensor)

#    xs = np.array(X_train_shap)
#    ys = tf.keras.utils.to_categorical(y_train,2)

#    attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
#    attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
    #attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
    #attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys) #Deeplift. Incompatible w/ model arch.
#    attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
    #attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

    # Compare Gradient * Input with approximate Shapley Values
    # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
    # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
#    attributions_sv     = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)

#save_obj(attributions_gradin, 'attributions_gradin')
#save_obj(attributions_sal, 'attributions_sal')
#save_obj(attributions_elrp, 'attributions_elrp')
#save_obj(attributions_sv, 'attributions_sv')
