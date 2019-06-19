# hicss2020
Repo for the work required for the HICSS conference

# Getting Started
Main work is done in the notebooks labelled 'explainability'

A few ways to go about installing requirements (requirements.txt is buggy, don't build conda from):

- 1:
  - You can pip install the following packages that you may not have installed. This should solve most of the dependecy issues:
  ```python
  conda create --name tf_gpu tensorflow-gpu
  conda activate tf_gpu
  conda install jupyter
  conda install keras
  conda install pandas
  conda install scikit-learn
  conda install matplotlib
  conda install seaborn
  conda install plotly
  conda install ipywidgets
  pip install cufflinks
  pip install shap==0.28.5
  pip install lime

  ```
- 2. This is still buggy:
    - You can access the notebook built on images of the git repo @ https://mybinder.org/v2/gh/drcastillo/hicss2020/master



The First 2 blocks of code deal with functions w/ global dependencies and the loading of all data objects.
Objects were stored locally during model training to cut down on load times.

# Brief Function Descriptions
Most functions were designed using Plotly to add interactivity and lessen dependency on manual parameter entry.
A few examples.
### Classification reports
- toggle model
![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/classreport_jpg.PNG "Logo Title Text 1")

### Sklearn Feature Importance Graphs
  - toggle feature count to display more or less features
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/sklearnfeatimp_jpg.PNG "Logo Title Text 1")

### Perturb Graphs
  - Two modes: Accuracy & proportion.
    'accuracy' shows the percentage of correct predictions on the holdout set as we iterate through a range of perturbations,
    with a perturbation of 1 = no perturbation at all (scalar multiple of 1 of specified colmns).
    'proportion' shows the percentage of observations classified as being of class 1 as we iterate through perturbations.
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/pert_graph_2.jpg "Logo Title Text 1")  

### Lime Local Explanations
  - toggles: model and observation. Visualize Local explanation as you cycle through different models and different observations within the test set.
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/lime_local.jpg "Logo Title Text 1")  

### Shap Local Explanations
  - toggles: model and observation. Visualize Local explanation as you cycle through different models and different observations within the test set.
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/shap_local.jpg "Logo Title Text 1")  

### Shap Summary Plots
  - toggles: model. View density dotplots to visualize feature importance at a more global level
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/shap_summary.jpg "Logo Title Text 1")  

### GAM explanations
  - toggles: models, clusters, and number of observations
  - Running GAM on > 200 sets of shap values is slow.
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/gam_jpg.PNG "Logo Title Text 1")  

### There are 3 different blocks of code that run some form of GAM
  - The first block runs the Keras Neural Net's full (4000) shap values through GAM & displays feature importance w/ 2 explanations.
      This took about 8 hours to run, so k = 2 was the only iteration I ran.
  - The second block adds functionality for on the fly testing. You can toggle the model and the number of observations to display feature importance.
  - The last block allows for the use of various attribution values inplace of Shap values. Such options include
    - Grad * Input
    - Saliency Maps
    - Layerwise Relevance Propogation
    - Integrated Gradients
