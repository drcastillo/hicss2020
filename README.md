# hicss2020
Repo for the work required for the HICSS conference.

# Getting Started

- 1:
  - You can pip install the following packages that you may not have installed. This should solve most of the dependency issues:
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
  After installing the necessary packages, there are some jupyter magic commands that are run in the first cell of the notebook, namely for plotting purposes. The following must be run to render interactive jupyter plots via plotly:
  ```python
  !jupyter nbextension enable --py widgetsnbextension
  ```


This is a concatenation of model_train and explainability meant for reproducibility. You can train your models from scratch and generate the local attribution values, although shap values take a large number of samples to converge.

# Brief Function Descriptions
Most functions were designed using Plotly to add interactivity and lessen dependency on manual parameter entry.
A few examples.

### Sklearn Feature Importance Graphs
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/sklearnfeatimp_jpg.PNG "Logo Title Text 1")

### Perturb Graphs
  - Two modes: Accuracy & proportion.
    'accuracy' shows the percentage of correct predictions on the holdout set as we iterate through a range of perturbations,
    with a perturbation of 1 = no perturbation at all (scalar multiple of 1 of specified colmns).
    'proportion' shows the percentage of observations classified as being of class 1 as we iterate through perturbations.
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/pert_graph_2.jpg "Logo Title Text 1")  

### Lime Local Explanations
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/lime_local.jpg "Logo Title Text 1")  

### Shap Local Explanations
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/shap_local.jpg "Logo Title Text 1")  

### Shap Summary Plots
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/shap_summary.jpg "Logo Title Text 1")  

### GAM explanations
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/Explanation_BAD.png "Logo Title Text 1")  
  ![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/Explanation_GOOD.png "Logo Title Text 1")  


### References
1. Ancona, M., Ceolini, E., Gross, M. A unified view of gradient-based attribution methods for deep neural networks. ArXiv 2017.
2. Breiman, L., Friedman, C., Stone, C., and Olshen, R., Classification and Regression Trees, Taylor & Francis, 1984.
3. Chen, T., and Guestrin, C. 2016. Xgboost: A scalable tree boosting system. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, 785–794. New York, NY, USA: ACM.
5. Donges, N. The random forest algorithm. https://machinelearning-blog.com 6. Friedman, J.H. Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 2001, Vol. 29, No. 5, 1189–1232 7. Ho, T.K.. Random decision forests. Proceedings of the 3rd International Conference on Document Analysis and Recognition, Montreal, QC, 14–16 August 1995. pp. 278–282
8. Ibrahim, M., Louie, M., Modarres, C., Paisley, J. Global explanations of neural networks: Mapping the landscape of predictions. AAAI/ACM Conference on Artificial Intelligence, Ethics and Society, Honolulu, HI, Jan 27-28, 2019. arXiv:1902.02384v1
9. Kridel, D. and Dolk, D. Automated self-service modeling: Predictive analytics as a service. Information Systems for e_Business Management (11:1). (2013), 119-140.
10. Lipovetsky, S. Entropy criterion in logistic regression and Shapley value of predictors, Journal of Modern Applied Statistical Methods (5:1), (2006) 95-106.
11. Lipovetsky, S. and Conklin, M. Analysis of regression in game theory approach. In: Applied Stochastic Models in Business and Industry 17.4 (2001), pp. 319–330.
12. Lundberg and Lee. A unified approach to interpreting model predictions. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. 13. Mishra, M. Hands-On introduction to Scikit-learn (sklearn). Towards Data Science. https://towardsdatascience.com/hands-on-introduction-to-scikit-learn-sklearn-f3df652ff8f2 2018.
14. Modarres, C., Ibrahim, M., Louie, M., Paisley, J. Towards explainable deep learning for credit lending: A case study. The Thirty-second Annual Conference on Neural Information Processing Systems (NeurIPS), 2018. 15. C. Molnar. Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. March 2019.
16. J. Pearl and D. Mackenzie. The Book of Why: The New Science of Cause and Effect. Basic Books. May 2018.
17. Quinlan, "Simplifying decision trees", Int J Man-Machine Studies 27, Dec 1987, 221-234.
18. Reagen, B., Whatmough, P., Adolf, R., Rama, S., Lee, H., Lee, Hernández-Lobato, H., Wei, G-Y., Brooks, D. Minerva: Enabling low-power, highly-accurate deep neural network accelerators. ISCA 2016.
19. Ribeiro, M., Singh, S. and Guestrin, C. “why should I trust you?”: Explaining the predictions of any classifier. In Knowledge Discovery and Data Mining (KDD), 2016.
20. Ribeiro, M., Singh, S. and Guestrin, C.. Model-agnostic interpretability of machine learning. In Human Interpretability in Machine Learning workshop, ICML ’16, 2016.
21. Shrikumar, A.; Greenside, P.; and Kundaje, A. Learning important features through propagating activation differences. arXiv:1704.02685 2017.
22. Sundararajan, M.; Taly, A.; and Yan, Q. Axiomatic attribution for deep networks. arXiv:1703.01365 2017.
23. Tegmark, M. Life 3.0: Being Human in the Age of Artificial Intelligence. Vintage Books, 2018. 24. Train, K. Discrete Choice Methods with Simulation. Cambridge University Press 1st ed., 2003 2nd edition, 2009.
