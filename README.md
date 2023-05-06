**Readme Under Construction!**


This repository contains the implementation of the paper [Bayesian Critiquing with Indirect Evidence (BCIE)](https://ssanner.github.io/papers/sigir23_bcie.pdf) from SIGIR 2023.

In order to perform critiquing using the BCIE model, you can follow these steps:

## 1- Install requirements
~~~
pip install -r requirements.txt
~~~
## 2- Download or Preprocess the Datasets
You can either download our preprocessed datasets from here, or do the preprocessing on your own. The entity matching data are obtained from [KB4rec](https://github.com/RUCDM/KB4Rec).
~~~
python proc.py
python AB.py
~~~
## 3- The first step of BCIE's framework is training a SimplE model to obtain KG embedding. You can either download our trained models from here, or run the following command to perform it yourself. Running this code, the inner loop of the nested cross-validation will be performed which is used to perform hyperparameter tuning on the validation data. You can identify the hyperparameters that you want to be tuned by uncommenting the respective key and value pairs in the *param_dict* from *tune_utils.py*
~~~
python inner_cv.py -cv_type train
~~~

You can also train our baselines (WRMF and SVD) by specifying the corresponding *model_type* argument in running the *inner_loop* script. After the inner loop is finished, you can run the outer loop of nested cross validation as:
~~~
python outer_cv.py -cv_type train
~~~
## 3- In order to perform critiquing tests, you can download our tuned hyperparameters for baselines and the BCIE model from here and test it on the test set by using:
~~~
python critique.py
~~~
please remember to enter the tuned hyperparameters for the corresponding arguments.
You can also do the hyperparameter tuning by running the inner loop using:
~~~
python inner_cv.py -cv_type crit
~~~
and next, test on the outer loop using:
~~~
python outer_cv.py -cv_type crit
~~~
Again, please enter the file name arguments as the ones you used for the inner loop. 
