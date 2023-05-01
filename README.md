Readme Under Construction!


This repository contains the implementation of the paper Bayesian Critiquing with Indirect Evidence (BCIE) from SIGIR 2023.

In order to perform critiquing using the BCIE model, you can follow these steps:

## 1- Install requirements
~~~
pip install -r requirements.txt
~~~
## 1- Download or Preprocess the Datasets
You can either download our preprocessed datasets from here, or do the preprocessing on your own. The entity matching data are obtained from [KB4rec](https://github.com/RUCDM/KB4Rec).
~~~
python proc.py
python LFM.py
python AB.py
~~~
