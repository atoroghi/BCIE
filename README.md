# BK-KGE
# Bayesian Keyphrase Critiquing with Knowledge Graph Embedding


#### Dataset 
dataset.py is used to encode the KG triplets into integer values, extract the list of items and users, save the record of items that each user likes and perform manipulations on the encoded triplets that are necessary for training and testing.

#### Trainer
trainer.py is used for training SimplE and get the initial embeddings for each entity and relation. The models will be saved in the *models* folder.


#### Tester

tester.py is used for testing and performing validation on trained SimplE models.

#### SimplE

Defining the embedding network and methods for passes through it.
#### Measure
Metrics for evaluation of the embedding class and the recommender system. (not complete yet.)
#### Updater

Performing one step of Bayesian Updating using Laplace approximation and returning the user posterior mean and covariance.

#### Recommender
Performing recommendation operations. Its methods will be used in the main critiquing loop for performing pre-critiquing recommendation, post-critiquing recommendation, obtaining the critique candidates and choosing among them. It has lots of smelly code and I will make the code cleaner and adjusted. (not complete yet.)

#### main-train
The main function using which I train the embeddings and test the trained models. 

#### main-datasetcheck
This was actually used just to check the dataset class, but now I perform initial experiments such as observing the effect of noise or generalization on synthetic data using it. After these experiments, it won't be used so you can just ignore it.

#### main
Supposed to be used as the main critiquing loop simulator. Its code is also smelly and incomplete.


In order to run the code, you can just call the main function that you intend to use and input the arguments, e.g. using:

 `python main-datasetcheck.py -ne 5000 -ni 1 -lr 0.05 -reg 0.03 -dataset ML -emb_dim 5 -neg_ratio 10 -batch_size 250 -save_each 200 -max_iters_laplace 10000 -alpha 0.01 -etta 15000`
 
 #### datasets
 Includes the synthetic datasets that I created for my experiments. It is tried to have a synthetic dataset close to what a real movie dataset will look like (consisting of 1-1, 1-N, N-1,N-N) relations. The `SimilarML` folder contains the duplicated version in which each item that our studied users (user 0 to 3) like have a similar counterpart which will be implicitly critiqued.
 
