# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:18:13 2020

@author: Daniel Kulikov

Matrix factorization from scratch. Using the toy dataset
"""
import numpy as np

class matrix_factorizer():
    def __init__(self, M, params, dim):
        """
        Initializes a matrix factorizer that trains by stochastic gradient descent.
        
        Arguments
        M: user-item rating matrix
        params: hyperparameter dictionary with keys:
            eps: learning rate
            lmbda: regularization parameter
            num_iter: number of iterations to run SGD\
        dim: number of latent dimensions to select
        """
        
        self.params = params
        self.M = M
        self.total_users, self.total_items = M.shape
        self.dim = dim
        self.P, self.Q = None, None
    
    def train(self):
        """
        Trains the matrix factorization model.
        """
        # get num_iterations and hyperparameters
        num_iter = self.params["num_iter"]
  
        # initialize our P,Q matrices (effectively our "weights")
        self.P = np.random.normal(scale=1, size=(self.total_users, self.dim))
        self.Q = np.random.normal(scale=1, size=(self.total_items, self.dim))
        # initialize our biases
        self.b_users = np.zeros(self.total_users)
        self.b_items = np.zeros(self.total_items)
                                
        # get all non-zero (AKA actually reviewed) instances in the rating matrix
        # we'll use these to train the model
        
        for i in range(num_iter):
            # compute loss
            loss = self.compute_loss()
            #print(loss)
            # run one iteration of SGD and update parameters
            self.update_parameters()
            # print update
            if(i % 5 == 0):
                print("Epoch: ", i, " Loss: ", loss)
               
        
    def compute_loss(self):
        """
        Compute the loss of the current state of the model.
        """
        if(self.params["loss"] == "mse"):
            return self.mean_squared_error()
    
    def update_parameters(self, batch=False):
        """
        Runs one iteration of stochastic gradient descent on 
        the matrix parameters. Optional batch mode. 
        """
        
        # update weights
        
        # update biases
        
        # regularization
        
    def compute_rating(self, i, j):
        """
        Computes the predicted rating for a particular user/item pair.
        """
        return np.dot(self.P[i, :], self.Q[j, :].T)
        
    def compute_large_matrix(self):
        return np.dot(self.P, self.Q.T)
    
    def mean_squared_error(self, regularize=False):
        """
        Computes the total mean squared error for the matrix
        """
        
        reg = 0
        
        den = 1/(self.total_users*self.total_items)
        pred = self.compute_large_matrix()
        se = np.sum((self.M - pred)**2)
        
        if(regularize):
            reg = self.params["lmbda"]*(0)
            
        return np.multiply(den, se) + reg
              
        
if __name__ == "__main__":
    # test the matrix factorization on the small movielens dataset - 100k ratings
    # load data
    movie_data = np.genfromtxt ('ml-latest-small/ratings.csv', delimiter=",")[1:, 0:3]
    movie_data = movie_data[np.logical_and(movie_data[:,0] <=100, movie_data[:,1] <= 2000)]
    
    # lets take a subset of 100 users and 2000 movies
    total_users = movie_data[:,0].max()
    total_movies = movie_data[:,1].max()
    
    # set up the rating matrix 
    rating_matrix = np.zeros((int(total_users), int(total_movies)))
    
    # Aside: we're not worrying about model evaluation here (splitting into train/test sets)
    # we just want to get some predictions and see if they are reasonable
    
    # set up parameters
    params = {}
    eps = 0.1
    lmbda = 0.1
    num_iter = 10
    dim = 3
    params["eps"], params["lmbda"], params["num_iter"] = eps, lmbda, num_iter
    params["loss"] = "mse"
    
    mf = matrix_factorizer(rating_matrix, params, dim)
    mf.train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    