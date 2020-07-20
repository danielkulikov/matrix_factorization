# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:18:13 2020

@author: Daniel Kulikov

Matrix factorization from scratch. Using the toy dataset
"""


class matrix_factorizer():
    def __init__(self, M, params, dim):
        """
        Initializes a matrix factorizer that trains by stochastic gradient descent.
        
        Arguments
        M: user-item rating matrix
        params: hyperparameter dictionary with keys:
            eps: learning rate
            lambda: regularization parameter
            num_iter: number of iterations to run SGD\
        dim: number of latent dimensions to select
        """
        
        self.params = params
        self.M = M
        self.total_users, self.total_items = M.shape
    
    def train(self):
        """
        Trains the matrix factorization model.
        """
        # get num_iterations and hyperparameters
        num_iter = self.params["num_iter"]
  
        # initialize model parameters
        
        for i in range(num_iter):
            # run one iteration of stochastic gradient descent
            
            # compute loss
            
            # print update
            
            pass
        
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
        pass
        
    def compute_rating(self, i, j=None):
        pass
        
    def compute_large_matrix(self):
        pass
    
    def mean_squared_error(self):
        """
        Computes the total mean squared error for the matrix
        """
        
if __name__ == "__main__":
    # test the matrix factorization on the small movielens dataset - 100k ratings
    # load data
    # not worrying about model evaluation here (splitting into train/test sets)
    # we just want to get some predictions and see if they are reasonable

    pass