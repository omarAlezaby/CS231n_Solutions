import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_tr = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # Forwad Prob
  f = X.dot(W)
  # we add a constant of C = -MAx(fj) to surpass numerical errors
  ex = np.exp(f - np.max(f, axis = 1, keepdims = 1))
  soft = ex[range(num_tr), y] / np.sum(ex, axis = 1)
  loss = - np.sum(np.log(soft))
  loss /= num_tr
  loss += reg * np.sum(W*W)
    
  # back prob
  # I devide the back prob for 2 parts
  # first f = X*W from it dW = X.T*df
  # secound eq is softmax for f, using eq: Li=−f(yi)+log(∑j e^(fj))
  # the dervative of this eq respect to f : dfi = -1(yi) + e^(fj)/ (∑j e^(fj))
  # so we first find df and the from it find dW
  df = ex / np.sum(ex, axis = 1, keepdims = 1)
  df[range(num_tr), y] -= 1
  dW = X.T.dot(df)
  dW /= num_tr
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_tr = X.shape[0]
   # Forwad Prob
  f = X.dot(W)
  # we add a constant of C = -MAx(fj) to surpass numerical errors
  ex = np.exp(f - np.max(f, axis = 1, keepdims = 1))
  soft = ex[range(num_tr), y] / np.sum(ex, axis = 1)
  loss = - np.sum(np.log(soft))
  loss /= num_tr
  loss += reg * np.sum(W*W)
    
  # back prob
  # I devide the back prob for 2 parts
  # first f = X*W from it dW = X.T*df
  # secound eq is softmax for f, using eq: Li=−f(yi)+log(∑j e^(fj))
  # the dervative of this eq respect to f : dfi = -1(yi) + e^(fj)/ (∑j e^(fj))
  # so we first find df and the from it find dW
  df = ex / np.sum(ex, axis = 1, keepdims = 1)
  df[range(num_tr), y] -= 1
  dW = X.T.dot(df)
  dW /= num_tr
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

