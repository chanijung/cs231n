from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    scores = X.dot(W) # (N,C)
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores, axis=1) #(N,1)
 
    cross_entropy_loss_sum = 0
    ds = np.zeros(scores.shape)
    for i in range(num_train):
        exp_score_correct = exp_scores[i,y[i]]
        prob = exp_score_correct/exp_scores_sum[i]
        cross_entropy_loss_sum += (-1)*np.log(prob)
        ds[i] = exp_scores[i]/(exp_scores_sum[i]*num_train)
        ds[i,y[i]] -= 1/num_train
    loss = cross_entropy_loss_sum/num_train + reg*np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dW = X.T.dot(ds) + reg*2*W
    
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    scores = X.dot(W) # (N,C)
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores, axis=1) #(N,1)
    
    cross_entropy_loss = (-1) * np.log(exp_scores[range(num_train), y]/exp_scores_sum)
    loss = np.mean(cross_entropy_loss) + reg*np.sum(W*W)
    
    ds = exp_scores / (exp_scores_sum.reshape(exp_scores.shape[0],1)*num_train)
    ds[range(num_train), y] -= 1/num_train
    
    dW = X.T.dot(ds) + reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
