import numpy as np
import theano
import lasagne
import theano.tensor as T

if __name__ == "__main__":
    X ="../data/X.txt"
    Y ="../data/Y.txt"

    # create Theano variables for input and target minibatch
    input_var = T.tensor4('X')
    target_var = T.ivector('y')



