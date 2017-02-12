import numpy as np
import theano
import lasagne
import theano.tensor as T

def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 32, 28), input_var=input_var)

if __name__ == "__main__":
    SIZE_NBR_INPUT=32

    X ="../data/X.txt"
    random_data_X = np.random.rand(100, 2*SIZE_NBR_INPUT);
    Y ="../data/Y.txt"
    random_data_Y = np.random.rand(100, 2*SIZE_NBR_INPUT);

    # create Theano variables for input and target minibatch
    input_var = T.bvector('X')
    target_var = T.bvector('y')

    l_in = lasagne.layers.InputLayer((2*SIZE_NBR_INPUT,), input_var=input_var)
    # should try this experiement without the middle layer.

    l_hidden = lasagne.layers.DenseLayer(l_in, num_units=SIZE_NBR_INPUT)
    l_out = lasagne.layers.DenseLayer(l_hidden, num_units=SIZE_NBR_INPUT, nonlinearity=T.nnet.softmax)


    loss = lasagne.objectives.squared_error(prediction, target_var)

    # The following function will output both the out and the hidden layer's
    # output when some data is applied to it
    f = theano.function([l_in.input_var], lasagne.layers.get_output(l_out, l_hidden))
    print f(random_data_X[0])

