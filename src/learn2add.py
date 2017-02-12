import numpy as np
import theano
import lasagne
import theano.tensor as T


if __name__ == "__main__":
    SIZE_NBR_INPUT=32
    BATCH_SIZE=1

    X ="../data/X.txt"
    random_data_X = np.random.rand(100, 2*SIZE_NBR_INPUT);
    random_test_data_X = np.random.rand(1, 2*SIZE_NBR_INPUT);
    Y ="../data/Y.txt"
    # random_data_Y = np.zeros((100, SIZE_NBR_INPUT)) #np.random.rand(100, SIZE_NBR_INPUT);
    random_data_Y = np

    # create Theano variables for input and target minibatch
    input_var = T.dmatrix('X')
    target_var = T.dvector('y')

    # We create the layesr of our NN
    l_in = lasagne.layers.InputLayer((BATCH_SIZE, 2*SIZE_NBR_INPUT), input_var=input_var)
    l_hidden = lasagne.layers.DenseLayer(l_in, num_units=SIZE_NBR_INPUT) # should try this experiement without the middle layer.
    l_out = lasagne.layers.DenseLayer(l_hidden, num_units=SIZE_NBR_INPUT)

    # create loss function
    prediction = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    # create parameter update expressions
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.02, momentum=0.9)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)

    # train network (assuming you've got some training data in numpy arrays)
    NB_EPOCH = 100
    for epoch in range(NB_EPOCH):
        loss = 0
        for input_batch, target_batch in zip(random_data_X, random_data_Y):
            train_fn_result = train_fn(np.array([input_batch]), target_batch)
            loss += train_fn_result
            print("Epoch %d: Loss %g" % (epoch + 1, loss / len(random_data_X)))


    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)
    print("Predicted class for first test input:\n %r" % ("".join(map(str,list((1*(0.5 < predict_fn(random_test_data_X)[0])))))))

    # The following function will output both the out and the hidden layer's
    # output when some data is applied to it
    # f = theano.function([l_in.input_var], lasagne.layers.get_output(l_out, l_hidden))
    # print f(random_data_X[0])[1]


