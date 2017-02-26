import numpy as np
import theano
import lasagne
import theano.tensor as T


if __name__ == "__main__":
    SIZE_NBR_INPUT=16
    BATCH_SIZE=1 # increasing this will help stabilize the gradient. I've already tried it without success.

    X = np.genfromtxt("../data/input.txt", delimiter=1)
    Y = np.genfromtxt("../data/output.txt", delimiter=1)

    # we batchify the data
    random_data_X = X.reshape(len(X)/BATCH_SIZE, BATCH_SIZE, 2*SIZE_NBR_INPUT)
    random_data_Y = Y.reshape(len(Y)/BATCH_SIZE, BATCH_SIZE, SIZE_NBR_INPUT)

    # random_data_X = np.random.rand(100, 2*SIZE_NBR_INPUT, )
    random_test_data_X = X
    # for e in random_data_X:
        # print e
    # random_data_Y = np.zeros((100, SIZE_NBR_INPUT)) #np.random.rand(100, SIZE_NBR_INPUT)
    # random_data_Y = np.sum(random_data_X,

    # create Theano variables for input and target minibatch
    input_var = T.dmatrix('X')
    target_var = T.dvector('y')

    # We create the layesr of our NN
    l_in = lasagne.layers.InputLayer((BATCH_SIZE, 2*SIZE_NBR_INPUT), input_var=input_var)
    l_hidden = lasagne.layers.DenseLayer(l_in, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_hidden = lasagne.layers.DenseLayer(l_in, num_units=SIZE_NBR_INPUT+1, nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_hidden, num_units=SIZE_NBR_INPUT, nonlinearity=lasagne.nonlinearities.sigmoid)

    # create loss function
    prediction = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    # create parameter update expressions
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)


    # train network (assuming you've got some training data in numpy arrays)
    NB_EPOCH = 7000
    for epoch in range(NB_EPOCH):
        loss = 0
        for i in range(len(random_data_X)):
        # for input_batch, target_batch in zip(random_data_X, random_data_Y):
            input_batch = random_data_X[i][0]
            target_batch = random_data_Y[i][0]
            train_fn_result = train_fn(np.array([input_batch]), target_batch)
            loss += train_fn_result
        if epoch % 5 == 0:
            print("Epoch %d: Loss %g" % (epoch + 1, loss / len(random_data_X)))
            # use trained network for predictions
            test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
            predict_fn = theano.function([input_var], test_prediction)
            nb_errors = []
            for j in range(len(X)):
                pred_bin = 1*(0.5 < predict_fn([X[j]])[0])
                nb_errors.append(np.sum(np.absolute(Y[j] - pred_bin)))
                # print("bit errors:\n %r" %  ( np.absolute(random_data_Y[i] - predict_fn([random_test_data_X[i]])[0])))
            print("Average nb errors: ", np.mean(nb_errors))

    # The following function will output both the out and the hidden layer's
    # output when some data is applied to it
    # f = theano.function([l_in.input_var], lasagne.layers.get_output(l_out, l_hidden))
    # print f(random_data_X[0])[1]


