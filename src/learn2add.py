from future import print_function
import numpy as np
import theano
import lasagne
import theano.tensor as T


# Helper function:
# ( np, ratio_training, ratio_valadation, ratio_testing)-> [np, np, np]
def prepare_data(np_array, training_r, validation_r) :
  return [training_array, validation_array]


def creating_model(nb_bits_number, batch_size=1):

    # create Theano variables for input and target minibatch
    input_var = T.dmatrix('X')
    target_var = T.dvector('y')

    l_in = lasagne.layers.InputLayer(
            (batch_size, 2*nb_bits_number),
            input_var=input_var)
    l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=2*nb_bits_number,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=nb_bits_number,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # Creating loss function
    prediction = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    # create parameter update expressions
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss,
            params,
            learning_rate=0.01,
            momentum=0.9)

    # Compile training function that updates parameters and returns training
    # loss
    train_fn = theano.function(
            [input_var, target_var],
            loss,
            updates=updates,
            allow_input_downcast=True)

    return input_var, train_fn, l_out


def get_sets(path_main_dataset_x, path_main_dataset_y):

    X = np.genfromtxt(path_main_dataset_x, delimiter=1)
    Y = np.genfromtxt(path_main_dataset_y,  delimiter=1)

    return [X, Y, [], [], [], []]


def main(nb_epoch, path_main_dataset_x, path_main_dataset_y, batch_size=1):
    # Increasing the batch_size will help stabilize the gradient.  I've already
    # tried it without much success.
    batch_size = 1

    train_x, train_y, val_x, val_y, test_x, test_y = get_sets(
            path_main_dataset_x,
            path_main_dataset_y)

    # This is the number of bits of one of the number we are adding
    nb_bits_number = train_x.shape[1] / 2

    # Converting the data into mini-batch
    train_x_batch = train_x.reshape(
            len(train_x)/batch_size,
            batch_size,
            2*nb_bits_number)
    train_y_batch = train_y.reshape(
            len(train_y)/batch_size,
            batch_size,
            nb_bits_number)

    # We create the layers for our neural network
    input_var, train_fn, l_out = creating_model(nb_bits_number)

    NB_EPOCH = 7000
    for epoch in range(NB_EPOCH):
        loss = 0
        for i in range(len(train_x_batch)):
            input_batch = train_x_batch[i][0]
            target_batch = train_y_batch[i][0]
            train_fn_result = train_fn(np.array([input_batch]), target_batch)
            loss += train_fn_result
        if epoch % 5 == 0:
            print("Epoch %d: Loss %g" % (epoch + 1, loss / len(train_x_batch)))

            # Use trained network for predictions
            test_prediction = lasagne.layers.get_output(
                    l_out,
                    deterministic=True)
            predict_fn = theano.function([input_var], test_prediction)
            nb_errors = []
            for j in range(len(train_x)):
                pred_bin = 1*(0.5 < predict_fn([train_x[j]])[0])
                nb_errors.append(np.sum(np.absolute(train_y[j] - pred_bin)))
            print("Average nb errors: ", np.mean(nb_errors))


if __name__ == "__main__":
    main(batch_size=1,
         path_main_dataset_x="../data/features.txt",
         path_main_dataset_y="../data/targets.txt")
