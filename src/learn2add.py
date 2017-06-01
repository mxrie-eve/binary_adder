from __future__ import print_function
import numpy as np
import theano
import lasagne
import theano.tensor as T
import matplotlib.pyplot as plt

def data_splitter(array_data, array_ratios):
    size = array_data.shape
    array_returned = []

    for i in range(len(array_ratios) - 1):
        nb_datas = int(math.floor(array_ratios[i]*size[0]))
        array_returned.append(array_data[:nb_datas])
        array_data = np.delete(array_data, np.arange(nb_datas), 0)

    return array_returned.append(array_data)


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

    return input_var, train_fn, l_out, l_hidden


def get_sets(path_main_dataset_x, path_main_dataset_y):

    X = np.genfromtxt(path_main_dataset_x, delimiter=1)
    Y = np.genfromtxt(path_main_dataset_y,  delimiter=1)

    return [X, Y, [], [], [], []]


def main(nb_epoch, path_main_dataset_x, path_main_dataset_y, batch_size=1):
    # Increasing the batch_size will help stabilize the gradient.  I've already
    # tried it without much success.
    batch_size = 1
    x_value = []
    y_value = []

    train_x, train_y, val_x, val_y, test_x, test_y = get_sets(
            path_main_dataset_x,
            path_main_dataset_y)

    val_x, val_y = train_x, train_y

    # This is the number of bits of one of the number we are adding
    nb_bits_number = train_x.shape[1] / 2

    # Converting the data into mini-batches. For now the size of the
    # mini-batches must divide the size of the training dataset.
    train_x_batch = train_x.reshape(
            len(train_x)/batch_size,
            batch_size,
            2*nb_bits_number)
    train_y_batch = train_y.reshape(
            len(train_y)/batch_size,
            batch_size,
            nb_bits_number)

    # We create the layers for our neural network
    input_var, train_fn, l_out, l_hidden = creating_model(nb_bits_number)

    plt.ion()
    plt.figure(1)
    for epoch in range(nb_epoch):
        loss = 0

        # Trainin the model on all mini-batches
        for i in range(len(train_x_batch)):
            input_batch = train_x_batch[i][0]
            target_batch = train_y_batch[i][0]
            train_fn_result = train_fn(np.array([input_batch]), target_batch)
            loss += train_fn_result

        if epoch % 1 == 0:
            # Every n epochs we will compute the error on the training and
            # valisation datasets
            print("Epoch %d: Loss %g" % (epoch, loss / len(train_x_batch)))

            # Use trained network for predictions
            test_prediction = lasagne.layers.get_output(
                    l_out,
                    deterministic=True)
	    
            print(test_prediction)
	
            # We define a theano function that takes one inputs and returns the
            # precition for that input
            predict_fn = theano.function([input_var], test_prediction)
            nb_errors = []
            for j in range(len(train_x)):
                pred_bin = 1*(0.5 < predict_fn([train_x[j]])[0])
                nb_errors.append(np.sum(np.absolute(train_y[j] - pred_bin)))

            print("Error on training set: %1.4f/%d ",
                    (np.mean(nb_errors), nb_bits_number))
            # print("Error on validation set: %1.4f/%d ",
            #         (np.mean(nb_errors), nb_bits_number)

            # Plot mean error and save model every n iterations.
            x_value.append(epoch)
            y_value.append(np.mean(nb_errors))
	    plt.scatter(epoch,np.mean(nb_errors),color='darkred')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Error')
            plt.title('Error as a function of the Epoch')
            plt.pause(0.0001)
            plt.draw()

	    model_saved = lasagne.layers.get_all_param_values(l_out)
	    #np.savetxt('ModelSave.txt', model_saved)

	    # Then we need to use the set_all_param_values to load the model
            print(model_saved)	

if __name__ == "__main__":
    main(batch_size=1,
         path_main_dataset_x="../data/features.txt",
         path_main_dataset_y="../data/targets.txt",
         nb_epoch=3)
