"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import timeit

import numpy as np
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from hidden_layer import HiddenLayer
from dropout_layer import DropoutLayer


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    # def __init__(self, rng, input, n_in, n_hidden, n_out, dropout):
    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        if not dropout:
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
            )
        else:
            self.hiddenLayer = DropoutLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
            )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input

        # This block only gets computed if the alternative cost function is use. Clever compiler...
        minibatch_size = self.hiddenLayer.output.shape[0]
        mean_activation = self.hiddenLayer.output.mean(0)
        centered_activation = self.hiddenLayer.output - mean_activation # casts over rows
        activation_covariance = 1.0/(minibatch_size-1) * centered_activation.T.dot(centered_activation)
        std_vec = (1.0/(minibatch_size-1) * (centered_activation**2).sum(0))**(0.5)
        self.activation_correlation = (activation_covariance / std_vec).T / std_vec # works because matrix is symmetric
        self.cor_sqr_sum = (self.activation_correlation**2).sum() # square element-wise and sum

def flatten_correlation_matrix(correlation_matrix):
    '''
    Accepts a matrix as input, and returns a 1D array containing its lower-triangular entries
    '''
    return np.array([correlation_matrix[i,j] for i in range(correlation_matrix.shape[0]) for j in range(correlation_matrix.shape[1]) if i>j])

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, cor_reg=0.00, cor_scaling=1.0, rand_seed=1234, dropout=False,
            n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, save_correlations=False):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    cor_reg_var = theano.shared(cor_reg) # symbolic variable storing cor_reg value
    alpha = T.dscalar('alpha') # scaling factor for weight decay
    is_train = T.bscalar('is_train') # boolean for switching between training and prediction
    index = T.lscalar()  # index to a [mini]batch
    perm = T.lvector()  # permutation of the indices of the training samples
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = np.random.RandomState(rand_seed)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        dropout=dropout
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    if cor_reg == 0:
        cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
        )
    else:
        cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
            + cor_reg_var * classifier.cor_sqr_sum
        )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    if save_correlations:
        validate_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors(y), classifier.activation_correlation],
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size],
                is_train: np.cast['int8'](0)
            }
        )
    else:
        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size],
                is_train: np.cast['int8'](0)
            }
        )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index,perm],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[perm[index * batch_size: (index + 1) * batch_size]],
            y: train_set_y[perm[index * batch_size: (index + 1) * batch_size]],
            is_train: np.cast['int8'](1)
        }
    )
    # end-snippet-5

    # update the symbolic cor_reg variable
    update_cor_reg = theano.function(
        inputs=[alpha],
        outputs=cor_reg_var,
        updates=[(cor_reg_var, cor_reg_var*alpha)]
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_validation_loss = np.inf
    best_epoch = 0
    start_time = timeit.default_timer()

    # Open file for writing validation losses, and write the header
    valid_loss_filename = 'ValidationLoss_Epoch%i_Batch%i_Cor%f_Drop%i_Scale%f.csv' % (n_epochs, n_epochs*n_train_batches, cor_reg, dropout, cor_scaling)
    valid_loss_filepath = os.path.join(os.path.split(__file__)[0], '..', 'output', 'MLP', valid_loss_filename)
    valid_loss_outfile = open(valid_loss_filepath, 'w')
    valid_loss_outfile.write('Epoch,Iteration,Error\n')
    if save_correlations:
        flat_corr_filename = 'FlatCorrelations_Epoch%i_Batch%i_Cor%f_Drop%i_Scale%f.csv' % (n_epochs, n_epochs*n_train_batches, cor_reg, dropout, cor_scaling)
        flat_corr_filepath = os.path.join(os.path.split(__file__)[0], '..', 'output', 'MLP', flat_corr_filename)
        flat_corr_outfile = open(flat_corr_filepath, 'w')

    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        index_perm = rng.permutation(train_set_x.get_value(borrow=True).shape[0])  # generate new permutation of indices

        # perform 1 epoch of training
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, index_perm)

        print "Hidden layer after training:\n"
        print classifier.hiddenLayer.output.get_value()

        # compute zero-one loss on validation set
        if save_correlations: # compute and save the average pairwise correlations
            validation_losses = []
            mean_correlations = 0 # contains mean correlation matrix once loop is finished
            for i in xrange(n_valid_batches):
                valid_loss, valid_corr = validate_model(i)
                validation_losses.append(valid_loss)
                mean_correlations += (1.0 * valid_corr / n_valid_batches) # iteratively constructs mean to save memory
            this_validation_loss = np.mean(validation_losses)
            flat_mean_correlation = flatten_correlation_matrix(mean_correlations)
            flat_corr_outfile.write(str(epoch)+','+','.join(map(str,flat_mean_correlation))+'\n')
        else:
            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

        # Write this epoch's validation error to the file
        valid_loss_outfile.write(('%i,%i,%f\n') % (epoch, epoch*n_train_batches, this_validation_loss))

        # ********COMMENT THIS OUT WHEN RUNNING MULTIPLE PARAMS OVERNIGHT********
        print('epoch %i (iteration %i), validation error %f %%, cor_reg %f' % (epoch, epoch*n_train_batches, this_validation_loss * 100., cor_reg_var.get_value()))
        # current_time = timeit.default_timer()
        # print('epoch %i (iteration %i), validation error %f %%, cor_reg %f, time elapsed %.2fm' % (epoch, epoch*n_train_batches, this_validation_loss * 100., cor_reg_var.get_value(), (current_time - start_time) / 60.))
        print "Hidden layer after validation:\n"
        print classifier.hiddenLayer.output.get_value()

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            best_epoch = epoch

        # Update the value of cor_reg for the next epoch
        # Only makes a difference when cor_scaling != 1, because multiplication
        if cor_scaling != 1:
            old_cor_reg = update_cor_reg(cor_scaling)

    valid_loss_outfile.close()
    if save_correlations:
        flat_corr_outfile.close()

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained following epoch %i (iteration %i)') %
          (best_validation_loss * 100., best_epoch, best_epoch*n_train_batches))

    print "Training process ran for %.2fm" % ((end_time - start_time) / 60.)


if __name__ == '__main__':

    # # Vanilla MLP
    # # test_mlp(n_epochs=300, batch_size=20, save_correlations=True)
    # # MLP with fixed cor_reg = 1e-5
    # test_mlp(n_epochs=300, batch_size=20, save_correlations=True, cor_reg=1e-5)
    # # MLP with decaying cor_reg s.t. cor_reg~1e-5 after 100 epochs
    # test_mlp(n_epochs=300, batch_size=20, save_correlations=True, cor_reg=1e-3, cor_scaling=0.95)

    test_mlp(n_epochs=10, batch_size=5, n_hidden=10)

    # cor_list = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 0.0]
    # for this_cor in cor_list:
    #     test_mlp(cor_reg=this_cor, n_epochs=100, batch_size=20, save_correlations=True)
