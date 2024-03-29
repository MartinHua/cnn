# -*- coding: utf-8 -*-
"""
借鉴自http://deeplearning.net/tutorial/code/convolutional_mlp.py
"""

from __future__ import print_function

import numpy
import cPickle as pickle
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from logistic_sgd import LogisticRegression, load_data, read_test
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, filter_shape, image_shape, poolsize=(2, 2), rng=None, W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))

        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        pooled_out = pool.pool_2d(
            input=conv_out,
            ws=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=200, nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]

    n_train_batches //= batch_size
    n_valid_batches //= batch_size

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng=rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=300,
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=300, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss *  improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    fo = open('best_cnn_model.pkl', 'wb')
                    pickle.dump([[layer0.W, layer0.b],
                                 [layer1.W, layer1.b],
                                 [layer2.W, layer2.b],
                                 [layer3.W, layer3.b]],fo)
                    fo.close()
            if patience <= iter:
                done_looping = True
                break

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, ' %
          (best_validation_loss * 100., best_iter + 1))
def predict(testData):
    test_set_x = read_test(testData)
    param = pickle.load(open('best_cnn_model.pkl', 'rb'))
    layer0_W, layer0_b = param[0]
    layer1_W, layer1_b = param[1]
    layer2_W, layer2_b = param[2]
    layer3_W, layer3_b = param[3]
    nkerns = [20, 50]
    x = T.matrix('x')
    size = test_set_x.get_value(borrow=True).shape[0]
    layer0_input = x.reshape((size, 1, 28, 28))
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        image_shape=(size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2),
        W=layer0_W,
        b=layer0_b
    )
    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        image_shape=(size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2),
        W=layer1_W,
        b=layer1_b
    )
    layer2 = HiddenLayer(
        input=layer1.output.flatten(2),
        n_in=nkerns[1] * 4 * 4,
        n_out=300,
        activation=T.tanh,
        W=layer2_W,
        b=layer2_b
    )
    layer3 = LogisticRegression(input=layer2.output, n_in=300, n_out=10, W=layer3_W, b=layer3_b)
    predict_model = theano.function(inputs=[layer0.input], outputs=layer3.y_pred)
    predicted_values = predict_model(test_set_x.get_value().reshape((size, 1, 28, 28)))
    return predicted_values

if __name__ == '__main__':
    evaluate_lenet5()