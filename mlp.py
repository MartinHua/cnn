# -*- coding: utf-8 -*-
"""
借鉴自http://blog.csdn.net/u012162613/article/details/43221829
注释：
MLP多层感知机层与层之间是全连接的，以三层为例，最底层是输入层，中间是隐藏层，最后是输出层。
输入层没什么好说，你输入什么就是什么，比如输入是一个n维向量，就有n个神经元。
隐藏层的神经元怎么得来？它与输入层是全连接的，假设输入是X，则隐藏层的输出就是
f(WX+b),W是权重，b是偏置，f可以是常用的sigmoid函数或者tanh函数。
最后就是输出层，输出层与隐藏层是什么关系？其实隐藏层到输出层可以看成时一个多类别的逻辑回归，也即softmax，所以输出层的输出就是softmax(W1X1+b1)，X1表示隐藏层的输出。
MLP整个模型就是这样子的，它所有的参数就是各个层之间的连接权重以及偏置，包括W、b、W1、b1。对于一个具体的问题，怎么确定这些参数？那就是梯度下降法了（SGD），首先随机初始化所有参数，然后迭代地训练，不断地更新参数，直到满足某个条件为止（比如误差足够小、迭代次数足够多时）。
"""
__docformat__ = 'restructedtext en'

import os
import time
import numpy
import theano
import cPickle as pickle
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data, read_test
import sys

"""
注释：
这是定义隐藏层的类，首先明确：隐藏层的输入即input，输出即隐藏层的神经元个数。输入层与隐藏层是全连接的。
假设输入是n_in维的向量（也可以说时n_in个神经元），隐藏层有n_out个神经元，则因为是全连接，
一共有n_in*n_out个权重，故W大小时(n_in,n_out),n_in行n_out列，每一列对应隐藏层的每一个神经元的连接权重。
b是偏置，隐藏层有n_out个神经元，故b是n_out维向量。
rng即随机数生成器，numpy.random.RandomState，用于初始化W。
input训练模型所用到的所有输入，并不是MLP的输入层，MLP的输入层的神经元个数时n_in，而这里的参数input大小是（n_example,n_in）,每一行一个样本，即每一行作为MLP的输入层。
activation:激活函数,这里定义为函数tanh
"""
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, activation=T.tanh, rng=None, W=None, b=None):
         self.input = input
         """
         注释：
         代码要兼容GPU，则必须使用 dtype=theano.config.floatX,并且定义为theano.shared
         另外，W的初始化有个规则：如果使用tanh函数，则在-sqrt(6./(n_in+n_hidden))到sqrt(6./(n_in+n_hidden))之间均匀
         抽取数值来初始化W，若时sigmoid函数，则以上再乘4倍。
         """
         #如果W没有给定，即等于None，则根据上述的规则随机初始化。
         #加入这个判断的原因是：有时候我们可以用训练好的参数来初始化W，见我的上一篇文章。
         if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

         if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

         #用上面定义的W、b来初始化类HiddenLayer的W、b
         self.W = W
         self.b = b

        #隐含层的输出
         lin_output = T.dot(input, self.W) + self.b
         self.output = (
            lin_output if activation is None
            else activation(lin_output)
         )

        #隐含层的参数
         self.params = [self.W, self.b]

"""
定义分类层，Softmax回归
在deeplearning tutorial中，直接将LogisticRegression视为Softmax，
而我们所认识的二类别的逻辑回归就是当n_out=2时的LogisticRegression
"""
#参数说明：
#input，大小就是(n_example,n_in)，其中n_example是一个batch的大小，
#因为我们训练时用的是Minibatch SGD，因此input这样定义
#n_in,即上一层(隐含层)的输出
#n_out,输出的类别数

#3层的MLP
class MLP(object):
    def __init__(self, input, n_in, n_hidden, n_out, rng=None,
                 hiddenLayer_W=None, hiddenLayer_b=None,
                 logRegressionLayer_W=None, logRegressionLayer_b=None):
        self.input = input
        self.hiddenLayer = HiddenLayer(
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            W=hiddenLayer_W,
            b=hiddenLayer_b,
            activation=T.tanh,
            rng=rng
        )

        #将隐含层hiddenLayer的输出作为分类层logRegressionLayer的输入，这样就把它们连接了
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            W=logRegressionLayer_W,
            b=logRegressionLayer_b
        )

        #规则化项：常见的L1、L2_sqr
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        #损失函数Nll（也叫代价函数）
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        #误差
        self.errors = self.logRegressionLayer.errors

        #MLP的参数
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3


"""
加载数据集
"""

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=500, batch_size=20, n_hidden=300):
    """
    注释：
    learning_rate学习速率，梯度前的系数。
    L1_reg、L2_reg：正则化项前的系数，权衡正则化项与Nll项的比重
    代价函数=Nll+L1_reg*L1或者L2_reg*L2_sqr
    n_epochs：迭代的最大次数（即训练步数），用于结束优化过程
    dataset：训练数据的路径
    n_hidden:隐藏层神经元个数
    batch_size=20，即每训练完20个样本才计算梯度并更新参数
    """
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    #shape[0]获得行数，一行代表一个样本，故获取的是样本数，除以batch_size可以得到有多少个batch
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    #index表示batch的下标，标量
    #x表示数据集
    #y表示类别，一维向量
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)
    #生成一个MLP，命名为classifier
    classifier = MLP(
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        rng=rng
    )

    #代价函数，有规则化项
    #用y来初始化，而其实还有一个隐含的参数x在classifier中
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )


    #这里必须说明一下theano的function函数，givens是字典，其中的x、y是key，冒号后面是它们的value。
    #在function被调用时，x、y将被具体地替换为它们的value，而value里的参数index就是inputs=[index]这里给出。
    #下面举个例子：
    #比如test_model(1)，首先根据index=1具体化x为test_set_x[1 * batch_size: (1 + 1) * batch_size]，
    #具体化y为test_set_y[1 * batch_size: (1 + 1) * batch_size]。然后函数计算outputs=classifier.errors(y)，
    #这里面有参数y和隐含的x，所以就将givens里面具体化的x、y传递进去。
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    #cost函数对各个参数的偏导数值，即梯度，存于gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    #参数更新规则
    #updates[(),(),()....],每个括号里面都是(param, param - learning_rate * gparam)，即每个参数以及它的更新公式
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    patience = 10000
    patience_increase = 2
    #提高的阈值，在验证误差减小到之前的0.995倍时，会更新best_validation_loss
    improvement_threshold = 0.995
    #这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()

    #epoch即训练步数，每个epoch都会遍历所有训练数据
    epoch = 0
    done_looping = False

    #下面就是训练过程了，while循环控制的时步数epoch，一个epoch会遍历所有的batch，即所有的图片。
    #for循环是遍历一个个batch，一次一个batch地训练。for循环体里会用train_model(minibatch_index)去训练模型，
    #train_model里面的updatas会更新各个参数。
    #for循环里面会累加训练过的batch数iter，当iter是validation_frequency倍数时则会在验证集上测试，
    #如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，
    #则更新best_validation_loss和best_iter，同时在testset上测试。
    #如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。
    #当达到最大步数n_epoch时，或者patience<iter时，结束训练
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):#训练时一个batch一个batch进行的

            minibatch_avg_cost = train_model(minibatch_index)
            # 已训练过的minibatch数，即迭代次数iter
            iter = (epoch - 1) * n_train_batches + minibatch_index
            #训练过的minibatch数是validation_frequency倍数，则进行交叉验证
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                #当前验证误差比之前的都小，则更新best_validation_loss，以及对应的best_iter，并且在testdata上进行test
                if this_validation_loss < best_validation_loss:
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    fo = open('best_mlp_model.pkl', 'wb')
                    pickle.dump([[classifier.hiddenLayer.W, classifier.hiddenLayer.b],
                                 [classifier.logRegressionLayer.W, classifier.logRegressionLayer.b]],fo)
                    fo.close()
            #patience小于等于iter，则终止训练
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i') %
          (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

def predict(testData):
    test_set_x = read_test(testData)
    hiddenLayer, logRegressionLayer = pickle.load(open('best_mlp_model.pkl'))
    hiddenLayer_W, hiddenLayer_b = hiddenLayer
    logRegressionLayer_W, logRegressionLayer_b = logRegressionLayer
    x = T.matrix('x')
    classifier = MLP(
        input=x,
        n_in=28 * 28,
        n_hidden=300,
        n_out=10,
        hiddenLayer_W=hiddenLayer_W,
        hiddenLayer_b=hiddenLayer_b,
        logRegressionLayer_W=logRegressionLayer_W,
        logRegressionLayer_b=logRegressionLayer_b
    )
    predict_model = theano.function(inputs=[classifier.input], outputs=classifier.logRegressionLayer.y_pred)
    predicted_values = predict_model(test_set_x.get_value())
    return predicted_values

if __name__ == '__main__':
    test_mlp()