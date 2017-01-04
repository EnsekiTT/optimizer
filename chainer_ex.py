# coding:utf-8
import numpy as np
import pandas as pd
import os
import wget
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class MLP(chainer.Chain):

    def __init__(self, in_units, mid_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(None, in_units),
            l2=L.Linear(None, mid_units),
            l3=L.Linear(None, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))

def iris_dataset():
    if not os.path.exists('iris.data'):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        wget.download(url)
        print("Download is done.")

    if not os.path.exists('iris.csv'):
        f = open('iris.data')
        iris_datas = f.readlines()
        f.close()
        wf = open('iris.csv',"w")
        for iris in iris_datas:
            iris = iris.replace('Iris-setosa','1.0,0.0,0.0')
            iris = iris.replace('Iris-versicolor','0.0,1.0,0.0')
            iris = iris.replace('Iris-virginica','0.0,0.0,1.0')
            wf.write(iris)
    else:
        data = pd.read_csv('iris.csv', names=[0,1,2,3,4,5,6])
    return data

def main():
    model = L.Classifier(MLP(10, 10, 3))

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    print(train[0])
    print(test[1][1])
    exit()

    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == "__main__":
    main()
