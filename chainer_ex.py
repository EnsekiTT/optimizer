# coding:utf-8
import numpy as np
import pandas as pd
import os
import wget
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class MLP(chainer.Chain):
    def __init__(self, in_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(None, in_units),
            l2=L.Linear(None, in_units),
            l3=L.Linear(None, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


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
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    #model = L.Classifier(MLP(1000, 1000, 3))
    model = L.Classifier(MLP(args.unit, 3))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu() # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the dataset
    iris = iris_dataset()
    iris = iris.reindex(np.random.permutation(iris.index))
    iris_d = np.array(iris.iloc[0:150, 0:4], dtype=np.float32)
    iris_t = np.array(iris.iloc[0:150, 4:7])
    label = np.zeros(len(iris_t))
    for i, t in enumerate(iris_t):
        if t[0] == 1.0:
            label[i] = 0
        if t[1] == 1.0:
            label[i] = 1
        if t[2] == 1.0:
            label[i] = 2
    label = np.array(label, dtype=np.int32)

    train = tuple_dataset.TupleDataset(iris_d[0:100], label[0:100])
    test = tuple_dataset.TupleDataset(iris_d[100:150], label[100:150])

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
        repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == "__main__":
    main()
