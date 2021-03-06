# coding:utf-8
import numpy as np
import pandas as pd
import os
import wget

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
    # Define
    np.random.seed(10)

    # Layer 1
    l1_in = 4
    l1_out = 100
    l1_w = np.random.rand(l1_out,l1_in)
    l1_h = np.vectorize(lambda x: max(0,x)) # ReLU
    l1_h_d = np.vectorize(lambda x: 1 if x > 0 else 0) # d ReLU

    # Layer 2
    l2_in = 100
    l2_out = 100
    l2_w = np.random.rand(l2_out,l2_in)
    l2_h = np.vectorize(lambda x: max(0,x)) # ReLU
    l2_h_d = np.vectorize(lambda x: 1 if x > 0 else 0) # d ReLU

    # Layer 3
    l3_in = 100
    l3_out = 3
    l3_w = np.random.rand(l3_out,l3_in)
    l3_h = np.vectorize(lambda x: x)
    l3_h_d = np.vectorize(lambda x: 1)

    # Loss Function
    softmax = lambda x: np.exp(x-max(x)) / sum(np.exp(x-max(x))) # softmax
    lf = lambda y,t: -t * np.log(y+1.0e-6) # multi cross entropy
    lf_d = np.vectorize(lambda y,t: t - y) # softmax cross entropy

    # Optimizer(SGD)
    alpha = 0.01
    lamb = 0.001

    # Get data
    iris = iris_dataset()
    iris = iris.reindex(np.random.permutation(iris.index))

    # Make Batch (150 datas)
    iris_leng = len(iris)
    iris_train_d = np.array(iris.iloc[0:150, 0:4])
    iris_train_t = np.array(iris.iloc[0:150, 4:7])
    #iris_test_d = np.array(iris.iloc[100:150, 0:4])
    #iris_test_t = np.array(iris.iloc[100:150, 4:7])

    # Learning Loop
    epoch = 30
    for e in range(epoch):
        batch_size = 100
        for start in range(0,len(iris_train_d),batch_size):
            in_data_matrix = iris_train_d[start:start+batch_size]
            in_teacher_matrix = iris_train_t[start:start+batch_size]

            # Reset Grads
            l1_dw = np.zeros((l1_w.shape))
            l2_dw = np.zeros((l2_w.shape))
            l3_dw = np.zeros((l3_w.shape))
            for in_vector, teacher_vector in zip(in_data_matrix, in_teacher_matrix):
                # Learning
                l1 = l1_h(l1_w.dot(in_vector))
                l2 = l2_h(l2_w.dot(l1))
                l3 = l3_w.dot(l2)

                # Calc Loss
                #print(l3)
                #print(softmax(l3))
                #input()
                E = lf(softmax(l3), teacher_vector)

                # Back Propagation
                l3_delta = l3_h_d(l3)*lf_d(softmax(l3), teacher_vector)
                l2_delta = l2_h_d(l2)*l3_w.T.dot(l3_delta)
                l1_delta = l1_h_d(l1)*l2_w.T.dot(l2_delta)

                l1_dw += in_vector[:,np.newaxis].dot(l1_delta[np.newaxis,:]).T/batch_size
                l2_dw += l1[:,np.newaxis].dot(l2_delta[np.newaxis,:]).T/batch_size
                l3_dw += l2[:,np.newaxis].dot(l3_delta[np.newaxis,:]).T/batch_size

            # Optimization(SGD)
            l1_w = l1_w - (alpha*l1_dw) - (alpha * lamb * l1_w)
            l2_w = l2_w - (alpha*l2_dw) - (alpha * lamb * l2_w)
            l3_w = l3_w - (alpha*l3_dw) - (alpha * lamb * l3_w)
        print(E)

if __name__ == "__main__":
    main()
