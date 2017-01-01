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
    in_vector = np.array([0.3, 0.5, 0.7, 0.2])
    teacher_vector = np.array([0.0, 0.0, 1.0])

    # Layer 1
    l1_w = np.random.rand(3,4)
    l1_h = np.vectorize(lambda x: max(0,x)) # ReLU
    l1_h_d = np.vectorize(lambda x: 1 if x > 0 else 0) # d ReLU

    # Layer 2
    l2_w = np.random.rand(3,3)
    l2_h = np.vectorize(lambda x: max(0,x)) # ReLU
    l2_h_d = np.vectorize(lambda x: 1 if x > 0 else 0) # d ReLU

    # Layer 3
    l3_w = np.random.rand(3,3)
    l3_h = np.vectorize(lambda x: x) # identity
    l3_h_d = np.vectorize(lambda x: 1) # d ReLU

    # Loss Function
    lf = lambda y,t: -t * np.log(y+0.0000001) # multi cross entropy
    lf_d = np.vectorize(lambda y,t: t/(y+0.0000001)) # d Cross entropy

    # Optimizer(SGD)
    alpha = 0.01

    # get_data
    iris = iris_dataset()
    iris.reindex(np.random.permutation(iris.index))
    iris_data = np.array(iris.iloc[:,0:4])
    iris_teacher = np.array(iris.iloc[:,4:7])
    for in_vector, teacher_vector in zip(iris_data, iris_teacher):
        # Learning
        l1 = l1_h(l1_w.dot(in_vector))
        l2 = l2_h(l2_w.dot(l1))
        l3 = l3_h(l3_w.dot(l2))

        # Calc Loss
        E = lf(l3, teacher_vector)
        print(l3)
        print(teacher_vector)
        print("Error:"+str(E))
        input(",")

        # Back Propagation
        l3_delta = l3_h_d(l3)*lf_d(l3, teacher_vector)
        l2_delta = l2_h_d(l2)*l3_w.T.dot(l3_delta)
        l1_delta = l1_h_d(l1)*l2_w.T.dot(l2_delta)

        # Optimization(SGD)
        l1_w = l1_w - alpha*in_vector[:,np.newaxis].dot(l1_delta[np.newaxis,:]).T
        l2_w = l2_w - alpha*l1[:,np.newaxis].dot(l2_delta[np.newaxis,:]).T
        l3_w = l3_w - alpha*l2[:,np.newaxis].dot(l3_delta[np.newaxis,:]).T


if __name__ == "__main__":
    main()
