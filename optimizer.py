# coding:utf-8
import numpy as np

def main():
    np.random.seed(10)

    in_vector = np.array([0.3, 0.5, 0.7, 0.2])
    teacher_vector = np.array([0.0, 0.0, 1.0])

    # Layer 1
    l1_w = np.random.rand(3,4)
    l1_h = np.vectorize(lambda x: max(0,x)) # ReLU

    # Layer 2
    l2_w = np.random.rand(3,3)
    l2_h = np.vectorize(lambda x: max(0,x)) # ReLU

    # Layer 3
    l3_w = np.random.rand(3,3)
    l3_h = np.vectorize(lambda x: x) # identity

    l1 = l1_h(l1_w.dot(in_vector))
    l2 = l2_h(l2_w.dot(l1))
    l3 = l3_h(l3_w.dot(l2))
    print("Layer3 output: " + str(l3))


    # Loss Function
    lf = lambda y,t: -t * np.log(y) # multi cross entropy

    # Calc Loss
    E = lf(l3, teacher_vector)

    # Back Propagation
    E_d = np.vectorize(lambda y,t: t/y) # d Cross entropy
    l3_h_d = np.vectorize(lambda x: 1) # d ReLU
    l3_delta = l3_h_d(l3)*E_d(l3, teacher_vector)

    l2_h_d = np.vectorize(lambda x: 1 if x > 0 else 0) # d ReLU
    l2_delta = l2_h_d(l2)*l3_w.T.dot(l3_delta)

    l1_h_d = np.vectorize(lambda x: 1 if x > 0 else 0) # d ReLU
    l1_delta = l1_h_d(l1)*l2_w.T.dot(l2_delta)

    # Optimization(SGD)
    l1_w = l1_w - alpha*in_vector[:,np.newaxis].dot(l1_delta[np.newaxis,:]) - alpha*lamd*l1_w
    l2_w = l2_w - alpha*l1[:,np.newaxis].dot(l2_delta[np.newaxis,:]) - alpha*lamb*l2_w
    l3_w = l3_w - alpha*l2[:,np.newaxis].dot(l3_delta[np.newaxis,:]) - alpha*lamb*l3_w

    

if __name__ == "__main__":
    main()
