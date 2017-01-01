# coding:utf-8
import numpy as np

def main():
    np.random.seed(10)

    in_vector = np.array([0.3, 0.5, 0.7, 0.2])
    out_vector = np.array([0.0, 0.0, 0.0])
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
    out_vector = l3_h(l3_w.dot(l2))
    print("Layer3 output: " + str(out_vector))


    # Loss Function
    lf = lambda y,t: -t * np.log(y) # multi cross entropy




if __name__ == "__main__":
    main()
