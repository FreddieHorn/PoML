import numpy as np
import numpy.linalg as la
import itertools


def lstsq_and_eval(X, y):
    w, _, _, _  = la.lstsq(X, y, rcond=None)
    return X @ w


def phi(x):
    n = len(x)
    # powerset of x
    combis = [itertools.combinations(x, i) for i in range(1, n+1)]
    # put those sets into one list, at the beginning is 1
    phi = [(1)] +  list(itertools.chain.from_iterable(combis))
    len_phi = len(phi)
    # iterate through sets and multiply the elements in each list
    phi = [np.prod(phi[i]) for i in range(len_phi)]
    return phi


def task1_2():
    # define matX
    matX = np.array([[1,1,1],
              [1,1,-1],
              [1,-1,1],
              [1,-1,-1],
              [-1,1,1],
              [-1,1,-1],
              [-1,-1,1],
              [-1,-1,-1]])
    # define the rules
    y_rule_110 = np.array([1, -1, -1, -1, 1, -1, -1, 1])
    y_rule_126 = np.array([1, -1, -1, -1, -1, -1, -1, 1])

    # task 1.2.1
    print("Task 1.2.1")
    y = lstsq_and_eval(matX, y_rule_110)
    y_hat = lstsq_and_eval(matX, y_rule_126)

    print(y) # = [0.25, -0.25, -0.25, -0.75, 0.75, 0.25, 0.25, -0.25]
    print(y_hat) # ~ [0, 0, 0, 0, 0, 0, 0, 0]

    # task 1.2.2
    # see implementation of "phi()"

    # task 1.2.3
    print("task 1.2.3")
    x_vectors = [matX[i, :] for i in range(8)]
    phi_vectors = [phi(x) for x in x_vectors]
    Phi_T = np.vstack(phi_vectors)

    y_phi = lstsq_and_eval(Phi_T, y_rule_110)
    y_hat_phi = lstsq_and_eval(Phi_T, y_rule_126)

    print(y_phi) # = y_rule_110
    print(y_hat_phi) # = y_rule_126


if __name__ == "__main__":
    task1_2()
