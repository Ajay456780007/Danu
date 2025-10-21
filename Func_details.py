import numpy as np


def func_details(fn):
    def ufun(x, a, k, m):
        return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))

    if fn == 'func1':
        fobj = func1
        lb = -100
        ub = 100
        dim = 30

    elif fn == 'func2':
        fobj = func2
        lb = -10
        ub = 10
        dim = 30

    elif fn == 'func3':
        fobj = func3
        lb = -100
        ub = 100
        dim = 30

    elif fn == 'func4':
        fobj = func4
        lb = -100
        ub = 100
        dim = 30

    elif fn == 'func5':
        fobj = func5
        lb = -30
        ub = 30
        dim = 30

    elif fn == 'func6':
        fobj = func6
        lb = -100
        ub = 100
        dim = 30

    elif fn == 'func7':
        fobj = func7
        lb = -1.28
        ub = 1.28
        dim = 30

    elif fn == 'func8':
        fobj = func8
        lb = -500
        ub = 500
        dim = 30

    elif fn == 'func9':
        fobj = func9
        lb = -32
        ub = 32
        dim = 30

    elif fn == 'func10':
        fobj = func10
        lb = -50
        ub = 50
        dim = 30

    elif fn == 'func11':
        fobj = func11
        lb = -50
        ub = 50
        dim = 30

    elif fn == 'func12':
        fobj = func12
        lb = -65.536
        ub = 65.536
        dim = 2

    elif fn == 'func13':
        fobj = func13
        lb = -5
        ub = 5
        dim = 4

    elif fn == 'func14':
        fobj = func14
        lb = -2
        ub = 2
        dim = 2

    elif fn == 'func15':
        fobj = func15
        lb = 0
        ub = 1
        dim = 3

    elif fn == 'func16':
        fobj = func16
        lb = 0
        ub = 1
        dim = 6

    elif fn == 'func17':
        fobj = func17
        lb = 0
        ub = 10
        dim = 4

    elif fn == 'func18':
        fobj = func18
        lb = 0
        ub = 10
        dim = 4

    elif fn == 'func19':
        fobj = func19
        lb = 0
        ub = 10
        dim = 4

    else:
        fobj, lb, ub, dim = None, None, None, None

    return lb, ub, dim, fobj


# Function implementations

def func1(x):
    return np.sum(x ** 2)


def func2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def func3(x):
    op = 0
    for i in range(1, len(x) + 1):
        op += np.sum(x[:i]) ** 2
    return op


def func4(x):
    return np.max(np.abs(x))


def func5(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def func6(x):
    return np.sum(np.abs(x + 0.5) ** 2)


def func7(x):
    dim = len(x)
    return np.sum(np.arange(1, dim + 1) * (x ** 4)) + np.random.rand()


def func8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def func9(x):
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim)) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)


def func10(x):
    dim = len(x)

    def ufun(x, a, k, m):
        return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))

    val1 = 10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2)
    val2 = np.sum((((x[:-1] + 1) / 4) ** 2) * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2)))
    val3 = ((x[-1] + 1) / 4) ** 2
    val4 = np.sum(ufun(x, 10, 100, 4))
    return (np.pi / dim) * (val1 + val2 + val3) + val4


def func11(x):
    dim = len(x)

    def ufun(x, a, k, m):
        return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))

    val1 = 0.1 * (np.sin(3 * np.pi * x[0]) ** 2)
    val2 = np.sum((x[:-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:]) ** 2)))
    val3 = ((x[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[-1]) ** 2))
    val4 = np.sum(ufun(x, 5, 100, 4))
    return val1 + val2 + val3 + val4


def func12(x):
    aS = np.array(
        [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
         [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j]) ** 6)
    return (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS))) ** (-1)


def func13(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - (x[0] * (bK ** 2 + x[1] * bK)) / (bK ** 2 + x[2] * bK + x[3])) ** 2)


def func14(x):
    return (1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) * \
        (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                    18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2)))


def func15(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array(
        [[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    op = 0
    for i in range(4):
        op -= cH[i] * np.exp(-np.sum(aH[i, :] * (x - pH[i, :]) ** 2))
    return op


def func16(x):
    aH = np.array(
        [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.665],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    op = 0
    for i in range(4):
        op -= cH[i] * np.exp(-np.sum(aH[i, :] * (x - pH[i, :]) ** 2))
    return op


def func17(x):
    aSH = np.array(
        [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
         [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    op = 0
    for i in range(5):
        op -= ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i]) ** -1)
    return op


def func18(x):
    aSH = np.array(
        [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
         [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    op = 0
    for i in range(7):
        op -= ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i]) ** -1)
    return op


def func19(x):
    aSH = np.array(
        [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
         [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    op = 0
    for i in range(10):
        op -= ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i]) ** -1)
    return op
