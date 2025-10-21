import numpy as np
import math


def fun_info(F):
    if F == 'F1':
        fitness = F1
        lowerbound = -100
        upperbound = 100
        dimension = 30

    elif F == 'F2':
        fitness = F2
        lowerbound = -10
        upperbound = 10
        dimension = 30

    elif F == 'F3':
        fitness = F3
        lowerbound = -100
        upperbound = 100
        dimension = 30

    elif F == 'F4':
        fitness = F4
        lowerbound = -100
        upperbound = 100
        dimension = 30

    elif F == 'F5':
        fitness = F5
        lowerbound = -30
        upperbound = 30
        dimension = 30

    elif F == 'F6':
        fitness = F6
        lowerbound = -100
        upperbound = 100
        dimension = 30

    elif F == 'F7':
        fitness = F7
        lowerbound = -1.28
        upperbound = 1.28
        dimension = 30

    elif F == 'F8':
        fitness = F8
        lowerbound = -500
        upperbound = 500
        dimension = 30

    elif F == 'F9':
        fitness = F9
        lowerbound = -5.12
        upperbound = 5.12
        dimension = 30

    elif F == 'F10':
        fitness = F10
        lowerbound = -32
        upperbound = 32
        dimension = 30

    elif F == 'F11':
        fitness = F11
        lowerbound = -600
        upperbound = 600
        dimension = 30

    elif F == 'F12':
        fitness = F12
        lowerbound = -50
        upperbound = 50
        dimension = 30

    elif F == 'F13':
        fitness = F13
        lowerbound = -50
        upperbound = 50
        dimension = 30

    elif F == 'F14':
        fitness = F14
        lowerbound = -65.536
        upperbound = 65.536
        dimension = 2

    elif F == 'F15':
        fitness = F15
        lowerbound = -5
        upperbound = 5
        dimension = 4

    elif F == 'F16':
        fitness = F16
        lowerbound = -5
        upperbound = 5
        dimension = 2

    elif F == 'F17':
        fitness = F17
        lowerbound = np.array([-5, 0])
        upperbound = np.array([10, 15])
        dimension = 2

    elif F == 'F18':
        fitness = F18
        lowerbound = -2
        upperbound = 2
        dimension = 2

    elif F == 'F19':
        fitness = F19
        lowerbound = 0
        upperbound = 1
        dimension = 3

    elif F == 'F20':
        fitness = F20
        lowerbound = 0
        upperbound = 1
        dimension = 6

    elif F == 'F21':
        fitness = F21
        lowerbound = 0
        upperbound = 10
        dimension = 4

    elif F == 'F22':
        fitness = F22
        lowerbound = 0
        upperbound = 10
        dimension = 4

    elif F == 'F23':
        fitness = F23
        lowerbound = 0
        upperbound = 10
        dimension = 4

    return lowerbound, upperbound, dimension, fitness


# === Benchmark Functions ===

def F1(x):
    return np.sum(x ** 2)


def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def F3(x):
    R = 0
    for i in range(len(x)):
        R += np.sum(x[:i + 1]) ** 2
    return R


def F4(x):
    return np.max(np.abs(x))


def F5(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def F6(x):
    return np.sum(np.floor(x + 0.5) ** 2)


def F7(x):
    n = len(x)
    return np.sum(np.arange(1, n + 1) * (x ** 4)) + np.random.rand()


def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    n = len(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * n


def F10(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e


def F11(x):
    n = len(x)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1)))) + 1


def F12(x):
    n = len(x)
    term1 = (np.pi / n) * (10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2)
    term2 = np.sum(((x[:-1] + 1) / 4) ** 2 * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4))) ** 2))
    term3 = ((x[-1] + 1) / 4) ** 2
    return term1 + term2 + term3 + np.sum(Ufun(x, 10, 100, 4))


def F13(x):
    n = len(x)
    term1 = (np.sin(3 * np.pi * x[0])) ** 2
    term2 = np.sum((x[:-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:])) ** 2))
    term3 = ((x[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[-1])) ** 2)
    return 0.1 * (term1 + term2 + term3) + np.sum(Ufun(x, 5, 100, 4))


def F14(x):
    aS = np.array([
        [-32, -16, 0, 16, 32] * 5,
        [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5
    ])
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j]) ** 6)
    return (1 / (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS))))


def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844,
                   0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - ((x[0] * (bK ** 2 + x[1] * bK)) / (bK ** 2 + x[2] * bK + x[3]))) ** 2)


def F16(x):
    return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)


def F17(x):
    return (x[1] - (x[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
                1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10


def F18(x):
    return ((1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) *
            (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                        18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)))


def F19(x):
    aH = np.array([[3, 10, 30],
                   [0.1, 10, 35],
                   [3, 10, 30],
                   [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.1170, 0.2673],
                   [0.4699, 0.4387, 0.7470],
                   [0.1091, 0.8732, 0.5547],
                   [0.03815, 0.5743, 0.8828]])
    R = 0
    for i in range(4):
        R -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :]) ** 2)))
    return R


def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    R = 0
    for i in range(4):
        R -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :]) ** 2)))
    return R


def F21(x):
    aSH = np.array([
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    R = 0
    for i in range(5):
        R -= 1 / ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i]))
    return R


def F22(x):
    aSH = np.array([
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    R = 0
    for i in range(7):
        R -= 1 / ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i]))
    return R


def F23(x):
    aSH = np.array([
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    R = 0
    for i in range(10):
        R -= 1 / ((np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i]))
    return R


# === Helper Function ===

def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)
