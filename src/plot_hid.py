import matplotlib.pyplot as plt

if __name__ == '__main__':

    fname = "record/pos.txt"
    AverageEpRet1 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet1.append(float(line.split('\t')[1]))

    fname = "record/pos128.txt"
    AverageEpRet2 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet2.append(float(line.split('\t')[1]))

    plt.title('Peg Insert -- compare hidden layer', size=14)
    plt.xlabel('epochs', size=12)
    plt.ylabel('AverageEpRet', size=12)
    plt.plot(range(len(AverageEpRet1)), AverageEpRet1, color='b', marker='.', label='64-64') 
    plt.plot(range(len(AverageEpRet2)), AverageEpRet2, color='r', marker='.', label='128-128')
    plt.legend(loc='upper left')
    plt.show()