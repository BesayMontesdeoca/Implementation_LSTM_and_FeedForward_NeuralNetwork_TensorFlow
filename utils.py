import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import math

def getData(path, option, percentage):
    data = readData(path)
    maxi = np.amax(data)
    if option[0] == "r":
        if option[1:].isdigit():
            if int(option[1:] == 1): return crossValidation(data, percentage)
            a, b = crossValidation(jumpData(data, int(option[1:])), percentage)
            return a, b, maxi
    elif option[0] == "p":
        if option[1:].isdigit():
            if int(option[1:] == 1): return crossValidation(data, percentage)
            a, b = crossValidation(proData(data, int(option[1:])), percentage)
            return a, b, maxi

def getData2(path, option, percentage):
    data = readData2(path)
    maxi = np.amax(data)
    if option[0] == "r":
        if option[1:].isdigit():
            if int(option[1:] == 1): return crossValidation(data, percentage)
            a, b = crossValidation(jumpData(data, int(option[1:])), percentage)
            return a, b, maxi
    elif option[0] == "p":
        if option[1:].isdigit():
            if int(option[1:] == 1): return crossValidation(data, percentage)
            a, b = crossValidation(proData(data, int(option[1:])), percentage)
            return a, b, maxi

def jumpData(dat, n):
    data = []
    for i in range(1, len(dat)):
        if i % n == 0:
            data.append(dat[i])
    return np.array(data)

def proData(dat, n):
    data = []
    for i in range(len(dat)):
        if i % n == 0:
            s = sum(dat[i:i+n])
            data.append(s/float(n))
    return np.array(data)

def readData(path):
    data = []
    reader = csv.reader(open(path, "rb"), delimiter=';')
    for index, row in enumerate(reader):
        data.append([float(row[2])])
    return np.array(data)

def readData2(path):
    data = []
    reader = csv.reader(open(path, "rb"), delimiter=';')
    for index, row in enumerate(reader):
        data.append(float(row[2]))
    return np.array(data)


def crossValidation(data, percentage):
    dataTrainSize = int(round((percentage/100.0)*len(data)))
    dataTrain = data[0:dataTrainSize]
    dataTest = data[dataTrainSize:]
    return dataTrain, dataTest

def plotPrediction(prediction, data, exportPath):
    prediction = prediction[1:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Prediction")
    d = mpatches.Patch(color='blue', label='Real data')
    p = mpatches.Patch(color='green', label='Prediction')
    plt.legend(handles=[d, p])
    ax.plot(data)
    ax.plot(prediction)
    plt.savefig(exportPath + "/prediction.png")
    plt.show()


def plotCostTraining(cost_training, exportPath):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Const-training")
    ax.plot(cost_training)
    plt.savefig(exportPath + "/cost_training.png")
    plt.show()

def saveTrainingResults(hyperparameters, cost_training, prediction, data, exportPath):
    if os.path.exists(exportPath)==0: os.mkdir(exportPath)
    f1 = open(exportPath + '/hyperparameters.txt', 'w')
    f1.write("num_epoch: " + str(hyperparameters[0]) + "\n")
    f1.write("window_size: " + str(hyperparameters[1]) + "\n")
    f1.write("batch_size: " + str(hyperparameters[2]) + "\n")
    f1.write("n_hidden: " + str(hyperparameters[3]) + "\n")
    f1.write("alpha_lr: " + str(hyperparameters[4]) + "\n")
    f1.close()

    plotCostTraining(cost_training, exportPath)
    plotPrediction(prediction, data, exportPath)

