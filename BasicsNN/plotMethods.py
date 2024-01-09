
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix

def bar(x: any,
        y: any,
        title: str,
        color: str = 'maroon',
        width: float = 0.4):
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x, y, color=color, width=width)
    plt.title(title)
    plt.ioff()
    return plt

def samplesOfSet(data: any,
                 label: any,
                 title: str,
                 nrows: int,
                 ncols: int,
                 firstIndex: int,
                 lastIndex: int,
                 height: int = 28,
                 width: int = 28,
                 cmap: str = "gray"):
    ncols = math.ceil(len(data) / ncols)
    fig = plt.figure(figsize=(10, 5))
    chooseFirstIndex = lambda a, b: 0 if (a > b) else a
    chooseLastIndex = lambda a, b : a if(a < b) else b
    firstIndex = chooseFirstIndex(firstIndex, len(data))
    lastIndex = chooseLastIndex(len(data), lastIndex)
    for index in range(firstIndex, lastIndex):
        plt.subplot(nrows, ncols, index + 1)
        plt.imshow(data[index].reshape(height, width), cmap=cmap)
        plt.title("Class " + str(label[index]))
        plt.axis('off')
    plt.ioff()
    fig.suptitle(title)
    return plt

def misclassificationReport(data: any,
                            indexesMisclassified: any,
                            trueLabel: any,
                            predictedLabel: any,
                            title: str,
                            nrows: int,
                            ncols: int,
                            firstIndex: int,
                            lastIndex: int,
                            height: int = 28,
                            width: int = 28,
                            cmap: str = "gray") :
    chooseFirstIndex = lambda a, b: 0 if (a > b) else a
    chooseLastIndex = lambda a, b: a if (a < b) else b
    firstIndex = chooseFirstIndex(firstIndex, len(indexesMisclassified))
    lastIndex = chooseLastIndex(ncols, lastIndex)
    fig = plt.figure(figsize=(10, 5))
    for index in range(firstIndex, lastIndex):
        plt.subplot(nrows, ncols, index + 1)
        plt.imshow(data[index].reshape(height, width), cmap=cmap)
        plt.title(str(trueLabel[indexesMisclassified[index]]) + " : " + str(predictedLabel[indexesMisclassified[index]]))
        plt.axis('off')
    plt.ioff()
    fig.suptitle(title)
    return plt

def compareResults(train: any,
                   val: any,
                   title: str,
                   xlabel: str,
                   ylabel: str,
                   trainColor: str='blue',
                   valColor: str='green'):
    plt.figure(figsize=(10, 5))
    plt.plot(train, color=trainColor)
    plt.plot(val, color=valColor)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.axis('off')
    plt.ioff()
    return plt

def show_confusion_matrix(testing : any,
                          predictions : any,
                          num_classes : any,
                          labels : any):
    matrix = confusion_matrix(testing, predictions)
    plt.figure(figsize=(10,5))
    hm = sns.heatmap(matrix,
                     cmap='coolwarm',
                     linecolor='white',
                     linewidths=1,
                     xticklabels=labels,
                     yticklabels=labels,
                     annot=True,
                     fmt='d')
    plt.yticks(rotation = 0)  # Don't rotate (vertically) the y-axis labels
    #hm.invert_yaxis() # Invert the labels of the y-axis
    hm.set_ylim(0, len(matrix))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.ioff()
    return plt