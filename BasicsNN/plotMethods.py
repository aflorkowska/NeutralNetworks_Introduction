
import matplotlib.pyplot as plt

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
                 cmap: str = "gray"):
    fig = plt.figure(figsize=(10, 5))
    for index in range(firstIndex, lastIndex):
        plt.subplot(nrows, ncols, index + 1)
        plt.imshow(data[index], cmap=cmap)
        plt.title("Class " + str(label[index]))
        plt.axis('off')
    plt.ioff()
    fig.suptitle(title)
    return plt



