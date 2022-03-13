import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.colors as mcolors
import matplotlib.colors as cl

def load_missing(link=None):
    matrix = []
    f = open(link, 'r')
    print(link)
    for line in f:
        elements = line[:-1].split(' ')
        matrix.append(list(map(int, elements)))
    f.close()

    matrix = np.array(matrix)  # list can not read by index while arr can be
    return 1 - matrix

def discrete_matshow(data):
    #get discrete colormap
    cmap = cl.ListedColormap(['yellow', 'blue'])
    boundaries = [0, 1]
    norm = cl.BoundaryNorm(boundaries, cmap.N, clip=True)
    # set limits .5 outside true range
    mat = plt.imshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5, aspect = 0.1)
    ax = mat.axes
    ax.invert_yaxis()
    plt.xlabel("Marker Entry")
    plt.ylabel("Time Frame")
    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1))

    # plt.show()
    plt.savefig('single.eps', format='eps')



# test_array = np.arange(100 * 100).reshape(100, 100)

matrix = load_missing("./test/1/49.txt")
matrix = 1 - matrix.astype(int)
xselect = [x*3 for x in range(41)]
matrix = matrix[:, xselect]
discrete_matshow(matrix)

