# generate missing gap in specific joint according to PLOS16
import numpy as np
from data_utils import *
from arguments import arg
import sys
import random
from datetime import datetime
import os


def generate_missing_frame(n, m):
    # CMU:
    gaps = 1
    matrix = np.ones((n, m))
    joints = np.arange(m // 3)

    counter = 0
    print('start : ')

    while counter < gaps:
        missing_joint = joints[random.randint(0, m // 3 - 1)]
        print(missing_joint)
        counter += 1
        matrix[:, missing_joint * 3: missing_joint * 3 + 3] = 0

    counter = 0
    for x in range(n):
        for y in range(m):
            if matrix[x][y] == 0:
                counter += 1
    print("end.")
    print("percent missing: ", 100.0 * counter / (n * m))
    return matrix


def process_gap_missing():

    test_location = arg.test_link
    test_reference = arg.missing_index
    sample = np.copy(Tracking3D[test_reference[0]:test_reference[1]])
    link = test_location + "th_out/"
    if not os.path.isdir(link):
        os.mkdir(link)
    for times in range(50):
        print("current: ", times)

        missing_matrix = generate_missing_frame(sample.shape[0], sample.shape[1])

        np.savetxt(test_location + "th_out/" + str(times) + ".txt", missing_matrix, fmt="%d")
    # f = open(test_location + "/info.txt", "w")
    # f.write(str(datetime.now()))
    # f.close()
    return


if __name__ == '__main__':

    Tracking3D, _ = read_tracking_data3D_v2(arg.data_link)
    Tracking3D = Tracking3D.astype(float)
    process_gap_missing()
