# PLOS algorithm
import numpy as np
from data_utils import *
from arguments import arg
import sys
import random
from algorithms.main import *
import os
from datetime import datetime


def load_missing(sub_link=None):
    if sub_link == None:
        link = arg.default_test_link
    else:
        link = sub_link
    matrix = []
    f = open(link, 'r')
    print(link)
    for line in f:
        elements = line[:-1].split(' ')
        matrix.append(list(map(int, elements)))
    f.close()

    matrix = np.array(matrix)  # list can not read by index while arr can be
    return matrix


def process_hub(data=None):
    result_list = []

    test_folder = arg.test_link
    prefix = arg.save_dir
    order_fol = []
    print("missing in data collection: ")
    for test_name in os.listdir(test_folder):
        current_folder = test_folder + test_name
        if os.path.isdir(current_folder):
            order_fol.append(test_name)
            result_subtest = []
            test_frameIdx = arg.missing_index

            number_test = 50
            for sub_test in range(number_test):
                test_path = current_folder + '/' + str(sub_test) + ".txt"
                print(test_path)
                # full_matrix = load_missing()
                full_matrix = load_missing(test_path)

                A1 = np.copy(Tracking3D[test_frameIdx[0]:test_frameIdx[1]])
                missing_matrix = np.copy(full_matrix)
                A1zero = np.copy(A1)
                A1zero[np.where(missing_matrix == 0)] = 0
                print("******************************************************************************************")
                print("**************************************** STARTING ****************************************")
                print("******************************************************************************************")

               
                print(" ********************PCAR2********************")
                timecounter = datetime.now()
                A1_star = PLOS_R2(np.copy(A1zero))
                result_subtest.append(np.around(calculate_mae_matrix(
                    A1[np.where(A1zero == 0)] - A1_star[np.where(A1zero == 0)]), decimals=17))
                print(str(datetime.now() - timecounter))
            
            result_list.append(np.asarray(result_subtest).mean())
            # break
    return [result_list], order_fol


if __name__ == '__main__':
    savingLocation = arg.save_dir
    f = open(savingLocation + 'resultPLOS1_HDM_duration.txt', "w")
    print("reference source:")
    Tracking3D, _ = read_tracking_data3D_v2(arg.data_link)
    Tracking3D = Tracking3D.astype(float)
    result, order_fol = process_hub()
    print(result)
    f.write(str(result) + "\n")
    f.write(str(order_fol) + "\n")
    f.write(str(datetime.now()))
    f.close()
