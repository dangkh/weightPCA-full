# compare with PCA or PLOS 1 with normalization
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

def extractLP(ref):
    l = 0
    res = []
    while l + 400 < ref[0]:
        res.append([l, l+400])
        l += 400

    r = ref[0] - 200
    while r + 400 < 3400:
        res.append([r, r+400])
        r+= 400
    return res

def process_hub(data=None):
    resultA5 = []
    resultA6 = []
    resultA7 = []
    resultA8 = []
    test_folder = arg.test_link
    for test_name in os.listdir(test_folder):
        current_folder = test_folder + test_name
        if os.path.isdir(current_folder):
            tmpA5 = []
            tmpA6 = []
            tmpA7 = []
            tmpA8 = []
            frame = random.randint(1, 3000)
            test_reference = [frame, frame + 400]
            list_patch = extractLP(test_reference)
            number_test = 200
            for sub_test in range(number_test):
                A_N3_source_added = np.vstack(
                    [np.copy(Tracking3D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])

                print("collection shape : ", A_N3_source_added.shape)
                order_fol = []
                print("missing in data collection: ")
                print(np.where(A_N3_source_added == 0))
                frames = np.where(A_N3_source_added == 0)[0]
                frames = np.unique(frames)
                A_N3_source_added = np.delete(A_N3_source_added, frames, 0)

                result_path = current_folder + '/' + str(sub_test) + ".txt"
                print(result_path)
                # if os.path.isdir(current_folder+'/'+sub_test) :
                tmpG = []
                tmpH = []
                tmpV = []
                tmpS = []
                # full_matrix = load_missing()
                full_matrix = load_missing(result_path)
                A1 = np.copy(Tracking3D[test_reference[0]:test_reference[1]])
                missing_matrix = np.copy(full_matrix)
                A1zero = np.copy(A1)
                A1zero[np.where(missing_matrix == 0)] = 0
                np.savetxt("missing_matrix.txt", A1zero, fmt = "%.2f")
                print("******************************************************************************************")
                print("**************************************** STARTING ****************************************")
                print("******************************************************************************************")

                # print(" ********************PCAR2********************")
                # timecounter = datetime.now()
                # A1_star5 = PLOS_R2(np.copy(A1zero))
                # value = np.around(calculate_mae_matrix(
                #     A1[np.where(A1zero == 0)] - A1_star5[np.where(A1zero == 0)]), decimals=17)
                # tmpG.append(value)
                # print(str(datetime.now() - timecounter))
                
                print(" ********************WPCA********************")
                timecounter = datetime.now()
                A1_star7 = WPCA(np.copy(A_N3_source_added), np.copy(A1zero))
                value = np.around(calculate_mae_matrix(
                    A1[np.where(A1zero == 0)] - A1_star7[np.where(A1zero == 0)]), decimals=17)
                tmpV.append(value)
                print(str(datetime.now() - timecounter))

                print(" ********************WPCA_local********************")
                timecounter = datetime.now()
                A1_star8 = WPCA_local(np.copy(A_N3_source_added), np.copy(A1zero))
                value = np.around(calculate_mae_matrix(
                    A1[np.where(A1zero == 0)] - A1_star8[np.where(A1zero == 0)]), decimals=17)
                tmpS.append(value)
                print(str(datetime.now() - timecounter))


                
                tmpA5.append(np.asarray(tmpG).sum())
                tmpA6.append(np.asarray(tmpH).sum())
                tmpA7.append(np.asarray(tmpV).sum())
                tmpA8.append(np.asarray(tmpS).sum())
                # break
            resultA5.append(np.asarray(tmpA5).mean())
            resultA6.append(np.asarray(tmpA6).mean())
            resultA7.append(np.asarray(tmpA7).mean())
            resultA8.append(np.asarray(tmpA8).mean())
            print(tmpA5, tmpA6, tmpA7, tmpA8)
            f.write(str([tmpA5, tmpA6, tmpA7, tmpA8]) + "\n")
    print(order_fol)
    return [resultA5, resultA6, resultA7, resultA8], order_fol


if __name__ == '__main__':
    savingLocation = arg.save_dir
    f = open(savingLocation + 'result.txt', "w")
    print("reference source:")
    source_AN3 = arg.source_AN3
    Tracking3D, _ = read_tracking_data3D_v2(arg.data_link)
    Tracking3D = Tracking3D.astype(float)
    # # HDM
    result, order_fol = process_hub()
    # CMU
    print(result)
    f.write(str(result) + "\n")
    f.write(str(order_fol) + "\n")
    f.write(str(datetime.now()))
    f.close()