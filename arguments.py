import sys
import argparse
sys.path.append("/Users/kieudang/Desktop/weightPCA/")
from data_utils import *

class arguments(argparse.Namespace):

    length3D = 400
    

    #  # """ CMU config """
    # missing_index = [750, 1150]
    # # # current result
    # inner_testing = [ [1250, 1650], [550, 950], [950, 1350], [2250, 2650]]
    
    # collection_link = ["./data3D/135_01.txt", "./data3D/135_03.txt"]
    # outter_testing = [[1450, 1850], [1450, 1850]]
    # data_link = "./data3D/CMU2.txt"
    # # # single case
    # # # weightScale = 2000
    # # # MMweight = 0.001

    # # # mul case
    # # # weightScale = 2000
    # # # MMweight = 0.02
    # # # others = 0.001
    

    
    """ HDM config """
    missing_index = [1650, 2050]
    
    inner_testing = [[1050, 1450], [1450, 1850], [200, 600], [2000, 2400]]
    collection_link = ["./data3D/HDM3.txt", "./data3D/HDM4.txt"]
    outter_testing = [[1450, 1850], [1450, 1850]]
    data_link = "./data3D/HDM5.txt"
    # single case
    # weightScale = 500
    # MMweight = 0.001

    # mul case
    # weightScale = 500
    # MMweight = 0.02
    # others = 0.001


    tmp_AN3 = []
    counter = 0
    for x in collection_link:
        print("reading source: ", x)
        source, _ = read_tracking_data3D_v2(x)
        source = source.astype(float)
        source = source[outter_testing[counter][0]:outter_testing[counter][1]]
        counter += 1
        K = source.shape[0] // length3D
        list_patch = [[x * 400, (x + 1) * 400] for x in range(K)]
        AN3_source = np.copy(source[list_patch[0][0]: list_patch[-1][1]])
        tmp_AN3.append(AN3_source)
    source_AN3 = np.vstack(tmp_AN3)

    
    test_link = "./test/"
    save_dir = "./list_result/"
    default_test_link = "./test/th_out/2.txt"


arg = arguments
