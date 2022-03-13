import numpy as np
from algorithms.PCAR2 import *
from algorithms.utils import *
from algorithms.WPCA import *
from algorithms.WPCAG import *
from algorithms.WPCA_locally import *
from algorithms.WPCA_wholeFrame import *
from algorithms.LWPCA_wholeFrame import *
from algorithms.WPCA_throughout import *
from algorithms.WPCA_best import *


def PLOS_2013(test_data):
    result = PCA_2013(test_data)
    return result


def PLOS_R2(test_data):
    interpolation = PCA_R2(test_data)
    result = interpolation.result_norm
    return result

def Lis2020(source_data, test_data):
    interpolation = PCA_Li(source_data, test_data)
    result = interpolation.result
    return result

def interpolation_best(source_data, test_data):
    interpolation = WPCA_best(source_data, test_data)
    result = interpolation.result
    return result

def WPCAG(source_data, test_data):
    interpolation = interpolation_WPCAG(source_data, test_data)
    result = interpolation.result
    return result


def WPCA(source_data, test_data):
    interpolation = interpolation_WPCA(source_data, test_data)
    result = interpolation.result
    return result


def WPCA_local(source_data, test_data):
    interpolation = interpolation_WPCA_local(source_data, test_data)
    result = interpolation.result
    return result


def WPCA_wholeFrame(source_data, test_data, db=False):
    if not db:
        interpolation = interpolation_WPCA_wholeFrame(source_data, test_data)
    else:
        interpolation = Li1st_wholeFrame(source_data, test_data)
    result = interpolation.result
    return result

def LWPCA_wholeFrame(source_data, test_data, db=False):
    if not db:
        interpolation = interpolation_LWPCA_wholeFrame(source_data, test_data)
    else:
        interpolation = interpolation_MaskWPCA_wholeFrame(source_data, test_data)
    result = interpolation.result
    result = interpolation.result
    return result


def WPCA_throughout(source_data, test_data, db=False):
    if not db:
        interpolation = interpolation_WPCA_through(source_data, test_data)
    else:
        interpolation = Li1st_through1(source_data, test_data)
    result = interpolation.result
    return result


if __name__ == '__main__':
    print("ok")
