import numpy as np
import random
import math
from algorithms.utils import *


class PCA_Li_2():

    def __init__(self, reference_matrix, missing_matrix, refine=False, local=False):
        self.fix_leng = missing_matrix.shape[0]
        self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
        self.missing_matrix = missing_matrix
        self.reference_matrix = np.copy(reference_matrix)
        self.local = local
        self.prepare()

    def prepare(self, remove_patches=False, current_mean=-1):

        source_data = np.copy(self.reference_matrix)
        self.numberPatch = int(source_data.shape[0] / self.fix_leng)
        AA = np.copy(self.combine_matrix)
        columnindex = np.where(AA == 0)[1]
        columnwithgap = np.unique(columnindex)
        frameindex = np.where(AA == 0)[0]
        framewithgap = np.unique(frameindex)
        markerwithgap = np.unique(columnwithgap // 3)
        self.markerwithgap = markerwithgap
        self.numGap = len(markerwithgap)
        [frames, columns] = AA.shape
        Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
        columnWithoutGap = Data_without_gap.shape[1]

        x_index = [x for x in range(0, columnWithoutGap, 3)]
        mean_data_withoutgap_vecX = np.mean(Data_without_gap[:, x_index], 1).reshape(frames, 1)

        y_index = [x for x in range(1, columnWithoutGap, 3)]
        mean_data_withoutgap_vecY = np.mean(Data_without_gap[:, y_index], 1).reshape(frames, 1)

        z_index = [x for x in range(2, columnWithoutGap, 3)]
        mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:, z_index], 1).reshape(frames, 1)

        joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
        self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1] // 3)
        AA = AA - self.MeanMat
        AA[np.where(self.combine_matrix == 0)] = 0
        self.AA = AA

        N_nogap = np.delete(self.AA, framewithgap, 0)
        N_zero = np.copy(N_nogap)
        N_zero[:, columnwithgap] = 0

        # test
        mean_N_nogap = np.mean(N_nogap, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(N_zero, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(N_nogap, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

        weightScale = 200
        MMweight = 0.02
        new_weight = [1] * 41
        for x in markerwithgap:
            new_weight[x] = 1

        new_weight = np.asarray(new_weight)

        column_weight = np.ravel(np.ones((3,1)) * new_weight, order='F')
        column_weight = column_weight.reshape((1, column_weight.shape[0]))
        m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
        m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
        m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
        m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

        m1 = np.matmul(np.ones((AA.shape[0],1)),mean_N_zero)
        m2 = np.ones((AA.shape[0],1))*stdev_N_no_gaps
        m3 = np.matmul( np.ones((AA.shape[0], 1)), column_weight)
        AA = np.multiply(((AA-m1) / m2),m3)

        N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
        N_zero = np.multiply(((N_zero-m6) / m5),m33)
        _, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(np.matmul(N_nogap.T, N_nogap), full_matrices = False)
        U_N_nogap = U_N_nogap_VH.T
        _, Sigma_zero , U_N_zero_VH = np.linalg.svd(np.matmul(N_zero.T, N_zero), full_matrices = False)
        U_N_zero = U_N_zero_VH.T
        ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
        U_N_nogap = U_N_nogap[:, :ksmall]
        U_N_zero = U_N_zero[:, :ksmall]

        numPatch = AA.shape[0] // self.fix_leng
        listUi = []
        for patch in range(numPatch):
            l = patch * self.fix_leng
            r = l + self.fix_leng
            Ni_zero = N_zero[l:r]
            _, Sigma_zero , Ui_N_zero_VH = np.linalg.svd(np.matmul(Ni_zero.T, Ni_zero), full_matrices = False)
            Ui_N_zero = Ui_N_zero_VH.T
            Ui_N_zero = Ui_N_zero[:, :ksmall]
            listUi.append(Ui_N_zero)


        left = []
        for patch in range(numPatch):
            l = patch * self.fix_leng
            r = l + self.fix_leng
            Ni_zero = N_zero[l:r]
            Ni_nogap = N_nogap[l:r]
            tmp = [listUi[patch].T, Ni_zero.T, Ni_nogap, U_N_nogap]
            left.append(matmul_list(tmp, True))
        leftMatrix = summatrix_list(left)

        right = []
        for patch in range(numPatch):
            l = patch * self.fix_leng
            r = l + self.fix_leng
            Ni_zero = N_zero[l:r]
            tmp = [listUi[patch].T, Ni_zero.T, Ni_zero, listUi[patch]]
            right.append(matmul_list(tmp))
        rightMatrix = summatrix_list(right)

        T_matrix =  np.matmul(leftMatrix , np.linalg.inv(rightMatrix))

        reconstruct = np.matmul(np.matmul(np.matmul(AA, U_N_zero), T_matrix), U_N_nogap.T)

        m7 = np.ones((AA.shape[0],1))*mean_N_nogap
        m8 = np.ones((AA.shape[0],1))*stdev_N_no_gaps
        m3 = np.matmul( np.ones((AA.shape[0], 1)), column_weight)
        reconstruct = m7 + (np.multiply(reconstruct, m8) / m3) + self.MeanMat

        result = reconstruct[-self.fix_leng:, :]

        final_result = np.copy(self.missing_matrix)
        final_result[np.where(self.missing_matrix == 0)] = result[np.where(self.missing_matrix == 0)]
        
        self.result = final_result


class PCA_Li():

    def __init__(self, reference_matrix, missing_matrix, refine=False, local=False):
        self.fix_leng = missing_matrix.shape[0]
        self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
        self.missing_matrix = missing_matrix
        self.reference_matrix = np.copy(reference_matrix)
        self.local = local
        self.prepare()

    def prepare(self, remove_patches=False, current_mean=-1):

        source_data = np.copy(self.reference_matrix)
        self.numberPatch = int(source_data.shape[0] / self.fix_leng)
        AA = np.copy(self.combine_matrix)
        columnindex = np.where(AA == 0)[1]
        columnwithgap = np.unique(columnindex)
        frameindex = np.where(AA == 0)[0]
        framewithgap = np.unique(frameindex)
        markerwithgap = np.unique(columnwithgap // 3)
        self.markerwithgap = markerwithgap
        self.numGap = len(markerwithgap)
        [frames, columns] = AA.shape
        Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
        columnWithoutGap = Data_without_gap.shape[1]

        x_index = [x for x in range(0, columnWithoutGap, 3)]
        mean_data_withoutgap_vecX = np.mean(Data_without_gap[:, x_index], 1).reshape(frames, 1)

        y_index = [x for x in range(1, columnWithoutGap, 3)]
        mean_data_withoutgap_vecY = np.mean(Data_without_gap[:, y_index], 1).reshape(frames, 1)

        z_index = [x for x in range(2, columnWithoutGap, 3)]
        mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:, z_index], 1).reshape(frames, 1)

        joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
        self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1] // 3)
        AA = AA - self.MeanMat
        AA[np.where(self.combine_matrix == 0)] = 0
        self.AA = AA

        N_nogap = np.delete(self.AA, framewithgap, 0)
        N_zero = np.copy(N_nogap)
        N_zero[:, columnwithgap] = 0

        # test
        mean_N_nogap = np.mean(N_nogap, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(N_zero, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(N_nogap, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

        weightScale = 200
        MMweight = 0.2
        new_weight = [1] * 41
        for x in markerwithgap:
            new_weight[x] = 0.001

        new_weight = np.asarray(new_weight)

        column_weight = np.ravel(np.ones((3,1)) * new_weight, order='F')
        column_weight = column_weight.reshape((1, column_weight.shape[0]))
        m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
        m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
        m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
        m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

        m1 = np.matmul(np.ones((AA.shape[0],1)),mean_N_zero)
        m2 = np.ones((AA.shape[0],1))*stdev_N_no_gaps
        m3 = np.matmul( np.ones((AA.shape[0], 1)), column_weight)
        AA = np.multiply(((AA-m1) / m2),m3)

        N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
        N_zero = np.multiply(((N_zero-m6) / m5),m33)
        _, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap, full_matrices = False)
        U_N_nogap = U_N_nogap_VH.T
        _, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero, full_matrices = False)
        U_N_zero = U_N_zero_VH.T
        ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
        U_N_nogap = U_N_nogap[:, :ksmall]
        U_N_zero = U_N_zero[:, :ksmall]

        T_matrix =  np.matmul(U_N_nogap.T , U_N_zero)

        reconstruct = np.matmul(np.matmul(np.matmul(AA, U_N_zero), T_matrix), U_N_nogap.T)

        m7 = np.ones((AA.shape[0],1))*mean_N_nogap
        m8 = np.ones((AA.shape[0],1))*stdev_N_no_gaps
        m3 = np.matmul( np.ones((AA.shape[0], 1)), column_weight)
        reconstruct = m7 + (np.multiply(reconstruct, m8) / m3) + self.MeanMat

        result = reconstruct[-self.fix_leng:, :]

        final_result = np.copy(self.missing_matrix)
        final_result[np.where(self.missing_matrix == 0)] = result[np.where(self.missing_matrix == 0)]
        
        self.result = final_result

def PCA_2013(inputdata):

    combine_matrix = np.copy(inputdata)
    weightScale = 200
    MMweight = 0.02
    [frames, columns] = combine_matrix.shape
    columnindex = np.where(combine_matrix == 0)[1]
    frameindex = np.where(combine_matrix == 0)[0]
    columnwithgap = np.unique(columnindex)
    markerwithgap = np.unique(columnwithgap // 3)
    framewithgap = np.unique(frameindex)
    Data_without_gap = np.delete(combine_matrix, columnwithgap, 1)
    mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
    columnWithoutGap = Data_without_gap.shape[1]

    x_index = [x for x in range(0, columnWithoutGap, 3)]
    mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

    y_index = [x for x in range(1, columnWithoutGap, 3)]
    mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

    z_index = [x for x in range(2, columnWithoutGap, 3)]
    mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

    joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
    MeanMat = np.tile(joint_meanXYZ, combine_matrix.shape[1]//3)
    Data = np.copy(combine_matrix - MeanMat)
    Data[np.where(combine_matrix == 0)] = 0

    weight_vector = compute_weight_vect_norm(markerwithgap, Data)
    weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
    weight_vector[markerwithgap] = MMweight
    M_zero = np.copy(Data)
    
    N_nogap = np.delete(Data, framewithgap, 0)
    N_zero = np.copy(N_nogap)
    N_zero[:,columnwithgap] = 0
    mean_N_nogap = np.mean(N_nogap, 0)
    mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

    mean_N_zero = np.mean(N_zero, 0)
    mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
    stdev_N_no_gaps = np.std(N_nogap, 0)
    stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

    
    m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
    m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
    
    column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
    column_weight = column_weight.reshape((1, column_weight.shape[0]))
    m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
    m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
    m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
    m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
    m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

    M_zero = np.multiply(((M_zero-m1) / m2),m3)
    N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
    N_zero = np.multiply(((N_zero-m6) / m5),m33)

    _, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
    U_N_nogap = U_N_nogap_VH.T
    print(U_N_nogap.shape)
    _, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
    U_N_zero = U_N_zero_VH.T
    print(U_N_zero.shape)
    ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
    U_N_nogap = U_N_nogap[:, :ksmall]
    U_N_zero = U_N_zero[:, :ksmall]
    T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
    reconstructData = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
    
    # reverse normalization
    m7 = np.ones((Data.shape[0],1))*mean_N_nogap
    m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
    m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
    reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
    tmp = reconstructData + MeanMat
    result = np.copy(tmp[-inputdata.shape[0]:,:])
    final_result = np.copy(inputdata)
    final_result[np.where(inputdata == 0)] = result[np.where(inputdata == 0)]
    return final_result


class interpolation_gap_patch_PLOS_R2():
    def __init__(self, marker, missing_matrix_origin, full_test2, full_data):
        weightScale = 200
        MMweight = 0.02
        missing_frame = np.where(missing_matrix_origin[:, marker*3] == 0)[0]

        weight_vector = compute_weight_vect_norm_v2([marker], full_data)
        weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
        weight_vector[marker] = MMweight
        for x in range(len(weight_vector)):
            if math.isnan(weight_vector[x]) :
                weight_vector[x] = 0
        
        missing_frame_Mzero = np.where(full_test2[:, marker*3] == 0)[0]
        list_frame = np.arange(full_test2.shape[0])
        list_full_frame_Mzero= np.asarray([i for i in list_frame if i not in missing_frame_Mzero])

        M_zero = np.copy(full_test2)

        N_nogap = np.copy(full_test2[list_full_frame_Mzero,:])
        N_zero = np.copy(N_nogap)
        N_zero[:,marker*3: marker*3+3] = 0
        mean_N_nogap = np.mean(N_nogap, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(N_zero, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(N_nogap, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


        column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
        column_weight = column_weight.reshape((1, column_weight.shape[0]))
        m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
        m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
        m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
        m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

        m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
        m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
        m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
        M_zero = np.multiply(((M_zero-m1) / m2),m3)

        N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
        N_zero = np.multiply(((N_zero-m6) / m5),m33)
        _, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
        U_N_nogap = U_N_nogap_VH.T
        _, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
        U_N_zero = U_N_zero_VH.T
        ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
        U_N_nogap = U_N_nogap[:, :ksmall]
        U_N_zero = U_N_zero[:, :ksmall]

        self.ksmall = ksmall

        T_matrix =  np.matmul(U_N_nogap.T , U_N_zero)

        reconstruct_Mzero = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)

        m7 = np.ones((M_zero.shape[0],1))*mean_N_nogap
        m8 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
        m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
        reconstruct_Mzero = m7 + (np.multiply(reconstruct_Mzero, m8) / m3)

        resultA = np.zeros(full_test2.shape)

        resultA[:, marker*3: marker*3+3] = np.copy(reconstruct_Mzero[:, marker*3: marker*3+3])
        self.result = resultA

class PCA_R2():
    def __init__(self, missing_matrix):
        self.fix_leng = missing_matrix.shape[0]
        self.combine_matrix = np.copy(missing_matrix)
        self.missing_matrix = missing_matrix
        self.original_missing = missing_matrix
        self.mean_error = -1

        self.F_matrix = self.prepare()
        self.result_norm = self.interpolate_missing()

    def prepare(self, remove_patches = False, current_mean = -1):
        list_F_matrix = []
        DistalThreshold = 0.5
        test_data = np.copy(self.missing_matrix)

        AA = np.copy(self.combine_matrix)
        columnindex = np.where(AA == 0)[1]
        columnwithgap = np.unique(columnindex)
        markerwithgap = np.unique(columnwithgap // 3)
        self.markerwithgap = markerwithgap
        missing_frame_testdata = np.unique(np.where(test_data == 0)[0])
        list_frame = np.arange(test_data.shape[0])
        full_frame_testdata = np.asarray([i for i in list_frame if i not in missing_frame_testdata])
        self.full_frame_testdata = full_frame_testdata

        [frames, columns] = AA.shape
        Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
        mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
        columnWithoutGap = Data_without_gap.shape[1]

        x_index = [x for x in range(0, columnWithoutGap, 3)]
        mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

        y_index = [x for x in range(1, columnWithoutGap, 3)]
        mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

        z_index = [x for x in range(2, columnWithoutGap, 3)]
        mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

        joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
        self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
        AA = AA - self.MeanMat
        AA[np.where(self.combine_matrix == 0)] = 0
        self.norm_Data = np.copy(AA)
        resultPatch = np.zeros(self.combine_matrix.shape)

        for marker in markerwithgap:
            missing_frame = np.where(test_data[:, marker*3] == 0)
            EuclDist2Marker = compute_weight_vect_norm([marker], AA)
            thresh = np.mean(EuclDist2Marker) * DistalThreshold
            Data_remove_joint = np.copy(AA)
            for sub_marker in range(len(EuclDist2Marker)):
                if (EuclDist2Marker[sub_marker] > thresh) and (sub_marker in markerwithgap):
                    Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
            Data_remove_joint[:, marker*3:marker*3+3] = np.copy(AA[:, marker*3:marker*3+3])
            frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)[0]
            for sub_marker in markerwithgap:
                if sub_marker != marker:
                    if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
                        Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0

            gap_interpolation = interpolation_gap_patch_PLOS_R2(marker, self.missing_matrix, Data_remove_joint, np.copy(AA))

            missing_frame = np.where(Data_remove_joint[:, marker*3] == 0)[0]
            resultPatch[missing_frame, marker*3:marker*3+3] = gap_interpolation.result[missing_frame, marker*3:marker*3+3]
        return resultPatch

    def interpolate_missing(self):
        result = self.F_matrix[-self.missing_matrix.shape[0]:] + self.MeanMat[-self.missing_matrix.shape[0]:]

        final_result = np.copy(self.original_missing)
        final_result[np.where(self.original_missing == 0)] = result[np.where(self.original_missing == 0)]
        return final_result
