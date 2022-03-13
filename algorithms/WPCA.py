# regarding to our weighted PCA

import numpy as np
import random
import math
from algorithms.utils import *


class reconstructGap():
    def __init__(self, columnwithgap, marker, inputN_nogap, inputN_zero):
        N_nogap = inputN_nogap
        N_zero = inputN_zero
        markerwithgap = np.unique(columnwithgap // 3)

        mean_N_nogap = np.mean(N_nogap, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(N_zero, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(N_nogap, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1
        new_weight = [1] * 41
        for x in markerwithgap:
            new_weight[x] = 0.001

        new_weight[marker] = 0.1
        new_weight = np.asarray(new_weight)
        
        column_weight = np.ravel(np.ones((3, 1)) * new_weight, order='F')
        column_weight = column_weight.reshape((1, column_weight.shape[0]))
        m3 = np.matmul(np.ones((N_nogap.shape[0], 1)), column_weight)

        m4 = np.ones((N_nogap.shape[0], 1)) * mean_N_nogap
        m5 = np.ones((N_nogap.shape[0], 1)) * stdev_N_no_gaps

        N_nogap = np.multiply(((N_nogap - m4) / m5), m3)
        N_zero = np.copy(N_nogap)
        N_zero[:, columnwithgap] = 0

        _, Sigma_nogap, U_N_nogap_VH = np.linalg.svd(N_nogap, full_matrices=False)
        U_N_nogap = U_N_nogap_VH.T
        _, Sigma_zero, U_N_zero_VH = np.linalg.svd(N_zero, full_matrices=False)
        U_N_zero = U_N_zero_VH.T

        ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
        U_N_nogap = U_N_nogap[:, :ksmall]
        U_N_zero = U_N_zero[:, :ksmall]
        self.T_matrix = np.matmul(U_N_nogap.T, U_N_zero)

        # for construct:
        self.mean_N_zero = mean_N_zero
        self.mean_N_nogap = mean_N_nogap
        self.stdev_N_no_gaps = stdev_N_no_gaps
        self.column_weight = column_weight
        self.U_N_nogap = U_N_nogap
        self.U_N_zero = U_N_zero

    def getPredict(self, matrix):
        m1 = np.matmul(np.ones((matrix.shape[0], 1)), self.mean_N_zero)
        m2 = np.ones((matrix.shape[0], 1)) * self.stdev_N_no_gaps
        m3 = np.matmul(np.ones((matrix.shape[0], 1)), self.column_weight)

        M_zero = np.multiply(((np.copy(matrix) - m1) / m2), m3)

        reconstructData = np.matmul(np.matmul(np.matmul(M_zero, self.U_N_zero), self.T_matrix), self.U_N_nogap.T)

        # reverse normalization
        m7 = np.ones((matrix.shape[0], 1)) * self.mean_N_nogap
        m8 = np.ones((reconstructData.shape[0], 1)) * self.stdev_N_no_gaps
        reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)

        return reconstructData


class interpolation_WPCA():

    def __init__(self, reference_matrix, missing_matrix, refine=False, local=False):
        self.fix_leng = missing_matrix.shape[0]
        self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
        self.missing_matrix = missing_matrix
        self.reference_matrix = np.copy(reference_matrix)
        self.local = local
        self.info = self.prepare()
        self.compute_weight()
        self.compute_alpha()
        self.result = self.interpolate_missing()

    def prepare(self, remove_patches=False, current_mean=-1):

        self.mask = []
        self.listWeight = []
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

        self.listGapInfo = []
        for marker in self.markerwithgap:
            gapInfo = reconstructGap(columnwithgap, marker, N_nogap, N_zero)
            self.listGapInfo.append(gapInfo)


    def compute_weight(self):
        Bmatrix = np.zeros(self.missing_matrix.shape)
        for gth, marker in enumerate(self.markerwithgap):
            for k_patch in range(self.numberPatch):
                l = k_patch * self.fix_leng
                r = l + self.fix_leng
                sample = self.AA[l:r]
                sample_g = np.copy(sample)
                sample_g[:, marker * 3: marker * 3 + 3] = 0
                deduct = sample - self.listGapInfo[gth].getPredict(sample_g)
                Bmatrix = Bmatrix + deduct
        # Bmatrix = np.matmul(Bmatrix.T, Bmatrix)
        # Bmatrix = sumDiff

        _, Sigma_B, _ = np.linalg.svd(Bmatrix)
        # print(Sigma_B)
        accumulateNum = setting_rank(Sigma_B)
        for x in range(len(Sigma_B)):
            if x > accumulateNum:
                Sigma_B[x] = 1
        Wii = 1.0 / Sigma_B
        self.W = diag_matrix(Wii[0], self.missing_matrix.shape[0])
        return

    def compute_alpha(self):
        list_R_matrix = []
        for g, marker in enumerate(self.markerwithgap):
            tmp_R = []
        #     # missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]
            for j in range(self.numberPatch):
                l = j * self.fix_leng
                r = l + self.fix_leng
                Aj = self.AA[l:r]
                A0j = np.copy(Aj)
                A0j[np.where(self.missing_matrix == 0)] = 0
                left = self.listGapInfo[g].getPredict(A0j)
        #         # A0j[missing_frame, marker*3: marker*3+3] = 0
        #         U0N = self.list_UN0_matrix[idx]
        #         T_matrix = self.list_T_matrix[idx]
        #         UN = self.UN_matrix
        #         leftmatrix = matmul_list([A0j, U0N, T_matrix, UN.T])
                tmp = [left.T, self.W, Aj]
                tmp_R.append(matmul_list(tmp))
            list_R_matrix.append(summatrix_list(tmp_R))
        right_form = summatrix_list(list_R_matrix)
        xx, yy = right_form.shape
        right_form = right_form.reshape(xx * yy, 1)

        lAlphaMatrix = []
        for idAlpha in range(self.numGap):
            lMatrixByEquation = []
            for idEqua in range(self.numGap):
                tmp_matrix = []

                # marker = self.markerwithgap[idEqua]
        #       missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]
                for j in range(self.numberPatch):
                    #           # P current_patch in equation idEqua and alpha
                    l = self.fix_leng * j
                    r = l + self.fix_leng
                    Aj = self.AA[l:r, :]
                    A0j = np.copy(Aj)
                    # A0j[missing_frame, marker*3 : marker*3+3] = 0
                    # A0j[:, marker*3 : marker*3+3] = 0
                    A0j[np.where(self.missing_matrix == 0)] = 0
                    Pjg = self.listGapInfo[idEqua].getPredict(A0j)
        #             U0N = self.list_UN0_matrix[idEqua]
        #             T_matrix = self.list_T_matrix[idEqua]
        #             UN = self.UN_matrix
        #             tmp = [A0j, U0N, T_matrix, UN.T]
        #             Pjg = matmul_list(tmp)

        #             U0N_alpha = self.list_UN0_matrix[idAlpha]
        #             T_matrix_alpha = self.list_T_matrix[idAlpha]
                    remain = self.listGapInfo[idAlpha].getPredict(A0j)
                    tmp = [Pjg.T, self.W, remain]
                    tmp_matrix.append(matmul_list(tmp))
                lMatrixByEquation.append(summatrix_list(tmp_matrix))
            tmp = summatrix_list(lMatrixByEquation)
            xx, yy = tmp.shape
            matrix2vec = np.copy(tmp.reshape(xx * yy, 1))
            lAlphaMatrix.append(matrix2vec)

        left_form = np.hstack(lAlphaMatrix)
        self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form),
                                          np.matmul(left_form.T, right_form), rcond=None)[0]
        # self.list_alpha = [1.0 / len(self.markerwithgap)] * len(self.markerwithgap)
        print(self.list_alpha)
        return self.list_alpha

    def interpolate_missing(self):

        result = np.zeros(self.AA.shape)
        for gth in range(len(self.markerwithgap)):
            sample = self.listGapInfo[gth].getPredict(self.AA) + self.MeanMat
            result += self.list_alpha[gth] * sample
        result = result[-self.fix_leng:, :]

        final_result = np.copy(self.missing_matrix)
        final_result[np.where(self.missing_matrix == 0)] = result[np.where(self.missing_matrix == 0)]
        return final_result


# commented is first version.

# def compute_reconstruction(M_zero, N_nogap, N_zero, weight_vector, patch_number = -1, patcher_leng = -1, marker = -1, useFullMatrix = False):
#   backupNN = np.copy(N_nogap)
#   mean_N_nogap = np.mean(N_nogap[:400], 0)
#   mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

#   mean_N_zero = np.mean(N_zero[:400], 0)
#   mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
#   stdev_N_no_gaps = np.std(N_nogap[:400], 0)
#   stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


#   column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
#   column_weight = column_weight.reshape((1, column_weight.shape[0]))
#   m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
#   m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
#   m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
#   m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

#   m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
#   m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
#   m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
#   M_zero = np.multiply(((M_zero-m1) / m2),m3)

#   N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
#   N_zero = np.multiply(((N_zero-m6) / m5),m33)

#   _, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
#   U_N_nogap = U_N_nogap_VH.T
#   _, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
#   U_N_zero = U_N_zero_VH.T
#   ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
#   U_N_nogap = U_N_nogap[:, :ksmall]
#   U_N_zero = U_N_zero[:, :ksmall]
#   if patch_number == -1:
#       T_matrix =  np.matmul(U_N_nogap.T , U_N_zero)
#   else:
#       totalPatch = N_nogap.shape[0] // patcher_leng
#       leftover = N_nogap.shape[0] - totalPatch * patcher_leng
#       l = patch_number * patcher_leng
#       r = l + patcher_leng
#       Ai = N_nogap[l: r, :]
#       Ai0 = N_zero[l: r, :]
#       # AiUN = np.matmul(Ai, U_N_nogap)
#       # Ai0UN0 = np.matmul(Ai0, U_N_zero)
#       N_nogap_excl = np.vstack((N_nogap[l:r, :], N_nogap[totalPatch * patcher_leng:, :]))
#       N_zero_excl = np.vstack((N_zero[l:r, :], N_zero[totalPatch * patcher_leng:, :]))
#       _, Sigma_nogap_excl , U_N_nogap_VH_excl = np.linalg.svd(N_nogap_excl/np.sqrt(N_nogap_excl.shape[0]-1), full_matrices = False)
#       U_N_nogap_excl = U_N_nogap_VH_excl.T
#       _, Sigma_zero_excl , U_N_zero_VH_excl = np.linalg.svd(N_zero_excl/np.sqrt(N_zero_excl.shape[0]-1), full_matrices = False)
#       U_N_zero_excl = U_N_zero_VH_excl.T
#       ksmall_excl = max(setting_rank(Sigma_zero_excl), setting_rank(Sigma_nogap_excl))
#       U_N_nogap_excl = U_N_nogap_excl[:, :ksmall_excl]
#       U_N_zero_excl = U_N_zero_excl[:, :ksmall_excl]


#       # X = np.linalg.lstsq(np.matmul(Ai0UN0.T, Ai0UN0), np.matmul(Ai0UN0.T,AiUN), rcond = None)
#       T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
#       T_matrix1 =  np.matmul(U_N_nogap.T , U_N_zero)
#       T_matrix2 =  np.matmul(U_N_nogap_excl.T , U_N_zero_excl)
#       aireconst = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
#       aireconst1 = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix1), U_N_nogap.T)
#       aireconst2 = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero_excl), T_matrix2), U_N_nogap_excl.T)
#       cmp00 = aireconst1 - M_zero
#       cmp01 = aireconst2 - M_zero
#       cmp00 = cmp00[:-patcher_leng]
#       cmp01 = cmp01[:-patcher_leng]
#       # print(np.sum(np.abs(cmp00[marker*3:marker*3+3])))
#       # print(np.sum(np.abs(cmp01[marker*3:marker*3+3])))
#       # if np.sum(np.abs(cmp00[marker*3:marker*3+3])) > np.sum(np.abs(cmp01[marker*3:marker*3+3])):
#       if not useFullMatrix:
#           T_matrix = T_matrix2
#           U_N_zero = U_N_zero_excl
#           U_N_nogap = U_N_nogap_excl
#       #np.savetxt("checkres0.txt", cmp00, fmt = "%.2f")
#       #np.savetxt("checkres1.txt", cmp01, fmt = "%.2f")
#       #np.savetxt("check"+str(patch_number)+".txt", T_matrix, fmt = "%.2f")

#   reconstruct_Mzero = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
#   # mean_N_nogap = np.mean(backupNN[:400], 0)
#   # mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))
#   # stdev_N_no_gaps = np.std(backupNN[:400], 0)
#   # stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1
#   m7 = np.ones((M_zero.shape[0],1))*mean_N_nogap
#   m8 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
#   m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
#   reconstruct_Mzero = m7 + (np.multiply(reconstruct_Mzero, m8) / m3)
#   return reconstruct_Mzero

# class interpolation_gap_patch():
#   def __init__(self, patch_number, marker, missing_matrix_origin, full_data, weight_vector_input = None, useFullMatrix = False):
#       missing_matrix = np.copy(missing_matrix_origin)

#       weightScale = 200
#       MMweight = 0.02
#       DistalThreshold = 0.5
#       missing_frame = np.where(missing_matrix_origin[:, marker*3] == 0)[0]

#       columnindex = np.where(missing_matrix_origin == 0)[1]
#       columnwithgap = np.unique(columnindex)
#       markerwithgap = np.unique(columnwithgap // 3)
#       try:
#           if not weight_vector_input:
#               weight_vector = compute_weight_vect_norm_v2([marker], full_data)
#       except Exception as e:
#           weight_vector = weight_vector_input

#       # if not weight_vector_input:
#       #   weight_vector = compute_weight_vect_norm_v2([marker], full_data)
#       # else:
#       #   weight_vector = weight_vector_input
#       EuclDist2Marker = weight_vector
#       thresh = np.mean(EuclDist2Marker) * DistalThreshold
#       Data_remove_joint = np.copy(full_data)
#       for sub_marker in range(len(EuclDist2Marker)):
#           if (EuclDist2Marker[sub_marker] > thresh) and (sub_marker in markerwithgap):
#               Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
#       Data_remove_joint[:, marker*3:marker*3+3] = np.copy(full_data[:, marker*3:marker*3+3])
#       closest_joint = get_closest(marker, weight_vector)
#       nonOverlap, overlap = get_overlap(marker, closest_joint, missing_matrix_origin)

#       frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)[0]
#       for sub_marker in markerwithgap:
#           if (sub_marker != marker) and (sub_marker!= closest_joint):
#               if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
#                   Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0

#       weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
#       weight_vector[marker] = MMweight
#       for x in range(len(weight_vector)):
#           if math.isnan(weight_vector[x]) :
#               weight_vector[x] = 0

#       numFrames = full_data[:-missing_matrix.shape[0],:].shape[0]
#       reconstruct = np.copy(full_data)
#       if len(nonOverlap) > 0:
#           tmp_frame = np.asarray(overlap) + numFrames
#           if len(overlap) > 0:
#               M_zero = np.delete(Data_remove_joint, tmp_frame, 0)
#           else:
#               M_zero = np.copy(Data_remove_joint)

#           tmp_frame = np.asarray(missing_frame) + numFrames
#           N_nogap = np.delete(Data_remove_joint, tmp_frame, 0)
#           N_zero = np.copy(N_nogap)
#           N_zero[:,marker*3: marker*3+3] = 0

#           reconstruct_Mzero = compute_reconstruction(M_zero, N_nogap, N_zero, weight_vector, patch_number, missing_matrix_origin.shape[0], marker, useFullMatrix)

#           tmp_frame = np.asarray(nonOverlap) + numFrames
#           missingMzero = np.where(M_zero[:, marker*3] == 0)[0]
#           reconstruct[tmp_frame, marker*3:marker*3+3] = reconstruct_Mzero[missingMzero, marker*3:marker*3+3]

#       if len(overlap) > 0:
#           tmp_frame = np.asarray(nonOverlap) + numFrames
#           Data_remove_joint[:,closest_joint*3:closest_joint*3+3] = 0
#           M_zero = np.delete(Data_remove_joint, tmp_frame, 0)

#           tmp_frame = np.asarray(missing_frame) + numFrames
#           N_nogap = np.delete(Data_remove_joint, tmp_frame, 0)
#           N_zero = np.copy(N_nogap)
#           N_zero[:,marker*3: marker*3+3] = 0

#           reconstruct_Mzero = compute_reconstruction(M_zero, N_nogap, N_zero, weight_vector, patch_number, missing_matrix_origin.shape[0], marker, useFullMatrix)

#           tmp_frame = np.asarray(overlap) + numFrames
#           missingMzero = np.where(M_zero[:, marker*3] == 0)[0]
#           reconstruct[tmp_frame, marker*3:marker*3+3] = reconstruct_Mzero[missingMzero, marker*3:marker*3+3]

#       self.result = reconstruct


# class interpolation_WPCA():

#   def __init__(self, reference_matrix, missing_matrix, refine = False):
#       self.fix_leng = missing_matrix.shape[0]
#       self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
#       self.reference_matrix = np.copy(reference_matrix)
#       self.original_missing = missing_matrix
#       self.normed_matries, self.reconstruct_matries = self.normalization()
#       self.A1 = np.copy(self.normed_matries[0])
#       self.AN = np.copy(self.normed_matries[1])
#       self.AN0 = np.copy(self.normed_matries[2])
#       self.K = int(self.AN.shape[0] / self.fix_leng)
#       self.compute_alpha()
#       self.F_matrix = self.prepare()
#       self.result = self.interpolate_missing()

#   def prepare(self, remove_patches = False, current_mean = -1):
#       list_F_matrix = []
#       DistalThreshold = 0.5
#       test_data = np.copy(self.original_missing)
#       source_data = np.copy(self.reference_matrix)

#       AA = np.copy(self.combine_matrix)
#       columnindex = np.where(AA == 0)[1]
#       columnwithgap = np.unique(columnindex)
#       markerwithgap = np.unique(columnwithgap // 3)
#       self.markerwithgap = markerwithgap
#       missing_frame_testdata = np.unique(np.where(test_data == 0)[0])
#       list_frame = np.arange(test_data.shape[0])
#       full_frame_testdata = np.asarray([i for i in list_frame if i not in missing_frame_testdata])
#       self.full_frame_testdata = full_frame_testdata

#       [frames, columns] = AA.shape
#       Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
#       mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
#       columnWithoutGap = Data_without_gap.shape[1]

#       x_index = [x for x in range(0, columnWithoutGap, 3)]
#       mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

#       y_index = [x for x in range(1, columnWithoutGap, 3)]
#       mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

#       z_index = [x for x in range(2, columnWithoutGap, 3)]
#       mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

#       joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
#       self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
#       AA = AA - self.MeanMat
#       AA[np.where(AA == 0)] = 0.0001
#       AA[np.where(self.combine_matrix == 0)] = 0
#       self.norm_Data = np.copy(AA)
#       self.K = self.reference_matrix.shape[0] + len(full_frame_testdata)
#       self.K = int(self.K / self.fix_leng)
#       list_F_matrix = []
#       list_weight_vector_marker = []
#       self.useFullMatrix = True
#       if len(self.markerwithgap) > 0:
#           self.useFullMatrix = False

#       for marker in markerwithgap:
#           wv = compute_weight_vect_norm_v2([marker], np.copy(AA))
#           list_weight_vector_marker.append(wv)
#       for patch_counter in range(self.K):
#           resultPatch = np.copy(AA)
#           for idx, marker in enumerate(markerwithgap):
#               wv = list_weight_vector_marker[idx]
#               gap_interpolation = interpolation_gap_patch(patch_counter, marker, self.original_missing, np.copy(AA), weight_vector_input = wv, useFullMatrix = self.useFullMatrix)
#               missing_frame = np.where(AA[:, marker*3] == 0)[0]
#               resultPatch[missing_frame, marker*3:marker*3+3] = gap_interpolation.result[missing_frame, marker*3:marker*3+3]
#           list_F_matrix.append(resultPatch)
#       return list_F_matrix


#   def normalization(self):
#       normed_matries, reconstruct_matries = compute_norm(self.combine_matrix)
#       return normed_matries, reconstruct_matries

#   def compute_alpha(self):
#       p_AN = self.AN
#       p_AN0 = self.AN0
#       list_A0 = []
#       list_A = []

#       r = len(self.AN0)
#       l = r - self.fix_leng

#       while l >= 0:
#           list_A.append(np.copy(p_AN[l:r]))
#           list_A0.append(np.copy(p_AN0[l:r]))
#           l -= self.fix_leng
#           r -= self.fix_leng

#       _, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
#       self.UN = np.copy(tmp_U.T)

#       _, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
#       self.UN0 = np.copy(tmp_U0.T)

#       ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma))

#       self.UN = self.UN[:, :ksmall]
#       self.UN0 = self.UN0[:, :ksmall]
#       self.list_Ti = []

#       for patch_number in range(self.K):
#           AiUN = np.matmul(list_A[patch_number], self.UN)
#           Ai0UN0 = np.matmul(list_A0[patch_number], self.UN0)

#           X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
#           self.list_Ti.append(np.copy(X[0]))

#       list_left_matrix = []
#       for patch_number in range(self.K):
#           current_patch = np.matmul(list_A[patch_number], self.UN) - matmul_list(
#               [list_A0[patch_number], self.UN0, self.list_Ti[patch_number]])
#           for column in range(ksmall):
#               for clm in range(ksmall):
#                   tmp = np.multiply(current_patch[:, column], current_patch[:, clm])
#                   list_left_matrix.append(tmp)

#       left_matrix = np.vstack(list_left_matrix)

#       u, d, v = np.linalg.svd(left_matrix)
#       v = v.T
#       weight_list = v[:, -1]
#       self.W = np.diag(weight_list)
#       # compute alpha

#       list_Qjk = []
#       for j in range(self.K):
#           for h in range(self.K) :
#               tmpQ = matmul_list([matmul_list([list_A0[j], self.UN0, self.list_Ti[h]]).T,
#                   self.W, list_A[j], self.UN])
#               list_Qjk.append(tmpQ)
#       right_form = summatrix_list(list_Qjk)
#       xx, yy = right_form.shape
#       right_form = right_form.reshape(xx*yy, 1)
#       list_Pij_patch = []
#       for patch_number in range(self.K):
#           list_tmp = []
#           for j in range(self.K):
#               for h in range(self.K):
#                   tmpP = matmul_list([matmul_list([list_A0[j], self.UN0, self.list_Ti[h]]).T,
#                       self.W, list_A0[j], self.UN0, self.list_Ti[patch_number]])
#                   list_tmp.append(tmpP)
#           tmp = summatrix_list(list_tmp)
#           xx, yy = tmp.shape
#           list_Pij_patch.append(tmp.reshape(xx*yy, 1))

#       left_form = np.hstack([ x for x in list_Pij_patch])
#       # self.list_alpha = np.linalg.lstsq(left_form, right_form, rcond = None)[0]
#       self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
#       print(self.list_alpha)
#       return self.list_alpha


#   def interpolate_missing(self):
#       list_matrix = []
#       for patch_number in range(self.K):
#           tmp = (self.list_alpha[patch_number]) * self.F_matrix[patch_number]
#           list_matrix.append(tmp)
#       result = summatrix_list(list_matrix)
#       result = result[-self.original_missing.shape[0]:] + self.MeanMat[-self.original_missing.shape[0]:]

#       final_result = np.copy(self.original_missing)
#       final_result[np.where(self.original_missing == 0)] = result[np.where(self.original_missing == 0)]
#       return final_result
