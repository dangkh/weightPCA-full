# backup file locally

# import numpy as np
# import random
# import math
# from algorithms.utils import *

# class interpolation_WPCA_local():

#     def __init__(self, reference_matrix, missing_matrix, refine=False, local=False):
#         self.fix_leng = missing_matrix.shape[0]
#         self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
#         self.missing_matrix = missing_matrix
#         self.reference_matrix = np.copy(reference_matrix)
#         self.local = local
#         self.info = self.prepare()
#         if not local:
#             self.compute_weight()
#             self.compute_alpha()
#         else:
#             self.compute_weight_local()
#             self.compute_alpha_local()
#         self.result = self.interpolate_missing()

#     def prepare(self, remove_patches=False, current_mean=-1):
#         self.mask = 0
#         self.lenPatch = self.fix_leng
#         list_F_matrix = []
#         source_data = np.copy(self.reference_matrix)
#         self.numberPatch = int(source_data.shape[0] / self.lenPatch)
#         AA = np.copy(self.combine_matrix)
#         columnindex = np.where(AA == 0)[1]
#         columnwithgap = np.unique(columnindex)
#         frameindex = np.where(AA == 0)[0]
#         framewithgap = np.unique(frameindex)
#         markerwithgap = np.unique(columnwithgap // 3)
#         self.markerwithgap = markerwithgap
#         self.numGap = len(markerwithgap)
#         [frames, columns] = AA.shape
#         Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
#         mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
#         columnWithoutGap = Data_without_gap.shape[1]

#         x_index = [x for x in range(0, columnWithoutGap, 3)]
#         mean_data_withoutgap_vecX = np.mean(Data_without_gap[:, x_index], 1).reshape(frames, 1)

#         y_index = [x for x in range(1, columnWithoutGap, 3)]
#         mean_data_withoutgap_vecY = np.mean(Data_without_gap[:, y_index], 1).reshape(frames, 1)

#         z_index = [x for x in range(2, columnWithoutGap, 3)]
#         mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:, z_index], 1).reshape(frames, 1)

#         joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
#         self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1] // 3)
#         AA = AA - self.MeanMat
#         AA[np.where(self.combine_matrix == 0)] = 0
#         self.AA = AA

#         N_nogap = np.delete(self.AA, framewithgap, 0)
#         N_zero = np.copy(N_nogap)
#         N_zero[:, columnwithgap] = 0

#         # test
#         mean_N_nogap = np.mean(N_nogap, 0)
#         mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

#         mean_N_zero = np.mean(N_zero, 0)
#         mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
#         stdev_N_no_gaps = np.std(N_nogap, 0)
#         stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

#         weightScale = 200
#         MMweight = 0.02
#         weight_vector = np.ones((len(markerwithgap), columns // 3))
#         weight_vector = np.min(weight_vector, 0)
#         defaultWv = np.copy(weight_vector)
#         # weight_vector = compute_weight_vect_norm(markerwithgap, self.combine_matrix)
#         # if not self.local:
#         # if True:
#         # 	for idx,_ in enumerate(weight_vector):
#         # 		weight_vector[idx] = 1
#         # print(weight_vector)
#         # weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
#         weight_vector[markerwithgap] = MMweight
#         # print(weight_vector)
#         column_weight = np.ravel(np.ones((3, 1)) * weight_vector, order='F')
#         column_weight = column_weight.reshape((1, column_weight.shape[0]))
#         m3 = np.matmul(np.ones((self.AA.shape[0], 1)), column_weight)
#         m33 = np.matmul(np.ones((N_nogap.shape[0], 1)), column_weight)

#         m1 = np.matmul(np.ones((self.AA.shape[0], 1)), mean_N_zero)
#         m2 = np.ones((self.AA.shape[0], 1)) * stdev_N_no_gaps

#         m4 = np.ones((N_nogap.shape[0], 1)) * mean_N_nogap
#         m5 = np.ones((N_nogap.shape[0], 1)) * stdev_N_no_gaps
#         m6 = np.ones((N_zero.shape[0], 1)) * mean_N_zero

#         self.AA = np.multiply(((self.AA - m1) / m2), m3)
#         N_nogap = np.multiply(((N_nogap - m4) / m5), m33)
#         self.N_nogap = N_nogap
#         N_zero = np.copy(N_nogap)
#         N_zero[:, columnwithgap] = 0

#         self.m7 = np.ones((self.AA.shape[0], 1)) * mean_N_nogap
#         self.m8 = np.ones((self.AA.shape[0], 1)) * stdev_N_no_gaps
#         self.m3 = np.matmul(np.ones((self.AA.shape[0], 1)), column_weight)

#         # create Qtildle mask
#         if self.local:
#             self.list_Qtildle = []
#             self.list_Qfull = []
#             fullMat = getFullMat(self.missing_matrix)
#             self.distMatrix = calDistMatrix(fullMat)
#             for marker in self.markerwithgap:
#                 # weight_vector = compute_weight_vect_norm_v2([marker], np.copy(self.missing_matrix))
#                 weight_vector = calWeight(marker, fullMat, self.distMatrix)
#                 weight_vector = np.exp(np.divide(-np.square(weight_vector), (2 * np.square(200))))
#                 # weight_vector = self.refine(weight_vector)
#                 column_weight = np.ravel(np.ones((3, 1)) * weight_vector, order='F')
#                 column_weight = column_weight.reshape((1, column_weight.shape[0]))
#                 Qtildle = np.matmul(np.ones((self.missing_matrix.shape[0], 1)), column_weight)
#                 Qfull = np.matmul(np.ones((self.AA.shape[0], 1)), column_weight)
#                 self.list_Qfull.append(Qfull)
#                 self.list_Qtildle.append(Qtildle)

#         list_maskQ = []
#         # creating mask Q_g
#         Q_tmp = np.ones(self.missing_matrix.shape)
#         full_mask = np.ones(self.missing_matrix.shape)
#         full_mask[np.where(self.missing_matrix) == 0] = 0
#         for g in markerwithgap:
#             Q_gi = np.copy(Q_tmp)
#             Q_gi[:, g * 3:g * 3 + 3] = 0
#             list_maskQ.append(Q_gi)
#         self.list_maskQ = np.copy(list_maskQ)

#         list_Ag = []
#         for g in markerwithgap:
#             sample = np.copy(N_nogap)
#             sample[:, g * 3:g * 3 + 3] = 0
#             list_Ag.append(sample)
#         # compute UN full and compute UN missing regarding to missing gap
#         full_source = np.copy(N_nogap)
#         _, Sigma_nogap, U_N_nogap_VH = np.linalg.svd(
#             full_source / np.sqrt(full_source.shape[0] - 1), full_matrices=False)
#         UN_matrix = U_N_nogap_VH.T

#         full_source_missing = np.copy(N_zero)  # [:self.lenPatch*self.numberPatch,:]
#         _, Sigma_zero, U_N_zero_VH = np.linalg.svd(
#             full_source_missing / np.sqrt(full_source_missing.shape[0] - 1), full_matrices=False)
#         UN0_matrix = U_N_zero_VH.T
#         l2 = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))

#         list_UN0_matrix = []
#         for idx, marker in enumerate(markerwithgap):
#             current_N0 = list_Ag[idx]
#             _, Sigma_zero, U_N_zero_VHg = np.linalg.svd(
#                 current_N0 / np.sqrt(current_N0.shape[0] - 1), full_matrices=False)
#             U_N_zero = U_N_zero_VHg.T
#             U_N_zero = U_N_zero[:, :l2]
#             list_UN0_matrix.append(U_N_zero)
#         # choosing l
#         # UTAT
#         # list_U0TA0T_AU = []
#         # for idx in range(self.numberPatch):
#         # 	l = idx * self.lenPatch
#         # 	r = l + self.lenPatch
#         # 	sample = np.copy(N_nogap[l:r, :])
#         # 	A0 = sample * full_mask
#         # 	U0TA0T = np.matmul(UN0_matrix.T, A0.T)
#         # 	AU = np.matmul(A0, UN_matrix)
#         # 	U0TA0T_AU = np.matmul(U0TA0T, AU)
#         # 	list_U0TA0T_AU.append(U0TA0T_AU)
#         # sum_U0TA0T_AU = summatrix_list(list_U0TA0T_AU)
#         # l_vector = np.linalg.matrix_rank(sum_U0TA0T_AU)
#         # print(sum_U0TA0T_AU.shape)
#         # print(l, l2)
#         l_vector = l2
#         list_T_matrix = []
#         UN_matrix = UN_matrix[:, :l_vector]
#         UN0_matrix = UN0_matrix[:, :l_vector]
#         # for idx,_ in enumerate(list_UN0_matrix):
#         # 	list_UN0_matrix[idx] = list_UN0_matrix[idx][:,:l_vector]

#         self.T_matrix = np.matmul(UN_matrix.T, UN0_matrix)
#         for idx, _ in enumerate(markerwithgap):
#             T_matrix = np.matmul(UN_matrix.T, list_UN0_matrix[idx])
#             list_T_matrix.append(T_matrix)

#         self.full_source = np.copy(full_source)
#         self.full_source_missing = np.copy(full_source_missing)
#         self.UN_matrix = np.copy(UN_matrix)
#         self.UN0_matrix = np.copy(UN0_matrix)
#         self.list_T_matrix = np.copy(list_T_matrix)
#         self.list_UN0_matrix = np.copy(list_UN0_matrix)

#     def compute_weight(self):
#         Bmatrix = np.zeros(self.missing_matrix.shape)
#         for gth, marker in enumerate(self.markerwithgap):
#             for k_patch in range(self.numberPatch):
#                 l = k_patch * self.lenPatch
#                 r = l + self.lenPatch
#                 sample = self.full_source[l:r]
#                 sample_g = sample * self.list_maskQ[gth]
#                 list_tmp = [sample_g, self.list_UN0_matrix[gth], self.list_T_matrix[gth], self.UN_matrix.T]
#                 deduct = sample - matmul_list(list_tmp)
#                 Bmatrix = Bmatrix + deduct
#         # Bmatrix = np.matmul(Bmatrix.T, Bmatrix)
#         # Bmatrix = sumDiff
#         _, Sigma_B, _ = np.linalg.svd(Bmatrix)
#         # print(Sigma_B)
#         accumulateNum = setting_rank(Sigma_B)
#         for x in range(len(Sigma_B)):
#             if x > accumulateNum:
#                 Sigma_B[x] = 1
#         Wii = 1.0 / Sigma_B
#         self.W = diag_matrix(Wii[0], self.missing_matrix.shape[0])
#         return

#     def compute_weight_local(self):

#         Bmatrix = np.zeros(self.missing_matrix.shape)
#         for gth, marker in enumerate(self.markerwithgap):
#             Qtildle = self.list_Qtildle[gth]
#             for k_patch in range(self.numberPatch):
#                 l = k_patch * self.lenPatch
#                 r = l + self.lenPatch
#                 sample = self.full_source[l:r]
#                 sample_g = sample * self.list_maskQ[gth]
#                 list_tmp = [sample_g, self.list_UN0_matrix[gth], self.list_T_matrix[gth], self.UN_matrix.T]
#                 deduct = sample - matmul_list(list_tmp)
#                 deduct = deduct * Qtildle
#                 Bmatrix = Bmatrix + deduct
#         # Bmatrix = np.matmul(Bmatrix.T, Bmatrix)
#         # Bmatrix = sumDiff
#         _, Sigma_B, _ = np.linalg.svd(Bmatrix)
#         # print(Sigma_B)
#         accumulateNum = setting_rank(Sigma_B)
#         for x in range(len(Sigma_B)):
#             if x > accumulateNum:
#                 Sigma_B[x] = 1
#         Wii = 1.0 / Sigma_B
#         self.W = diag_matrix(Wii[0], self.missing_matrix.shape[0])
#         return

#     def compute_alpha_local(self):
#         right_form = np.zeros((self.missing_matrix.shape[1], self.missing_matrix.shape[1]))

#         for idEqua in range(self.numGap):
#             for current_patch in range(self.numberPatch):
#                 # right_part
#                 l = current_patch * self.lenPatch
#                 r = l + self.lenPatch
#                 Aj = self.AA[l:r]
#                 A0j = np.copy(Aj)
#                 A0j[np.where(self.missing_matrix == 0)] = 0
#                 right_part = np.zeros(self.missing_matrix.shape)
#                 for g in range(self.numGap):
#                     right_part += (Aj * self.list_Qtildle[g])
#                 # compute R current_patch(idEqua)
#                 Rjr = np.zeros(right_form.shape)
#                 for i in range(self.numGap):

#                     Ur = self.list_UN0_matrix[idEqua]
#                     Tr = self.list_T_matrix[idEqua]
#                     Qtildle = self.list_Qtildle[i]
#                     left_part = matmul_list([A0j, Ur, Tr, self.UN_matrix.T]).T * Qtildle.T
#                     Rjr += matmul_list([left_part, self.W, right_part])
#                 right_form += Rjr

#         xx, yy = right_form.shape
#         right_form = right_form.reshape(xx * yy, 1)

#         lAlphaMatrix = []
#         for idAlpha in range(self.numGap):
#             lMatrixByEquation = []
#             for idEqua in range(self.numGap):
#                 KG_element = np.zeros((self.missing_matrix.shape[1], self.missing_matrix.shape[1]))

#                 for current_patch in range(self.numberPatch):
#                     l = self.lenPatch * current_patch
#                     r = l + self.lenPatch
#                     Aj = self.AA[l:r, :]
#                     A0j = np.copy(Aj)
#                     A0j[np.where(self.missing_matrix == 0)] = 0
#                     # compute P_currentPatch(idEquation)
#                     Pjg = np.zeros(self.missing_matrix.shape)
#                     for i in range(self.numGap):
#                         Qtildle = self.list_Qtildle[i]
#                         left_part = matmul_list([A0j, self.list_UN0_matrix[idEqua],
#                                                  self.list_T_matrix[idEqua], self.UN_matrix.T])
#                         Pjg += left_part * Qtildle
#                     for s in range(self.numGap):
#                         Qtildle = self.list_Qtildle[s]
#                         right_part = matmul_list([A0j, self.list_UN0_matrix[idAlpha],
#                                                   self.list_T_matrix[idAlpha], self.UN_matrix.T]) * Qtildle
#                         KG_element += matmul_list([Pjg.T, self.W, right_part])

#                 lMatrixByEquation.append(KG_element)

#             tmp = summatrix_list(lMatrixByEquation)
#             xx, yy = tmp.shape
#             matrix2vec = np.copy(tmp.reshape(xx * yy, 1))
#             lAlphaMatrix.append(matrix2vec)

#         left_form = np.hstack(lAlphaMatrix)
#         self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form),
#                                           np.matmul(left_form.T, right_form), rcond=None)[0]
#         # self.list_alpha = [1.0 / len(self.markerwithgap)] * len(self.markerwithgap)
#         print(self.list_alpha)
#         return self.list_alpha

#     def compute_alpha(self):
#         list_R_matrix = []
#         for idx, marker in enumerate(self.markerwithgap):
#             tmp_R = []
#             # missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]
#             for current_patch in range(self.numberPatch):
#                 l = current_patch * self.lenPatch
#                 r = l + self.lenPatch
#                 Aj = self.AA[l:r]
#                 A0j = np.copy(Aj)
#                 A0j[np.where(self.missing_matrix == 0)] = 0
#                 # A0j[missing_frame, marker*3: marker*3+3] = 0
#                 U0N = self.list_UN0_matrix[idx]
#                 T_matrix = self.list_T_matrix[idx]
#                 UN = self.UN_matrix
#                 leftmatrix = matmul_list([A0j, U0N, T_matrix, UN.T])
#                 tmp = [leftmatrix.T, self.W, Aj]
#                 tmp_R.append(matmul_list(tmp))
#             list_R_matrix.append(summatrix_list(tmp_R))
#         right_form = summatrix_list(list_R_matrix)
#         xx, yy = right_form.shape
#         right_form = right_form.reshape(xx * yy, 1)

#         lAlphaMatrix = []
#         for idAlpha in range(self.numGap):
#             lMatrixByEquation = []
#             for idEqua in range(self.numGap):
#                 tmp_matrix = []

#                 # marker = self.markerwithgap[idEqua]
#         # 		missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]
#                 for current_patch in range(self.numberPatch):
#                     # 			# P current_patch in equation idEqua and alpha
#                     l = self.lenPatch * current_patch
#                     r = l + self.lenPatch
#                     Aj = self.AA[l:r, :]
#                     A0j = np.copy(Aj)
#                     # A0j[missing_frame, marker*3 : marker*3+3] = 0
#                     # A0j[:, marker*3 : marker*3+3] = 0
#                     A0j[np.where(self.missing_matrix == 0)] = 0
#                     U0N = self.list_UN0_matrix[idEqua]
#                     T_matrix = self.list_T_matrix[idEqua]
#                     UN = self.UN_matrix
#                     tmp = [A0j, U0N, T_matrix, UN.T]
#                     Pjg = matmul_list(tmp)
#                     U0N_alpha = self.list_UN0_matrix[idAlpha]
#                     T_matrix_alpha = self.list_T_matrix[idAlpha]
#                     tmp = [Pjg.T, self.W, A0j, U0N_alpha, T_matrix_alpha, UN.T]
#                     tmp_matrix.append(matmul_list(tmp))
#                 lMatrixByEquation.append(summatrix_list(tmp_matrix))
#             tmp = summatrix_list(lMatrixByEquation)
#             xx, yy = tmp.shape
#             matrix2vec = np.copy(tmp.reshape(xx * yy, 1))
#             lAlphaMatrix.append(matrix2vec)

#         left_form = np.hstack(lAlphaMatrix)
#         self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form),
#                                           np.matmul(left_form.T, right_form), rcond=None)[0]
#         # self.list_alpha = [1.0 / len(self.markerwithgap)] * len(self.markerwithgap)
#         print(self.list_alpha)
#         return self.list_alpha

#     def interpolate_missing(self):
#         result_tmp = np.zeros(self.AA.shape)
#         for gth in range(len(self.markerwithgap)):
#             sample = matmul_list([np.copy(self.AA), self.list_UN0_matrix[gth],
#                                   self.list_T_matrix[gth], self.UN_matrix.T])
#             result_tmp += self.list_alpha[gth] * sample
#         result = self.m7 + (np.multiply(result_tmp, self.m8) / self.m3) + self.MeanMat
#         result = result[-self.lenPatch:, :]

#         final_result = np.copy(self.missing_matrix)
#         final_result[np.where(self.missing_matrix == 0)] = result[np.where(self.missing_matrix == 0)]
#         return final_result

#     def refine(self, arr):
#         newArr = np.ones(arr.shape)
#         for idx, _ in enumerate(arr):
#             if arr[idx] < 0.01:
#                 newArr[idx] = 0
#         return newArr
