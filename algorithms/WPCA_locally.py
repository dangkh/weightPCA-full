# regarding to our locally weighted PCA

import numpy as np
import random
import math
from algorithms.utils import *

def createWeightCol(weight, marker, markerwithgap):
	dtype = [('id', int), ('dist', float)]
	values = []
	for idx, x in enumerate(weight):
		values.append((idx, x))
	a = np.array(values, dtype=dtype) 
	sortedDist = np.sort(a, order="dist")

	listClosest = []
	for x in sortedDist:
		if len(listClosest) > 1: break
		if (x[1] > 0) and (x[0] not in markerwithgap):
			listClosest.append(x[0])
	# CMU 2000
	weightScale = 500
	MMweight = 0.001
	weight_vector = np.exp(np.divide(-np.square(weight),(2*np.square(weightScale))))
	weight_vector[listClosest] = 1
	weight_vector[markerwithgap] = 0.001
	weight_vector[marker] = MMweight
	return weight_vector


class reconstructGap():
	def __init__(self, weight, columnwithgap, marker, inputN_nogap, inputN_zero):
		N_nogap = inputN_nogap
		N_zero = inputN_zero
		markerwithgap = np.unique(columnwithgap // 3)

		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

		new_weight = np.copy(weight)
		# new_weight[markerwithgap] = 0.001

		column_weight = np.ravel(np.ones((3,1)) * new_weight, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m3 = np.matmul(np.ones((N_nogap.shape[0], 1)), column_weight)

		m4 = np.ones((N_nogap.shape[0], 1)) * mean_N_nogap
		m5 = np.ones((N_nogap.shape[0], 1)) * stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1)) * mean_N_zero	

		N_nogap = np.multiply(((N_nogap - m4) / m5), m3)
		N_zero = np.multiply(((N_zero-m6) / m5), m3)
		# N_zero[:, columnwithgap] = 0


		_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap, full_matrices = False)
		U_N_nogap = U_N_nogap_VH.T
		_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero, full_matrices = False)
		U_N_zero = U_N_zero_VH.T

		ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
		ksmall2 = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
		U_N_nogap = U_N_nogap[:, :ksmall]
		U_N_zero = U_N_zero[:, :ksmall]
		self.T_matrix = np.matmul(U_N_nogap.T , U_N_zero)


		# for construct:
		self.mean_N_zero = mean_N_zero
		self.mean_N_nogap = mean_N_nogap
		self.stdev_N_no_gaps = stdev_N_no_gaps
		self.column_weight = column_weight
		self.U_N_nogap = U_N_nogap
		self.U_N_zero = U_N_zero

	def getPredict(self, matrix):
		m1 = np.matmul(np.ones((matrix.shape[0],1)), self.mean_N_zero)
		m2 = np.ones((matrix.shape[0],1))* self.stdev_N_no_gaps
		m3 = np.matmul( np.ones((matrix.shape[0], 1)), self.column_weight)

		M_zero = np.multiply(((np.copy(matrix)-m1) / m2),m3)
		
		reconstructData = np.matmul(np.matmul(np.matmul(M_zero, self.U_N_zero), self.T_matrix), self.U_N_nogap.T)

		# reverse normalization
		m7 = np.ones((matrix.shape[0],1))* self.mean_N_nogap
		m8 = np.ones((reconstructData.shape[0],1))* self.stdev_N_no_gaps
		reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)

		return reconstructData
		

class interpolation_WPCA_local():

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

		# same as PCA_R2
		# weightScale = 200
		# MMweight = 0.2
		# new_weight = [1] * 41
		# for x in markerwithgap:
		# 	new_weight[x] = 0.001

		# new_weight = np.asarray(new_weight)

		# column_weight = np.ravel(np.ones((3,1)) * new_weight, order='F')
		# column_weight = column_weight.reshape((1, column_weight.shape[0]))
		# m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		# m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		# m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		# m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

		# m1 = np.matmul(np.ones((AA.shape[0],1)),mean_N_zero)
		# m2 = np.ones((AA.shape[0],1))*stdev_N_no_gaps
		# m3 = np.matmul( np.ones((AA.shape[0], 1)), column_weight)
		# AA = np.multiply(((AA-m1) / m2),m3)

		# N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		# N_zero = np.multiply(((N_zero-m6) / m5),m33)
		# _, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap, full_matrices = False)
		# U_N_nogap = U_N_nogap_VH.T
		# _, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero, full_matrices = False)
		# U_N_zero = U_N_zero_VH.T
		# ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
		# U_N_nogap = U_N_nogap[:, :ksmall]
		# U_N_zero = U_N_zero[:, :ksmall]

		# T_matrix =  np.matmul(U_N_nogap.T , U_N_zero)

		# reconstruct = np.matmul(np.matmul(np.matmul(AA, U_N_zero), T_matrix), U_N_nogap.T)

		# m7 = np.ones((AA.shape[0],1))*mean_N_nogap
		# m8 = np.ones((AA.shape[0],1))*stdev_N_no_gaps
		# m3 = np.matmul( np.ones((AA.shape[0], 1)), column_weight)
		# reconstruct = m7 + (np.multiply(reconstruct, m8) / m3) + self.MeanMat

		# result = reconstruct[-self.fix_leng:, :]

		# final_result = np.copy(self.missing_matrix)
		# final_result[np.where(self.missing_matrix == 0)] = result[np.where(self.missing_matrix == 0)]
		
		# self.result = final_result
		# return


		# compute Q mask:
		self.listGapInfo = []
		for marker in self.markerwithgap:
			Qg = dist(N_nogap, marker)
			weight = createWeightCol(Qg, marker, self.markerwithgap)
			self.listWeight.append(weight)
			gapInfo = reconstructGap(weight, columnwithgap, marker, N_nogap, N_zero)
			self.listGapInfo.append(gapInfo)

		# listBmatrix = []
		# for idx, gap in enumerate(markerwithgap):
		#     pass


	def compute_weight(self):
		Bmatrix = np.zeros(self.missing_matrix.shape)
		for gth, marker in enumerate(self.markerwithgap):
			for k_patch in range(self.numberPatch):
				l = k_patch * self.fix_leng
				r = l + self.fix_leng
				sample = self.AA[l:r]
				sample_g = np.copy(sample)
				sample_g[:,marker*3: marker*3+3] = 0
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

