import numpy as np
import random
import math
from algorithms.utils import *

class interpolation_WPCAG():

	def __init__(self, reference_matrix, missing_matrix, refine = False):
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.missing_matrix = missing_matrix
		self.reference_matrix = np.copy(reference_matrix)
		
		self.info = self.prepare()
		self.compute_weight2()
		self.compute_alpha()
		# self.compute_weight1()
		# self.compute_alpha()
		self.normed_matries, self.reconstruct_matries = self.normalization()
		self.A1 = np.copy(self.normed_matries[0])
		self.AN = np.copy(self.normed_matries[1])
		self.AN0 = np.copy(self.normed_matries[2])
		self.K = int(self.AN.shape[0] / self.fix_leng)
		# self.compute_beta()
		self.result = self.interpolate_missing()
		# self.result = self.debug

	def prepare(self, remove_patches = False, current_mean = -1):
		self.lenPatch = self.fix_leng
		list_F_matrix = []
		source_data = np.copy(self.reference_matrix)
		self.numberPatch = int(source_data.shape[0] / self.lenPatch)
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
		self.AA = AA
		
		N_nogap = np.delete(self.AA, framewithgap, 0)
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0
		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

		weightScale = 200
		MMweight = 0.02
		weight_vector = compute_weight_vect_norm(markerwithgap, self.AA)
		self.weight_vector = weight_vector
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[markerwithgap] = MMweight
		m1 = np.matmul(np.ones((self.AA.shape[0],1)),mean_N_zero)
		m2 = np.ones((self.AA.shape[0],1))*stdev_N_no_gaps
		
		column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m3 = np.matmul( np.ones((self.AA.shape[0], 1)), column_weight)
		m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

		self.AA = np.multiply(((self.AA-m1) / m2),m3)
		N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		self.N_nogap = N_nogap
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0
		self.m7 = np.ones((self.AA.shape[0],1))*mean_N_nogap
		self.m8 = np.ones((self.AA.shape[0],1))*stdev_N_no_gaps
		self.m3 = np.matmul( np.ones((self.AA.shape[0], 1)), column_weight)

		N_nogap = N_nogap[self.numberPatch * self.lenPatch - 400:, :]
		N_zero = N_zero[self.numberPatch * self.lenPatch - 400:, :]
		_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
		U_N_nogap = U_N_nogap_VH.T
		_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
		U_N_zero = U_N_zero_VH.T

		ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
		U_N_nogap = U_N_nogap[:, :ksmall]
		U_N_zero = U_N_zero[:, :ksmall]

		T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
		# tmp = [np.copy(self.AA), U_N_zero, T_matrix, U_N_nogap.T]
		# reconstructData = self.m7 + (np.multiply(matmul_list(tmp), self.m8) / self.m3)
		# interpolate = reconstructData + self.MeanMat
		# self.debug = interpolate[-self.lenPatch:,:]

		self.useFullMatrix = True
		if len(self.markerwithgap) > 0:
			self.useFullMatrix = False

		list_UN_matrix = []
		list_UN0_matrix = []
		list_T_matrix = []
		returnData = np.copy(self.AA)
		for marker in markerwithgap:			
			tmp_T_matrix = []	
			tmp_UN = []
			tmp_UNO = []
			N_nogap = self.N_nogap
			N_zero = np.copy(N_nogap)
			N_zero[:, marker*3: marker*3+3] = 0
			_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
			U_N_nogap = U_N_nogap_VH.T
			_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
			U_N_zero = U_N_zero_VH.T

			ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
			U_N_nogap = U_N_nogap[:, :ksmall]
			U_N_zero = U_N_zero[:, :ksmall]

			Tg_matrix = np.matmul(U_N_nogap.T , U_N_zero)
			# tinh duoc Tg o day

			missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]

			# for current_patch in range(self.numberPatch):
			# 	totalPatch = self.numberPatch
			# 	leftover = N_nogap.shape[0] - totalPatch * self.lenPatch
			# 	l = current_patch * self.lenPatch
			# 	r = l + self.lenPatch
			# 	# Ai = N_nogap[l: r, :]
			# 	# Ai0 = N_zero[l: r, :]
			# 	N_nogap_excl = np.vstack((N_nogap[l:r, :], N_nogap[totalPatch * self.lenPatch:, :]))
			# 	N_zero_excl = np.vstack((N_zero[l:r, :], N_zero[totalPatch * self.lenPatch:, :]))
			# 	_, Sigma_nogap_excl , U_N_nogap_VH_excl = np.linalg.svd(N_nogap_excl/np.sqrt(N_nogap_excl.shape[0]-1), full_matrices = False)
			# 	U_N_nogap_excl = U_N_nogap_VH_excl.T
			# 	_, Sigma_zero_excl , U_N_zero_VH_excl = np.linalg.svd(N_zero_excl/np.sqrt(N_zero_excl.shape[0]-1), full_matrices = False)
			# 	U_N_zero_excl = U_N_zero_VH_excl.T
			# 	ksmall_excl = max(setting_rank(Sigma_zero_excl), setting_rank(Sigma_nogap_excl))
			# 	U_N_nogap_excl = U_N_nogap_excl[:, :ksmall_excl]
			# 	U_N_zero_excl = U_N_zero_excl[:, :ksmall_excl]
			# 	T_matrix =  np.matmul(U_N_nogap_excl.T , U_N_zero_excl)
			# 	# compute T_matrix regarding to patch j and current marker
			# 	# l = current_patch * self.lenPatch
			# 	# r = l + self.lenPatch
			# 	# Ai = self.AA[l: r, :]
			# 	# Ai0 = np.copy(Ai)
			# 	# Ai0[missing_frame, marker*3 : marker*3+3] = 0
			# 	# AiUN = np.matmul(Ai, U_N_nogap)
			# 	# Ai0UN0 = np.matmul(Ai0, U_N_zero)
				
			# 	# X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
			# 	# T_matrix = np.copy(X[0])

				# if self.useFullMatrix:
				# 	T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
				# 	tmp_T_matrix.append(T_matrix)
				# 	tmp_UN.append(U_N_nogap)
				# 	tmp_UNO.append(U_N_zero)
				# else:
				# 	tmp_T_matrix.append(T_matrix)
				# 	tmp_UN.append(U_N_nogap_excl)
				# 	tmp_UNO.append(U_N_zero_excl)

			list_UN_matrix.append(U_N_nogap)
			list_UN0_matrix.append(U_N_zero)
			list_T_matrix.append(Tg_matrix)
		self.list_UN_matrix = list_UN_matrix
		self.list_UN0_matrix = list_UN0_matrix
		self.list_T_matrix = list_T_matrix
		return [list_UN_matrix, list_UN0_matrix, list_T_matrix]

	def compute_weight1(self):
		columnindex = np.where(self.missing_matrix == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		sumDiff = np.zeros(self.missing_matrix.shape)
		for idMarker, marker in enumerate(markerwithgap):
			# missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)
			for current_patch in range(self.numberPatch):
				A_nogap = self.AA[self.lenPatch*current_patch: self.lenPatch*(current_patch+1),:]
				A_zero = np.copy(A_nogap)
				A_zero[:, marker*3: marker*3+3] = 0
				U_N_nogap = self.list_UN_matrix[idMarker][current_patch]
				U_N_zero = self.list_UN0_matrix[idMarker][current_patch]
				T_matrix = self.list_T_matrix[idMarker][current_patch]
				interpolate = matmul_list([A_zero, U_N_zero, T_matrix, U_N_nogap.T])
				interpolate[np.where(self.missing_matrix != 0)] = A_nogap[np.where(self.missing_matrix != 0)]
				tmpDiff = A_nogap - interpolate
				sumDiff = sumDiff + tmpDiff
		# Bmatrix = np.matmul(sumDiff, sumDiff.T)
		Bmatrix = sumDiff
		_, Sigma_B , _ = np.linalg.svd(Bmatrix)
		accumulateNum = setting_rank(Sigma_B)
		for x in range(len(Sigma_B)):
			if x > accumulateNum:
				Sigma_B[x] = 1
		Wii = 1.0/ Sigma_B
		self.W = diag_matrix(Wii[0], self.missing_matrix.shape[0])
		# stop
		return

	def compute_weight2(self):
		MMweight = 0.02
		columnindex = np.where(self.missing_matrix == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		sumDiff = np.zeros(self.missing_matrix.shape)
		# compute_weight_vect_norm
		list_weight_vector_marker = [] 
		for marker in markerwithgap:
			wv = compute_weight_vect_norm_v2([marker], np.copy(self.AA))
			list_weight_vector_marker.append(wv)
		
		for idMarker, marker in enumerate(markerwithgap):
			# missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)
			weight_vector = list_weight_vector_marker[idMarker]
			column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
			column_weight = column_weight.reshape((1, column_weight.shape[0]))
			Qtildle = np.matmul( np.ones((self.missing_matrix.shape[0], 1)), column_weight)
			for current_patch in range(self.numberPatch):
				A_nogap = self.AA[self.lenPatch*current_patch: self.lenPatch*(current_patch+1),:]
				A_zero = np.copy(A_nogap)
				A_zero[:, marker*3: marker*3+3] = 0
				U_N_nogap = self.list_UN_matrix[idMarker]
				U_N_zero = self.list_UN0_matrix[idMarker]
				T_matrix = self.list_T_matrix[idMarker]
				interpolate = matmul_list([A_zero, U_N_zero, T_matrix, U_N_nogap.T])
				# interpolate[np.where(self.missing_matrix != 0)] = A_nogap[np.where(self.missing_matrix != 0)]
				tmpDiff = A_nogap - interpolate
				tmpDiff = np.multiply(tmpDiff, Qtildle)
				sumDiff = sumDiff + tmpDiff
		# Bmatrix = np.matmul(sumDiff.T, sumDiff)
		Bmatrix = sumDiff
		_, Sigma_B , _ = np.linalg.svd(Bmatrix)
		accumulateNum = setting_rank(Sigma_B)
		for x in range(len(Sigma_B)):
			if x > accumulateNum:
				Sigma_B[x] = 1
		Wii = 1.0/ Sigma_B
		self.W = diag_matrix(Wii[0], self.missing_matrix.shape[0])
		return

	def compute_alpha(self):
		list_R_matrix = []
		for idMarker, marker in enumerate(self.markerwithgap):
			tmp_R = []
			missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]
			for current_patch in range(self.numberPatch):
				l = current_patch * self.lenPatch
				r = l + self.lenPatch
				Aj = self.AA[l:r]
				A0j = np.copy(Aj)
				A0j[missing_frame, marker*3: marker*3+3] = 0
				U0N = self.list_UN0_matrix[idMarker]
				T_matrix = self.list_T_matrix[idMarker]
				UN = self.list_UN_matrix[idMarker]
				leftmatrix =  matmul_list([A0j,  U0N,T_matrix, UN.T])
				tmp = [leftmatrix.T, self.W, Aj]
				tmp_R.append(matmul_list(tmp))
			list_R_matrix.append(summatrix_list(tmp_R))
		right_form = summatrix_list(list_R_matrix)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)
		# stop

		lAlphaMatrix = []
		for idAlpha in range(self.numGap):
			lMatrixByEquation = []
			for idEqua in range(self.numGap):
				tmp_matrix = []
				marker = self.markerwithgap[idEqua]
				missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]
				for current_patch in range(self.numberPatch):
					# P current_patch in equation idEqua and alpha
					l = self.lenPatch * current_patch
					r = l + self.lenPatch
					Aj = self.AA[l:r, :]
					A0j = np.copy(Aj)
					A0j[missing_frame, marker*3 : marker*3+3] = 0
					U0N = self.list_UN0_matrix[idEqua]
					T_matrix = self.list_T_matrix[idEqua]
					UN = self.list_UN_matrix[idEqua]
					tmp = [A0j, U0N, T_matrix, UN.T]
					Pjg = matmul_list(tmp) 
					U0N_alpha = self.list_UN0_matrix[idAlpha]
					T_matrix_alpha = self.list_T_matrix[idAlpha]
					tmp = [Pjg.T, self.W, A0j, U0N_alpha, T_matrix_alpha, UN.T]
					tmp_matrix.append(matmul_list(tmp))
				lMatrixByEquation.append(summatrix_list(tmp_matrix))
			tmp = summatrix_list(lMatrixByEquation)
			xx, yy = tmp.shape
			matrix2vec = np.copy(tmp.reshape(xx*yy, 1))
			lAlphaMatrix.append(matrix2vec)
		
		left_form = np.hstack(lAlphaMatrix)
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		# print(self.list_alpha)
		return self.list_alpha

	def normalization(self):
		normed_matries, reconstruct_matries = compute_norm(self.combine_matrix)
		return normed_matries, reconstruct_matries

	def compute_beta(self):
		p_AN = self.AN
		p_AN0 = self.AN0
		list_A0 = []
		list_A = []

		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A.append(np.copy(p_AN[l:r]))
			list_A0.append(np.copy(p_AN0[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)

		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma))
		
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]
		self.list_Ti = []

		for patch_number in range(self.K):
			AiUN = np.matmul(list_A[patch_number], self.UN)
			Ai0UN0 = np.matmul(list_A0[patch_number], self.UN0)
			
			X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
			self.list_Ti.append(np.copy(X[0]))

		list_left_matrix = []
		for patch_number in range(self.K):
			current_patch = np.matmul(list_A[patch_number], self.UN) - matmul_list(
				[list_A0[patch_number], self.UN0, self.list_Ti[patch_number]])
			for column in range(ksmall):
				for clm in range(ksmall):
					tmp = np.multiply(current_patch[:, column], current_patch[:, clm])
					list_left_matrix.append(tmp)

		left_matrix = np.vstack(list_left_matrix)

		u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		self.Wtmp = np.diag(weight_list)
		# compute alpha
			
		list_Qjk = []
		for j in range(self.K):
			for h in range(self.K) :
				tmpQ = matmul_list([matmul_list([list_A0[j], self.UN0, self.list_Ti[h]]).T, 
					self.Wtmp, list_A[j], self.UN])
				list_Qjk.append(tmpQ)
		right_form = summatrix_list(list_Qjk)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)
		list_Pij_patch = []
		for patch_number in range(self.K):
			list_tmp = []
			for j in range(self.K):
				for h in range(self.K):
					tmpP = matmul_list([matmul_list([list_A0[j], self.UN0, self.list_Ti[h]]).T, 
						self.Wtmp, list_A0[j], self.UN0, self.list_Ti[patch_number]])
					list_tmp.append(tmpP)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pij_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pij_patch])
		# self.list_alpha = np.linalg.lstsq(left_form, right_form, rcond = None)[0]
		self.list_beta = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print(self.list_beta)
		return self.list_beta

	def interpolate_missing(self):
		list_matrix = []
		# for current_patch in range(self.numberPatch):
		for idMarker, marker in enumerate(self.markerwithgap):
			tmp = [self.AA,  self.list_UN0_matrix[idMarker], 
				self.list_T_matrix[idMarker], self.list_UN_matrix[idMarker].T]
			# interpolate = matmul_list(tmp) * (1.0 / self.numberPatch) * self.list_alpha[idMarker]
			# list_matrix.append(interpolate)
			interpolate = matmul_list(tmp) * self.list_alpha[idMarker]
			list_matrix.append(interpolate)
		result = self.m7 + (np.multiply( summatrix_list(list_matrix), self.m8) / self.m3) + self.MeanMat
		result = result[-self.lenPatch:,:]

		final_result = np.copy(self.missing_matrix)
		final_result[np.where(self.missing_matrix == 0)] = result[np.where(self.missing_matrix == 0)]
		return final_result
