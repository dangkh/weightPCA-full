import numpy as np
from algorithms.utils import *


def extractPatch(index, matrix, leng):
    l = index * leng
    r = l + leng
    return matrix[:, l:r]

def computeWeight(matrix, framewithgap, frame = None):
    result = np.ones(matrix.shape[0])
    firstMissing = framewithgap[0]
    tmp = firstMissing - 1
    while tmp >= 0:
        diff = matrix[firstMissing] - matrix[tmp]
        diff = np.square(diff)
        joint = 0
        sumDiff = 0
        while joint < len(diff) // 3:
            jointDiff = np.sqrt(np.sum(diff[joint*3:joint*3+3]))
            joint+= 1
            sumDiff += jointDiff
        result[tmp] = sumDiff
        tmp = tmp-1

    lastMissing = framewithgap[-1]
    tmp = lastMissing +1
    while tmp < matrix.shape[0]:
        diff = matrix[lastMissing] - matrix[tmp]
        diff = np.square(diff)
        joint = 0
        sumDiff = 0
        while joint < len(diff) // 3:
            jointDiff = np.sqrt(np.sum(diff[joint*3:joint*3+3]))
            joint+= 1
            sumDiff += jointDiff
        result[tmp] = sumDiff
        tmp = tmp+1

    weightScale = 50000
    weight_vector = np.exp(np.divide(-np.square(result),(2*np.square(weightScale))))
    weight_vector[firstMissing: lastMissing+1] = 0.001
    # if weight_vector is not None:
    #     weight_vector[frame] = 0.01
    # print(firstMissing, lastMissing+1)
    # print(frame)
    # print(weight_vector)
    # stop
    # weight_vector[firstMissing-1] = 1
    # weight_vector[lastMissing+1] = 1
    return weight_vector


class reconstructFrame():
    def __init__(self, weight, framewithgap, frame, inputN_nogap, inputN_zero, fullMat):
        N_nogap = inputN_nogap
        N_zero = inputN_zero

        mean_N_nogap = np.mean(N_nogap, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(fullMat, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(N_nogap, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1
        stdev = np.std(fullMat, 0)
        stdev[np.where(stdev == 0)] = 1

        weight_vector = np.copy(weight)
        weight_vector = weight_vector.reshape(-1,1)
        weight_factor = np.matmul(weight_vector, np.ones([1, inputN_nogap.shape[1]]))

        m4 = np.ones((N_nogap.shape[0], 1)) * mean_N_nogap
        m5 = np.ones((N_nogap.shape[0], 1)) * stdev_N_no_gaps

        N_nogap = np.multiply(((N_nogap - m4) / m5), weight_factor)
        N_zero = np.copy(N_nogap)
        N_zero[framewithgap, :] = 0


        Vmatrix, Sigma_Afull, _ = np.linalg.svd( np.matmul(N_nogap, N_nogap.T), full_matrices=False)
        VtiMatrix, Sigma_AtiFull, _ = np.linalg.svd( np.matmul(N_zero, N_zero.T), full_matrices=False)
        ksmall = max(get_zero(Sigma_Afull), get_zero(Sigma_AtiFull))
        Vmatrix = Vmatrix[:, :ksmall]
        VtiMatrix = Vmatrix[:, :ksmall]


        self.T_matrix = np.matmul(VtiMatrix.T , Vmatrix)


        # for construct:
        self.mean_N_zero = mean_N_zero
        self.mean_N_nogap = mean_N_nogap
        self.stdev_N_no_gaps = stdev_N_no_gaps
        self.stdev = stdev
        self.weight_vector = weight_vector
        self.Vmatrix = Vmatrix
        self.VtiMatrix = VtiMatrix

    def getPredict(self, matrix):
        stdev = np.std(matrix, 0)
        stdev[np.where(stdev == 0)] = 1
        mean_N_zero = np.mean(matrix, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        m1 = np.matmul(np.ones((matrix.shape[0],1)), mean_N_zero)
        m2 = np.ones((matrix.shape[0],1))* stdev
        weight_factor = np.matmul(self.weight_vector, np.ones([1, matrix.shape[1]]))
        M_zero = np.multiply(((np.copy(matrix)-m1) / m2),weight_factor)
        
        reconstructData = matmul_list([self.VtiMatrix, self.T_matrix, self.Vmatrix.T, M_zero])

        # reverse normalization
        m7 = np.ones((matrix.shape[0],1))* mean_N_zero
        m8 = np.ones((reconstructData.shape[0],1))* stdev

        reconstructData = m7 + np.multiply(reconstructData / weight_factor, m8)

        return reconstructData

class interpolation_LWPCA_wholeFrame():

    def __init__(self, reference_matrix, missing_matrix):
        self.fix_leng = missing_matrix.shape[0]

        listPatch = []
        self.numberPatch = reference_matrix.shape[0] // self.fix_leng
        for x in range(self.numberPatch):
            l = x * self.fix_leng
            r = l + self.fix_leng
            matrix = np.copy(reference_matrix[l:r])
            listPatch.append(matrix)
        self.listPatch = listPatch
        self.reference_matrix = np.hstack(listPatch)
        self.combine_matrix = np.hstack((np.copy(self.reference_matrix), np.copy(missing_matrix)))
        self.missing_matrix = missing_matrix
        self.info = self.prepare()
        # self.compute_weight()
        # self.compute_alpha()
        # self.result = self.interpolate_missing()

    def prepare(self, remove_patches=False, current_mean=-1):
        
        self.mask = []
        self.listWeight = []
        AA = np.copy(self.combine_matrix)
        columnindex = np.where(AA == 0)[1]
        columnwithgap = np.unique(columnindex)
        frameindex = np.where(AA == 0)[0]
        framewithgap = np.unique(frameindex)
        self.framewithgap = framewithgap
        markerwithgap = np.unique(columnwithgap // 3)
        self.markerwithgap = markerwithgap
        self.numGap = len(framewithgap)
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

        Afull = np.copy(self.AA[:, :-self.missing_matrix.shape[1]])
        AtiFull = np.copy(Afull)
        AtiFull[framewithgap, :] = 0
        
        mean_N_nogap = np.mean(Afull, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(self.AA, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(Afull, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

        tmp = np.ones(AA.shape[0])
        # compute without Q mask:
        self.listGapInfo = []
        self.listGapNoQ = []
        for frame in framewithgap:
            gapInfo = reconstructFrame(tmp, framewithgap, frame, Afull, AtiFull, self.AA )
            self.listGapNoQ.append(gapInfo)


        # compute Q mask:
        self.listGapInfo = []
        for frame in framewithgap:
            weight = computeWeight(Afull, framewithgap, frame)
            # weight here could be editable
            self.listWeight.append(weight)
            gapInfo = reconstructFrame(weight, framewithgap, frame, Afull, AtiFull, self.AA )
            self.listGapInfo.append(gapInfo)

    # def compute_weight(self):
    #     Bmatrix = np.zeros(self.missing_matrix.shape)
    #     # this was the last formula
    #     # for gth, frame in enumerate(self.framewithgap):
    #     #     for k_patch in range(self.numberPatch):
    #     #         l = k_patch * 123
    #     #         r = l + 123
    #     #         sample = self.AA[:,l:r]
    #     #         sample_g = np.copy(sample)
    #     #         sample_g[frame,:] = 0
    #     #         deduct = sample - self.listGapInfo[gth].getPredict(sample_g)
    #     #         Bmatrix = Bmatrix + deduct
    #     # Bmatrix = np.matmul(Bmatrix, Bmatrix.T)

    #     # this is the newest formula
    #     for k_patch in range(self.numberPatch):
    #         l = k_patch * 123
    #         r = l + 123
    #         sample = self.AA[:,l:r]
    #         sample_g = np.copy(sample)
    #         sample_g[self.framewithgap,:] = 0
    #         deduct = sample - self.listGapNoQ[0].getPredict(sample_g)
    #         Bmatrix = Bmatrix + deduct
    #     # Bmatrix = np.matmul(Bmatrix, Bmatrix.T)


    #     _, Sigma_B, _ = np.linalg.svd(Bmatrix)
    #     accumulateNum = setting_rank(Sigma_B)
    #     for x in range(len(Sigma_B)):
    #         if x > accumulateNum:
    #             Sigma_B[x] = 1
    #     Wii = 1.0 / Sigma_B
    #     value_array = np.ones(self.missing_matrix.shape[0])
    #     for x in range(len(Sigma_B)):
    #         value_array[x] = Wii[x]
    #     np.diag(value_array)
    #     # self.W = diag_matrix(Wii[0], self.missing_matrix.shape[0])
    #     self.W = np.diag(value_array)
    #     return

    # def compute_alpha(self):
    #     listBmatrix = []
    #     for alphaID in range(len(self.framewithgap)):
    #         listSubMatrix = []
    #         for patch in range(self.numberPatch):
    #             Hmatrix =





    #     lAlphaMatrix = []
    #     for idAlpha in range(self.numGap):
    #         lMatrixByEquation = []
    #         for idEqua in range(self.numGap):
    #             tmp_matrix = []

    #             # marker = self.markerwithgap[idEqua]
    #     #       missing_frame = np.where(self.missing_matrix[:, marker*3] == 0)[0]
    #             for j in range(self.numberPatch):
    #                 #           # P current_patch in equation idEqua and alpha
    #                 l = j * 123
    #                 r = l + 123
    #                 sample = self.AA[:,l:r]
    #                 sample_g = np.copy(sample)
    #                 sample_g[frame,:] = 0
    #                 Pjg = self.listGapInfo[idEqua].getPredict(sample_g)

    #     #             T_matrix_alpha = self.list_T_matrix[idAlpha]
    #                 remain = self.listGapInfo[idAlpha].getPredict(sample_g)
    #                 tmp = [Pjg.T, self.W, remain]
    #                 tmp_matrix.append(matmul_list(tmp))
    #             lMatrixByEquation.append(summatrix_list(tmp_matrix))
    #         tmp = summatrix_list(lMatrixByEquation)
    #         xx, yy = tmp.shape
    #         matrix2vec = np.copy(tmp.reshape(xx * yy, 1))
    #         lAlphaMatrix.append(matrix2vec)

    #     left_form = np.hstack(lAlphaMatrix)
    #     self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form),
    #                                       np.matmul(left_form.T, right_form), rcond=None)[0]
    #     # self.list_alpha = [1.0 / len(self.framewithgap)] * len(self.framewithgap)
    #     print(self.list_alpha)
    #     return self.list_alpha

    # def interpolate_missing(self):

    #     result = np.zeros(self.AA.shape)
    #     for gth in range(len(self.framewithgap)):
    #         sample = self.listGapInfo[gth].getPredict(self.AA) + self.MeanMat
    #         # tmp = sample - self.MeanMat
    #         # print(tmp[self.framewithgap])
    #         result += self.list_alpha[gth] * sample
    #     result = result[:, -123:]

    #     final_result = np.copy(self.missing_matrix)
    #     final_result[np.where(self.missing_matrix == 0)] = result[np.where(self.missing_matrix == 0)]
    #     return final_result
    #     