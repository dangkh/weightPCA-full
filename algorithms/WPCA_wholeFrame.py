import numpy as np
import random
import math
from algorithms.utils import *


class Li1st_wholeFrame():

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

    def prepare(self, remove_patches=False, current_mean=-1):
        AA = np.copy(self.combine_matrix)
        columnindex = np.where(AA == 0)[1]
        columnwithgap = np.unique(columnindex)
        frameindex = np.where(AA == 0)[0]
        framewithgap = np.unique(frameindex)
        self.framewithgap = framewithgap
        markerwithgap = np.unique(columnwithgap // 3)
        self.markerwithgap = markerwithgap
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

        m4 = np.ones((Afull.shape[0], 1)) * mean_N_nogap
        m5 = np.ones((Afull.shape[0], 1)) * stdev_N_no_gaps

        Afull = (Afull - m4) / m5
        Afull[framewithgap, :] = Afull[framewithgap, :] * 0.001
        # multiplize with weight

        AtiFull = np.copy(Afull)
        AtiFull[framewithgap, :] = 0

        m1 = np.matmul(np.ones((self.AA.shape[0],1)), mean_N_zero)
        stdev = np.std(self.AA, 0)
        stdev[np.where(stdev == 0)] = 1
        m2 = np.ones((self.AA.shape[0],1))* stdev
        self.AA = ((np.copy(self.AA)-m1) / m2)
        self.AA[framewithgap, :] = self.AA[framewithgap, :] * 1
        

        Vmatrix, Sigma_Afull, _ = np.linalg.svd( Afull / np.sqrt(Afull.shape[0] - 1), full_matrices=False)

        VtiMatrix, Sigma_AtiFull, _ = np.linalg.svd( AtiFull / np.sqrt(AtiFull.shape[0] - 1), full_matrices=False)
        ksmall = max(get_zero(Sigma_Afull), get_zero(Sigma_AtiFull))
        Vmatrix = Vmatrix[:, :ksmall]
        VtiMatrix = Vmatrix[:, :ksmall]


        self.T_matrix = np.matmul(VtiMatrix.T , Vmatrix)
        listMatrix = [VtiMatrix, self.T_matrix, Vmatrix.T, self.AA]
        reconstructData = matmul_list(listMatrix)
       
        m7 = np.ones((self.AA.shape[0],1))* mean_N_zero
        m8 = np.ones((reconstructData.shape[0],1))* stdev
        reconstructData[framewithgap, :] = reconstructData[framewithgap, :] / 0.001
        # print(m7.shape, m8.shape, reconstructData.shape)
        reconstructData = m7 + np.multiply(reconstructData, m8)


        returnResult = reconstructData + self.MeanMat
        self.result = returnResult[:, -self.missing_matrix.shape[1]:]


def extractPatch(index, matrix, leng):
    l = index * leng
    r = l + leng
    return matrix[:, l:r]

def computeWeight(matrix, framewithgap):
    result = np.ones(matrix.shape[0])
    firstMissing = framewithgap[0]
    tmp = firstMissing - 1
    while tmp >= 0:
        diff = matrix[firstMissing] - matrix[tmp]
        diff = np.square(diff)
        joint = 0
        sumDiff = 0
        while joint < len(diff):
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
        while joint < len(diff):
            jointDiff = np.sqrt(np.sum(diff[joint*3:joint*3+3]))
            joint+= 1
            sumDiff += jointDiff
        result[tmp] = sumDiff
        tmp = tmp+1

    weightScale = 1000
    weight_vector = np.exp(np.divide(-np.square(result),(2*np.square(weightScale))))
    weight_vector[firstMissing-1: lastMissing+1] = 1
    return weight_vector

class interpolation_WPCA_wholeFrame():

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

    def prepare(self, remove_patches=False, current_mean=-1):
        AA = np.copy(self.combine_matrix)
        columnindex = np.where(AA == 0)[1]
        columnwithgap = np.unique(columnindex)
        frameindex = np.where(AA == 0)[0]
        framewithgap = np.unique(frameindex)
        self.framewithgap = framewithgap
        markerwithgap = np.unique(columnwithgap // 3)
        self.markerwithgap = markerwithgap
        [frames, columns] = AA.shape
        Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
        # mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
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

        m4 = np.ones((Afull.shape[0], 1)) * mean_N_nogap
        m5 = np.ones((Afull.shape[0], 1)) * stdev_N_no_gaps

        Afull = (Afull - m4) / m5
        Afull[framewithgap, :] = Afull[framewithgap, :] * 0.01
        # multiplize with weight

        AtiFull = np.copy(Afull)
        AtiFull[framewithgap, :] = 0

        m1 = np.matmul(np.ones((self.AA.shape[0],1)), mean_N_zero)
        stdev = np.std(self.AA, 0)
        stdev[np.where(stdev == 0)] = 1
        m2 = np.ones((self.AA.shape[0],1))* stdev
        self.AA = ((np.copy(self.AA)-m1) / m2)
        self.AA[framewithgap, :] = self.AA[framewithgap, :] * 0.01
        

        Vmatrix, Sigma_Afull, _ = np.linalg.svd( Afull / np.sqrt(Afull.shape[0] - 1), full_matrices=False)

        VtiMatrix, Sigma_AtiFull, _ = np.linalg.svd( AtiFull / np.sqrt(AtiFull.shape[0] - 1), full_matrices=False)
        ksmall = max(get_zero(Sigma_Afull), get_zero(Sigma_AtiFull))
        Vmatrix = Vmatrix[:, :ksmall]
        VtiMatrix = Vmatrix[:, :ksmall]


        # Vmatrix_2 = Vmatrix[framewithgap, :]
        # Vmatrix_1 = np.delete(np.copy(Vmatrix), framewithgap, 0)

        VtiMatrix_2 = VtiMatrix[framewithgap, :]
        VtiMatrix_1 = np.delete(np.copy(VtiMatrix), framewithgap, 0)


        self.T_matrix = np.matmul(VtiMatrix.T , Vmatrix)

        Bmatrix = np.zeros(self.missing_matrix[framewithgap, :].shape)
        for idx in range(self.numberPatch):
            
            Ai = extractPatch(idx, Afull, self.missing_matrix.shape[1])

            Ai2 = np.copy(Ai[framewithgap,:])
            Ai1 = np.delete(np.copy(Ai), framewithgap, 0)
            
            listMat = [self.T_matrix.T, VtiMatrix_1.T, VtiMatrix_1, self.T_matrix]
            invMatrix = np.linalg.inv(matmul_list(listMat))

            listMat = [VtiMatrix_2, self.T_matrix, invMatrix, self.T_matrix.T, VtiMatrix_1.T, Ai1]

            reconstructMatrix = matmul_list(listMat)

            Bmatrix = Bmatrix + Ai2 - reconstructMatrix

        _, Sigma_B, _ = np.linalg.svd(Bmatrix)
        # _, Sigma_B , _ = np.linalg.svd(np.matmul(Bmatrix.T, Bmatrix))
        # accumulateNum = setting_rank(Sigma_B)
        for x in range(len(Sigma_B)):
            if x > 0:
                Sigma_B[x] = 1
        Wii = 1.0 / Sigma_B

        self.W = np.diag(Wii)
        listF = []
        listC = []
        alphaMatrix = np.zeros((len(framewithgap), len(framewithgap)))
        listMat = [self.T_matrix.T, VtiMatrix.T, VtiMatrix, self.T_matrix]
        invMatrix = np.linalg.inv(matmul_list(listMat))
        HTMatrix = matmul_list([self.T_matrix, invMatrix, self.T_matrix.T])

        for gapIdx in range(len(framewithgap)):
            alphaMatrix[gapIdx, gapIdx] = 1
            oneGMatrix = np.ones((len(framewithgap), self.missing_matrix.shape[1]))
            listMatrix = []
            for i in range(self.numberPatch):
                Ai = extractPatch(idx, Afull, self.missing_matrix.shape[1])
                Ai2 = np.copy(Ai[framewithgap,:])

                listTmp = [VtiMatrix_2, HTMatrix, VtiMatrix_2.T, alphaMatrix, oneGMatrix]
                listLeft = [matmul_list(listTmp).T, self.W, Ai2]
                
                listMatrix.append(matmul_list(listLeft))

            B1 = summatrix_list(listMatrix)
            listMatrix = []

            for i in range(self.numberPatch):
                Ai = extractPatch(idx, Afull, self.missing_matrix.shape[1])
                Ai1 = np.delete(np.copy(Ai), framewithgap, 0)

                listLeft = [VtiMatrix_2, HTMatrix, VtiMatrix_2.T, alphaMatrix, oneGMatrix]
                listRight = [VtiMatrix_2, HTMatrix, VtiMatrix_1.T, Ai1]
                listMulMatrix = [matmul_list(listLeft).T, self.W, matmul_list(listRight)]

                listMatrix.append(matmul_list(listMulMatrix))

            B2 = summatrix_list(listMatrix)
            Bmatrix = B1 - B2
            
            listCmatrix1 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_2.T, alphaMatrix, oneGMatrix])
            listCmatrix2 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_2.T])

            Cmatrix = self.numberPatch * matmul_list([listCmatrix1.T, self.W, listCmatrix2])
            alphaMatrix[gapIdx, gapIdx] = 0

            tmpMatrix = np.asarray([Bmatrix[x, 1] for x in range(self.missing_matrix.shape[1])])
            FMatrix = np.reshape(tmpMatrix, (-1, 1))
            listC.append(Cmatrix)
            listF.append(FMatrix)
        finalC = np.vstack(listC)
        finalF = np.vstack(listF)
        alpha = matmul_list([np.linalg.inv(np.matmul(finalC.T, finalC)), finalC.T, finalF])
        finalAlpha = np.diag(alpha[:, 0])
        oneGMatrix = np.ones((len(framewithgap), self.AA.shape[1]))
        
        P1 = np.delete(np.copy(self.AA), framewithgap, 0)
        P2_1 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_2.T, finalAlpha, oneGMatrix])
        P2_2 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_1.T, P1])
        P2 = P2_1 + P2_2

        result = np.zeros(self.AA.shape)
        l1Counter = 0
        l2Counter = 0
        for x in range(self.AA.shape[0]):
            if x not in framewithgap:
                result[x, :] = P1[l1Counter, :]
                l1Counter += 1
            else:
                result[x, :] = P2[l2Counter, :]
                l2Counter += 1
        reconstructData = np.copy(result)

        
        # reverse normalization
        m7 = np.ones((self.AA.shape[0],1))* mean_N_zero
        m8 = np.ones((reconstructData.shape[0],1))* stdev
        reconstructData[framewithgap, :] = reconstructData[framewithgap, :] / 0.01
        # print(m7.shape, m8.shape, reconstructData.shape)
        reconstructData = m7 + np.multiply(reconstructData, m8)


        returnResult = reconstructData + self.MeanMat
        self.result = returnResult[:, -self.missing_matrix.shape[1]:]


class interpolation_MaskWPCA_wholeFrame():
    # assuming each alphaGs are equal, treat LWPCA whole Frame as Mask + WPCA
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

    def prepare(self, remove_patches=False, current_mean=-1):
        AA = np.copy(self.combine_matrix)
        columnindex = np.where(AA == 0)[1]
        columnwithgap = np.unique(columnindex)
        frameindex = np.where(AA == 0)[0]
        framewithgap = np.unique(frameindex)
        self.framewithgap = framewithgap
        markerwithgap = np.unique(columnwithgap // 3)
        self.markerwithgap = markerwithgap
        [frames, columns] = AA.shape
        Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
        # mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
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
        weight_vector = computeWeight(Afull, framewithgap)
        weight_vector = weight_vector.reshape(-1,1)
        # print(weight_vector)
        # stop
        weight_factor = np.matmul(weight_vector, np.ones([1, Afull.shape[1]]))
        AtiFull[framewithgap, :] = 0
        
        mean_N_nogap = np.mean(Afull, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(self.AA, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(Afull, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

        m4 = np.ones((Afull.shape[0], 1)) * mean_N_nogap
        m5 = np.ones((Afull.shape[0], 1)) * stdev_N_no_gaps

        Afull = (Afull - m4) / m5 
        Afull[framewithgap, :] = Afull[framewithgap, :] * 0.001
        Afull = Afull * weight_factor
        # multiplize with weight

        AtiFull = np.copy(Afull)
        AtiFull[framewithgap, :] = 0

        m1 = np.matmul(np.ones((self.AA.shape[0],1)), mean_N_zero)
        stdev = np.std(self.AA, 0)
        stdev[np.where(stdev == 0)] = 1
        m2 = np.ones((self.AA.shape[0],1))* stdev
        self.AA = ((np.copy(self.AA)-m1) / m2)
        self.AA[framewithgap, :] = self.AA[framewithgap, :] * 0.001
        weight_factor = np.matmul(weight_vector, np.ones([1, self.AA.shape[1]]))
        self.AA = self.AA * weight_factor
        

        Vmatrix, Sigma_Afull, _ = np.linalg.svd( Afull , full_matrices=False)

        VtiMatrix, Sigma_AtiFull, _ = np.linalg.svd( AtiFull , full_matrices=False)
        ksmall = max(get_zero(Sigma_Afull), get_zero(Sigma_AtiFull))
        Vmatrix = Vmatrix[:, :ksmall]
        VtiMatrix = Vmatrix[:, :ksmall]


        # Vmatrix_2 = Vmatrix[framewithgap, :]
        # Vmatrix_1 = np.delete(np.copy(Vmatrix), framewithgap, 0)

        VtiMatrix_2 = VtiMatrix[framewithgap, :]
        VtiMatrix_1 = np.delete(np.copy(VtiMatrix), framewithgap, 0)


        self.T_matrix = np.matmul(VtiMatrix.T , Vmatrix)

        Bmatrix = np.zeros(self.missing_matrix[framewithgap, :].shape)
        for idx in range(self.numberPatch):
            
            Ai = extractPatch(idx, Afull, self.missing_matrix.shape[1])

            Ai2 = np.copy(Ai[framewithgap,:])
            Ai1 = np.delete(np.copy(Ai), framewithgap, 0)
            
            listMat = [self.T_matrix.T, VtiMatrix_1.T, VtiMatrix_1, self.T_matrix]
            invMatrix = np.linalg.inv(matmul_list(listMat))

            listMat = [VtiMatrix_2, self.T_matrix, invMatrix, self.T_matrix.T, VtiMatrix_1.T, Ai1]

            reconstructMatrix = matmul_list(listMat)

            Bmatrix = Bmatrix + Ai2 - reconstructMatrix

        _, Sigma_B, _ = np.linalg.svd(Bmatrix)
        # _, Sigma_B , _ = np.linalg.svd(np.matmul(Bmatrix.T, Bmatrix))
        # accumulateNum = setting_rank(Sigma_B)
        for x in range(len(Sigma_B)):
            if x > 0:
                Sigma_B[x] = 1
        Wii = 1.0 / Sigma_B

        self.W = np.diag(Wii)
        listF = []
        listC = []
        alphaMatrix = np.zeros((len(framewithgap), len(framewithgap)))
        listMat = [self.T_matrix.T, VtiMatrix.T, VtiMatrix, self.T_matrix]
        invMatrix = np.linalg.inv(matmul_list(listMat))
        HTMatrix = matmul_list([self.T_matrix, invMatrix, self.T_matrix.T])

        for gapIdx in range(len(framewithgap)):
            alphaMatrix[gapIdx, gapIdx] = 1
            oneGMatrix = np.ones((len(framewithgap), self.missing_matrix.shape[1]))
            listMatrix = []
            for i in range(self.numberPatch):
                Ai = extractPatch(idx, Afull, self.missing_matrix.shape[1])
                Ai2 = np.copy(Ai[framewithgap,:])

                listTmp = [VtiMatrix_2, HTMatrix, VtiMatrix_2.T, alphaMatrix, oneGMatrix]
                listLeft = [matmul_list(listTmp).T, self.W, Ai2]
                
                listMatrix.append(matmul_list(listLeft))

            B1 = summatrix_list(listMatrix)
            listMatrix = []

            for i in range(self.numberPatch):
                Ai = extractPatch(idx, Afull, self.missing_matrix.shape[1])
                Ai1 = np.delete(np.copy(Ai), framewithgap, 0)

                listLeft = [VtiMatrix_2, HTMatrix, VtiMatrix_2.T, alphaMatrix, oneGMatrix]
                listRight = [VtiMatrix_2, HTMatrix, VtiMatrix_1.T, Ai1]
                listMulMatrix = [matmul_list(listLeft).T, self.W, matmul_list(listRight)]

                listMatrix.append(matmul_list(listMulMatrix))

            B2 = summatrix_list(listMatrix)
            Bmatrix = B1 - B2
            
            listCmatrix1 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_2.T, alphaMatrix, oneGMatrix])
            listCmatrix2 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_2.T])

            Cmatrix = self.numberPatch * matmul_list([listCmatrix1.T, self.W, listCmatrix2])
            alphaMatrix[gapIdx, gapIdx] = 0

            tmpMatrix = np.asarray([Bmatrix[x, 1] for x in range(self.missing_matrix.shape[1])])
            FMatrix = np.reshape(tmpMatrix, (-1, 1))
            listC.append(Cmatrix)
            listF.append(FMatrix)
        finalC = np.vstack(listC)
        finalF = np.vstack(listF)
        alpha = matmul_list([np.linalg.inv(np.matmul(finalC.T, finalC)), finalC.T, finalF])
        finalAlpha = np.diag(alpha[:, 0])
        oneGMatrix = np.ones((len(framewithgap), self.AA.shape[1]))
        
        P1 = np.delete(np.copy(self.AA), framewithgap, 0)
        P2_1 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_2.T, finalAlpha, oneGMatrix])
        P2_2 = matmul_list([VtiMatrix_2, HTMatrix, VtiMatrix_1.T, P1])
        P2 = P2_1 + P2_2

        result = np.zeros(self.AA.shape)
        l1Counter = 0
        l2Counter = 0
        for x in range(self.AA.shape[0]):
            if x not in framewithgap:
                result[x, :] = P1[l1Counter, :]
                l1Counter += 1
            else:
                result[x, :] = P2[l2Counter, :]
                l2Counter += 1
        reconstructData = np.copy(result)

        
        # reverse normalization
        m7 = np.ones((self.AA.shape[0],1))* mean_N_zero
        m8 = np.ones((reconstructData.shape[0],1))* stdev
        reconstructData[framewithgap, :] = reconstructData[framewithgap, :] / 0.001
        # weight_factor = np.matmul(weight_vector, np.ones([1, reconstructData.shape[1]]))
        # print(m7.shape, m8.shape, reconstructData.shape)
        reconstructData = m7 + np.multiply(reconstructData / weight_factor, m8)


        returnResult = reconstructData + self.MeanMat
        self.result = returnResult[:, -self.missing_matrix.shape[1]:]

    