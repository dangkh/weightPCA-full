import numpy as np
import random
import math
from algorithms.utils import *

def createWeightCol(weight):
    weightScale = 200
    weight_vector = np.exp(np.divide(-np.square(weight),(2*np.square(weightScale))))
    return weight_vector

class Li1st_through1():

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
        self.reference_matrix = np.vstack(listPatch)
        self.combine_matrix = np.vstack((np.copy(self.reference_matrix), np.copy(missing_matrix)))
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
        # print(Data_without_gap.shape)
        # print(self.MeanMat.shape)
        # print(np.mean(Data_without_gap[:, z_index], 1).shape)
        AA = AA - self.MeanMat
        AA[np.where(self.combine_matrix == 0)] = 0
        self.AA = AA

        self.AA[:, columnwithgap] = self.AA[:, columnwithgap] * 0.2

        Afull = np.copy(self.AA[:-self.missing_matrix.shape[0], :])
        _, Sigma_Afull, Umatrix = np.linalg.svd(Afull / np.sqrt(Afull.shape[0] - 1), full_matrices=False)
        l2 = setting_rank(Sigma_Afull)
        Umatrix = Umatrix[:, :l2]
        # print(Umatrix)
        # print(l2)
        # np.savetxt("checkU.txt", Umatrix, fmt="%.2f")
        # stop
        Umatrix_2 = Umatrix[framewithgap, :]
        Umatrix_1 = np.delete(np.copy(Umatrix), framewithgap, 0)

        P1 = np.delete(np.copy(self.AA), framewithgap, 0)
        alpha = matmul_list([np.linalg.inv(np.matmul(Umatrix_1.T, Umatrix_1)), Umatrix_1.T, P1])
        P2 = np.matmul(Umatrix_2, alpha)
        stop
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
        tmp = np.copy(result)
        tmp[framewithgap, :] = tmp[framewithgap, :] / 0.2
        returnResult = tmp + self.MeanMat
        self.result = returnResult[:, -self.missing_matrix.shape[1]:]


class interpolation_WPCA_through():

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
        self.reference_matrix = np.vstack(listPatch)
        self.combine_matrix = np.vstack((np.copy(self.reference_matrix), np.copy(missing_matrix)))
        self.missing_matrix = missing_matrix
        returnResult = self.prepare()
        # returnResult = self.predict_ith_Marker([], 0, self.combine_matrix)
        self.result = returnResult[:, -self.missing_matrix.shape[1]:]

    def prepare(self, remove_patches=False, current_mean=-1):
        self.listA0 = []
        list_ai = []
        list_atii = []
        for x in range(self.numberPatch):
            Ai = self.listPatch[x]
            list_ai.append(np.mean(Ai, 0))
            
        ANew = np.vstack(list_ai)
        Cmatrix = np.matmul(ANew, ANew.T)

        list_AtiG = []
        list_Cg = []
        for g in range(41):
            AnewG = np.copy(ANew)
            AnewG[:, g*3:g*3+3] = 0
            list_AtiG.append(AnewG)
            list_Cg.append(np.matmul(AnewG, AnewG.T))

        Qmatrix = []
        for g in range(41):
            Qg = dist(self.reference_matrix, g)
            weight = createWeightCol(Qg)
            Qmatrix.append(weight)


        diagK = diag_matrix(1/self.numberPatch, self.numberPatch)

        Ch = Cmatrix - np.matmul(diagK, Cmatrix) - np.matmul(Cmatrix, diagK) - matmul_list([diagK, Cmatrix, diagK])
        Vmatrix, Sigma_Afull, _ = np.linalg.svd( Ch / np.sqrt(Ch.shape[0] - 1), full_matrices=False)
        
        list_Chg = []
        list_Vg = []
        for g in range(41):
            tmpC = list_Cg[g]
            tmpCh = tmpC - np.matmul(diagK, Cmatrix) - np.matmul(tmpC, diagK) - matmul_list([diagK, tmpC, diagK])    
            list_Chg.append(tmpCh)
            Vg, Sigma_Afull, _ = np.linalg.svd( tmpCh / np.sqrt(tmpCh.shape[0] - 1), full_matrices=False)
            list_Vg.append(Vg)

        list_Tmatrix = []
        for g in range(41):
            V = np.copy(Vmatrix)
            Vti = np.copy(list_Vg[g])
            T_matrix = np.matmul(Vti.T , V)
            list_Tmatrix.append(T_matrix)

        tmpLeft = matmul_list([Vmatrix.T, ANew, ANew.T, Vmatrix])
        leftMatrix = np.linalg.inv(tmpLeft)
        rightMatrix = np.matmul(Vmatrix.T, ANew)
        Hmatrix = np.matmul(leftMatrix, rightMatrix)
        listB = []
        for g in range(41):
            for patch in range(self.numberPatch):
                Ai = np.copy(list_ai[patch])
                Aig = np.copy(Ai)
                Aig[g*3:g*3+3] = 0
                rightMatrix = matmul_list([list_AtiG[g].T, list_Vg[g], list_Tmatrix[g], Hmatrix])
                sub = Ai - np.matmul(Aig.T, rightMatrix)
                listB.append(sub)
        Bmatrix = summatrix_list(listB)
        Bmatrix = Bmatrix.reshape(-1,1)
        print(Bmatrix.shape)
        _, Sigma_B , _ = np.linalg.svd(np.matmul(Bmatrix, Bmatrix.T))
        # accumulateNum = setting_rank(Sigma_B)
        for x in range(len(Sigma_B)):
            if x > 0:
                Sigma_B[x] = 1
        Wii = 1.0 / Sigma_B

        self.W = np.diag(Wii)
        print(self.W)
        stop
        

        columnindex = np.where(self.missing_matrix == 0)[1]
        columnwithgap = np.unique(columnindex)
        markerwithgap = np.unique(columnwithgap // 3)
        listAg = []
        listCg = []
        for idx in range(len(markerwithgap)):
            marker = markerwithgap[idx]
            tmp_list = []
            for patchIdx in range(self.numberPatch):
                Ai = self.listPatch[patchIdx]
                A0g = np.copy(Ai)
                A0g[:, marker * 3: marker * 3 + 3] = 0
                tmp_list.append(np.mean(A0g, 0))
            A0gNew = np.vstack(tmp_list)
            listAg.append(A0gNew)
            listCg.append(np.matmul(A0gNew, A0gNew.T))

        diag1_K = diag_matrix(1.0 / self.numberPatch, self.numberPatch)

        Cmark = Cmatrix - np.matmul(diag1_K,
                                    Cmatrix) - np.matmul(Cmatrix, diag1_K) + matmul_list([diag1_K, Cmatrix, diag1_K])
        C0mark = C0matrix - np.matmul(diag1_K,
                                    C0matrix) - np.matmul(C0matrix, diag1_K) + matmul_list([diag1_K, C0matrix, diag1_K])
        _, Sigma_Cmark , U_N_Cmark = np.linalg.svd(Cmark/np.sqrt(Cmark.shape[0]-1), full_matrices = False)
        _, Sigma_C0mark , U_N_C0mark = np.linalg.svd(C0mark/np.sqrt(C0mark.shape[0]-1), full_matrices = False)

        ksmall = max(setting_rank(Sigma_Cmark), setting_rank(Sigma_C0mark))
        
        U_N_Cmark = U_N_Cmark[:, :ksmall]
        U_N_C0mark = U_N_C0mark[:, :ksmall]
        listCgMark = []
        listUN_CgMark = []
        for idx, Cg in enumerate(listCg):
            Cgmark = Cg - np.matmul(diag1_K, Cg) - np.matmul(Cg, diag1_K) + matmul_list([diag1_K, Cg, diag1_K])
            listCgMark.append(Cgmark)
            _, Sigma_Cgmark , U_N_Cgmark = np.linalg.svd(Cgmark/np.sqrt(Cgmark.shape[0]-1), full_matrices = False)
            listUN_CgMark.append(U_N_Cgmark[:, :ksmall])
        
        # assuming all alphas are equal and == 1/K
        listTg = []
        for idx in range(len(markerwithgap)):
            tmpTg = np.matmul(U_N_Cmark.T, listUN_CgMark[0])
            listTg.append(tmpTg)
        # alphas = [1.0 / self.numberPatch] * self.numberPatch
        # print(ANew.shape)
        tmp1 = np.linalg.inv(matmul_list([U_N_Cmark.T, ANew, ANew.T, U_N_Cmark]))
        test = matmul_list([listAg[0].T, listUN_CgMark[0], listTg[0], tmp1, U_N_Cmark.T, ANew])
        print(test.shape)

        resTest = np.matmul(listAg[0], test)
        # print(resTest.shape)
        # stop
        # print(resTest)
        result = self.predict_ith_Marker(list_ai[0], 0, self.combine_matrix)
        return result
        # AA = np.copy(self.combine_matrix)
        # columnindex = np.where(AA == 0)[1]
        # columnwithgap = np.unique(columnindex)
        # frameindex = np.where(AA == 0)[0]
        # framewithgap = np.unique(frameindex)
        # self.framewithgap = framewithgap
        # markerwithgap = np.unique(columnwithgap // 3)
        # self.markerwithgap = markerwithgap
        # [frames, columns] = AA.shape
        # Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
        # # mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
        # columnWithoutGap = Data_without_gap.shape[1]

        # x_index = [x for x in range(0, columnWithoutGap, 3)]
        # mean_data_withoutgap_vecX = np.mean(Data_without_gap[:, x_index], 1).reshape(frames, 1)

        # y_index = [x for x in range(1, columnWithoutGap, 3)]
        # mean_data_withoutgap_vecY = np.mean(Data_without_gap[:, y_index], 1).reshape(frames, 1)

        # z_index = [x for x in range(2, columnWithoutGap, 3)]
        # mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:, z_index], 1).reshape(frames, 1)

        # joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
        # self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1] // 3)
        # AA = AA - self.MeanMat
        # AA[np.where(self.combine_matrix == 0)] = 0
        # self.AA = AA

        # self.AA[framewithgap, :] = self.AA[framewithgap, :] * 0.2
        # # multiplize with weight

        # Afull = np.copy(self.AA[:, :-self.missing_matrix.shape[1]])
        # Umatrix, Sigma_Afull, _ = np.linalg.svd(
        #     np.matmul(Afull, Afull.T) / np.sqrt(np.matmul(Afull, Afull.T).shape[0] - 1), full_matrices=False)
        # l2 = setting_rank(Sigma_Afull)
        # Umatrix = Umatrix[:, :l2]

        # Umatrix_2 = Umatrix[framewithgap, :]
        # Umatrix_1 = np.delete(np.copy(Umatrix), framewithgap, 0)

        # Bmatrix = np.zeros(self.missing_matrix[framewithgap, :].shape)
        # for idx in range(self.numberPatch):
        #     Ai2 = self.listPatch[idx][framewithgap, :]
        #     Ai1 = np.delete(self.listPatch[idx], framewithgap, 0)
        #     reconstructMatrix = matmul_list([Umatrix_2, np.linalg.inv(
        #         np.matmul(Umatrix_1.T, Umatrix_1)), Umatrix_1.T, Ai1])
        #     Bmatrix = Bmatrix + Ai2 - reconstructMatrix

        # _, Sigma_B, _ = np.linalg.svd(Bmatrix)
        # # _, Sigma_B , _ = np.linalg.svd(np.matmul(Bmatrix.T, Bmatrix))
        # # accumulateNum = setting_rank(Sigma_B)
        # for x in range(len(Sigma_B)):
        #     if x > 0:
        #         Sigma_B[x] = 1
        # Wii = 1.0 / Sigma_B
        # tmp = self.listPatch[0][framewithgap, :]
        # self.W = diag_matrix(Wii[0], tmp.shape[0])

        # listF = []
        # listC = []
        # alphaMatrix = np.zeros((len(framewithgap), len(framewithgap)))
        # HTMatrix = np.linalg.inv(np.matmul(Umatrix.T, Umatrix))
        # for gapIdx in range(len(framewithgap)):
        #     alphaMatrix[gapIdx, gapIdx] = 1
        #     oneGMatrix = np.ones((len(framewithgap), self.missing_matrix.shape[1]))
        #     listMatrix = []
        #     for i in range(self.numberPatch):
        #         Ai2 = self.listPatch[i][framewithgap, :]
        #         listTmp = [Umatrix_2, HTMatrix, Umatrix_2.T, alphaMatrix, oneGMatrix]
        #         listLeft = [matmul_list(listTmp).T, self.W, Ai2]
        #         listMatrix.append(matmul_list(listLeft))

        #     B1 = summatrix_list(listMatrix)
        #     listMatrix = []
        #     for i in range(self.numberPatch):
        #         Ai1 = np.delete(self.listPatch[i], framewithgap, 0)
        #         listLeft = [Umatrix_2, HTMatrix, Umatrix_2.T, alphaMatrix, oneGMatrix]
        #         listRight = [Umatrix_2, HTMatrix, Umatrix_1.T, Ai1]
        #         listMulMatrix = [matmul_list(listLeft).T, self.W, matmul_list(listRight)]

        #         listMatrix.append(matmul_list(listMulMatrix))

        #     B2 = summatrix_list(listMatrix)
        #     Bmatrix = B1 - B2
        #     listCmatrix1 = matmul_list([Umatrix_2, HTMatrix, Umatrix_2.T, alphaMatrix, oneGMatrix])
        #     listCmatrix2 = matmul_list([self.W, Umatrix_2, HTMatrix, Umatrix_2.T])
        #     Cmatrix = self.numberPatch * np.matmul((listCmatrix1).T, (listCmatrix2))
        #     alphaMatrix[gapIdx, gapIdx] = 0
        #     tmpMatrix = np.asarray([Bmatrix[x, 1] for x in range(self.missing_matrix.shape[1])])
        #     FMatrix = np.reshape(tmpMatrix, (-1, 1))
        #     listC.append(Cmatrix)
        #     listF.append(FMatrix)
        # finalC = np.vstack(listC)
        # finalF = np.vstack(listF)
        # alpha = matmul_list([np.linalg.inv(np.matmul(finalC.T, finalC)), finalC.T, finalF])
        # finalAlpha = np.diag(alpha[:, 0])
        # oneGMatrix = np.ones((len(framewithgap), self.AA.shape[1]))
        # P1 = np.delete(np.copy(self.AA), framewithgap, 0)
        # P2_1 = matmul_list([Umatrix_2, HTMatrix, Umatrix_2.T, finalAlpha, oneGMatrix])
        # P2_2 = matmul_list([Umatrix_2, HTMatrix, Umatrix_1.T, P1])
        # P2 = P2_1 + P2_2
        # result = np.zeros(self.AA.shape)
        # l1Counter = 0
        # l2Counter = 0
        # for x in range(self.AA.shape[0]):
        #     if x not in framewithgap:
        #         result[x, :] = P1[l1Counter, :]
        #         l1Counter += 1
        #     else:
        #         result[x, :] = P2[l2Counter, :]
        #         l2Counter += 1
        # tmp = np.copy(result)
        # tmp[framewithgap, :] = tmp[framewithgap, :] / 0.2
        # returnResult = tmp + self.MeanMat
        # self.result = returnResult[:, -self.missing_matrix.shape[1]:]

    def predict_ith_Marker(self, mean_vec, ithMarker, inputdata):
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
        mean_data_withoutgap_vecX = np.mean(Data_without_gap[:, x_index], 1).reshape(frames, 1)

        y_index = [x for x in range(1, columnWithoutGap, 3)]
        mean_data_withoutgap_vecY = np.mean(Data_without_gap[:, y_index], 1).reshape(frames, 1)

        z_index = [x for x in range(2, columnWithoutGap, 3)]
        mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:, z_index], 1).reshape(frames, 1)

        joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
        MeanMat = np.tile(joint_meanXYZ, combine_matrix.shape[1] // 3)
        Data = np.copy(combine_matrix - MeanMat)
        Data[np.where(combine_matrix == 0)] = 0

        weight_vector = compute_weight_vect_norm(markerwithgap, Data)
        weight_vector = np.exp(np.divide(-np.square(weight_vector), (2 * np.square(weightScale))))
        weight_vector[markerwithgap] = MMweight
        M_zero = np.copy(Data)

        N_nogap = np.delete(Data, framewithgap, 0)
        N_zero = np.copy(N_nogap)
        N_zero[:, columnwithgap] = 0
        mean_N_nogap = np.mean(N_nogap, 0)
        mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

        mean_N_zero = np.mean(N_zero, 0)
        mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
        stdev_N_no_gaps = np.std(N_nogap, 0)
        stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

        m1 = np.matmul(np.ones((M_zero.shape[0], 1)), mean_N_zero)
        m2 = np.ones((M_zero.shape[0], 1)) * stdev_N_no_gaps

        column_weight = np.ravel(np.ones((3, 1)) * weight_vector, order='F')
        column_weight = column_weight.reshape((1, column_weight.shape[0]))
        m3 = np.matmul(np.ones((M_zero.shape[0], 1)), column_weight)
        m33 = np.matmul(np.ones((N_nogap.shape[0], 1)), column_weight)
        m4 = np.ones((N_nogap.shape[0], 1)) * mean_N_nogap
        m5 = np.ones((N_nogap.shape[0], 1)) * stdev_N_no_gaps
        m6 = np.ones((N_zero.shape[0], 1)) * mean_N_zero

        M_zero = np.multiply(((M_zero - m1) / m2), m3)
        N_nogap = np.multiply(((N_nogap - m4) / m5), m33)
        N_zero = np.multiply(((N_zero - m6) / m5), m33)

        _, Sigma_nogap, U_N_nogap_VH = np.linalg.svd(N_nogap / np.sqrt(N_nogap.shape[0] - 1), full_matrices=False)
        U_N_nogap = U_N_nogap_VH.T
        print(U_N_nogap.shape)
        _, Sigma_zero, U_N_zero_VH = np.linalg.svd(N_zero / np.sqrt(N_zero.shape[0] - 1), full_matrices=False)
        U_N_zero = U_N_zero_VH.T
        print(U_N_zero.shape)
        ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
        U_N_nogap = U_N_nogap[:, :ksmall]
        U_N_zero = U_N_zero[:, :ksmall]
        T_matrix = np.matmul(U_N_nogap.T, U_N_zero)
        reconstructData = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)

        # reverse normalization
        m7 = np.ones((Data.shape[0], 1)) * mean_vec
        m8 = np.ones((reconstructData.shape[0], 1)) * stdev_N_no_gaps
        m3 = np.matmul(np.ones((M_zero.shape[0], 1)), column_weight)
        reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
        tmp = reconstructData + MeanMat
        result = np.copy(tmp[-inputdata.shape[0]:, :])
        final_result = np.copy(inputdata)
        final_result[np.where(inputdata == 0)] = result[np.where(inputdata == 0)]
        return final_result
