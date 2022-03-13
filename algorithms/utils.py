import numpy as np


def diag_matrix(value, k):
    value_array = [value] * k
    return np.diag(value_array)


def matmul_list(matrix_list, db=False):
    if db:
        for x in matrix_list:
            print(x.shape)
    number_matrix = len(matrix_list)
    result = np.copy(matrix_list[0])
    for i in range(1, number_matrix):
        result = np.matmul(result, matrix_list[i])
    return result


def summatrix_list(matrix_list):
    number_matrix = len(matrix_list)
    result = np.copy(matrix_list[0])
    for i in range(1, number_matrix):
        result = result + matrix_list[i]
    return result


def MSE(A, B, missing_map):
    return np.sum(np.abs(A - B) * (1 - missing_map)) / np.sum(missing_map)


def ARE(predict, original):
    return np.mean(np.abs((predict - original) / original))


def mysvd(dataMat):
    U, Sigma, VT = np.linalg.svd(dataMat)
    return U


def remove_joint(data):
    list_del = []
    list_del_joint = [5, 9, 14, 18]

    for x in list_del_joint:
        list_del.append(x * 3)
        list_del.append(x * 3 + 1)
        list_del.append(x * 3 + 2)
    data = np.delete(data, list_del, 1)
    #print('removed joints data', data.shape)
    return data


def euclid_dist(X, Y):
    XX = np.asarray(X)
    YY = np.asarray(Y)
    return np.sqrt(np.sum(np.square(XX - YY)))


def get_zero(matrix):
    counter = 0
    for x in matrix:
        if x > 0.01:
            counter += 1
    return counter


def get_point(Data, frame, joint):
    point = [Data[frame, joint * 3], Data[frame, joint * 3 + 1], Data[frame, joint * 3 + 2]]
    return point


def read_tracking_data3D(data_dir, patch):
    print("reading source: ", data_dir, " patch: ", patch)

    Tracking3D = []
    f = open(data_dir, 'r')
    for line in f:
        elements = line.split(',')
        Tracking3D.append(list(map(float, elements)))
    f.close()

    Tracking3D = np.array(Tracking3D)  # list can not read by index while arr can be
    Tracking3D = np.squeeze(Tracking3D)
    #print('original data', Tracking3D.shape)

    Tracking3D = Tracking3D.astype(float)
    Tracking3D = Tracking3D[patch[0]: patch[1]]
    #print('patch data', Tracking3D.shape)

    Tracking3D = remove_joint(Tracking3D)
    restore = np.copy(Tracking3D)
    return Tracking3D, restore


def read_tracking_data3D_without_RJ(data_dir, patch):
    #print("reading source: ", data_dir, " patch: ", patch)

    Tracking3D = []
    f = open(data_dir, 'r')
    for line in f:
        elements = line.split(' ')
        Tracking3D.append(list(map(float, elements)))
    f.close()

    Tracking3D = np.array(Tracking3D)  # list can not read by index while arr can be
    Tracking3D = np.squeeze(Tracking3D)
    #print('original data', Tracking3D.shape)

    Tracking3D = Tracking3D.astype(float)
    Tracking3D = Tracking3D[patch[0]: patch[1]]
    #print('patch data', Tracking3D.shape)

    restore = np.copy(Tracking3D)
    return Tracking3D, restore


def setting_rank(eigen_vector):
    minCumSV = 0.99
    current_sum = 0
    sum_list = np.sum(eigen_vector)
    for x in range(len(eigen_vector)):
        current_sum += eigen_vector[x]
        if current_sum > minCumSV * sum_list:
            return x + 1
    return len(eigen_vector)


def get_homogeneous_solve(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def solution(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))
    print(e_vals)
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)]


def compute_norm(combine_matrix, downsample=False, gap_strategies=False, testMean = False):
    AA = np.copy(combine_matrix)
    weightScale = 200
    MMweight = 0.02
    [frames, columns] = AA.shape
    columnindex = np.where(AA == 0)[1]
    frameindex = np.where(AA == 0)[0]
    columnwithgap = np.unique(columnindex)
    markerwithgap = np.unique(columnwithgap // 3)
    framewithgap = np.unique(frameindex)
    Data_without_gap = np.delete(AA, columnwithgap, 1)
    mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
    columnWithoutGap = Data_without_gap.shape[1]

    x_index = [x for x in range(0, columnWithoutGap, 3)]
    mean_data_withoutgap_vecX = np.mean(Data_without_gap[:, x_index], 1).reshape(frames, 1)

    y_index = [x for x in range(1, columnWithoutGap, 3)]
    mean_data_withoutgap_vecY = np.mean(Data_without_gap[:, y_index], 1).reshape(frames, 1)

    z_index = [x for x in range(2, columnWithoutGap, 3)]
    mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:, z_index], 1).reshape(frames, 1)

    joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
    MeanMat = np.tile(joint_meanXYZ, AA.shape[1] // 3)
    Data = np.copy(AA - MeanMat)
    Data[np.where(AA == 0)] = 0

    # calculate weight vector
    frame_range_start = 0
    if downsample:
        frame_range_start = max(frames - 400 - len(framewithgap), 0)

    weight_matrix = np.zeros((frames, columns // 3))
    weight_matrix_coe = np.zeros((frames, columns // 3))
    weight_vector = np.zeros((len(markerwithgap), columns // 3))
    for x in range(len(markerwithgap)):
        weight_matrix = np.zeros((frames, columns // 3))
        weight_matrix_coe = np.zeros((frames, columns // 3))
        for i in range(frame_range_start, frames):
            valid = True
            if euclid_dist([0, 0, 0], get_point(Data, i, markerwithgap[x])) == 0:
                valid = False
            if valid:
                for j in range(columns // 3):
                    if j != markerwithgap[x]:
                        point1 = get_point(Data, i, markerwithgap[x])
                        point2 = get_point(Data, i, j)
                        tmp = 0
                        if euclid_dist(point2, [0, 0, 0]) != 0:
                            weight_matrix[i][j] = euclid_dist(point2, point1)
                            weight_matrix_coe[i][j] = 1

        sum_matrix = np.sum(weight_matrix, 0)
        sum_matrix_coe = np.sum(weight_matrix_coe, 0)
        weight_vector_ith = sum_matrix / sum_matrix_coe
        weight_vector_ith[markerwithgap[x]] = 0
        weight_vector[x] = weight_vector_ith
    if gap_strategies == False:
        weight_vector = np.min(weight_vector, 0)
    weight_vector = np.exp(np.divide(-np.square(weight_vector), (2 * np.square(weightScale))))
    weight_vector[markerwithgap] = MMweight
    M_zero = np.copy(Data)
    # N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
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
    m7 = np.ones((Data.shape[0], 1)) * mean_N_nogap
    # if testMean:
    # 	print("OKKkkkkkkkKKkkkkkkkKKkkkkkkkKKkkkkkkkKKkkkkkkkKKkkkkkkkKKkkkkkkk")
    # 	mean_N_nogap = np.mean(N_nogap[:1], 0)
    # 	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))
    # 	m7 = np.ones((Data.shape[0], 1)) * mean_N_nogap

    return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat]


def compute_weight_vect_norm(markerwithgap, Data):
    weightScale = 200
    MMweight = 0.02
    AA = np.copy(Data)
    [frames, columns] = AA.shape
    marker = markerwithgap[0]
    frameindex = np.where(AA[:, marker * 3 + 1] == 0)[0]
    framewithgap = np.unique(frameindex)

    weight_vector = np.zeros((len(markerwithgap), columns // 3))
    for x in range(len(markerwithgap)):
        weight_matrix = np.zeros((frames, columns // 3))
        weight_matrix_coe = np.zeros((frames, columns // 3))
        for i in range(min(frames,200)):
            valid = True
            if euclid_dist([0, 0, 0], get_point(Data, i, markerwithgap[x])) == 0:
                valid = False
            if valid:
                for j in range(columns // 3):
                    if j != markerwithgap[x]:
                        point1 = get_point(Data, i, markerwithgap[x])
                        point2 = get_point(Data, i, j)
                        if euclid_dist(point2, [0, 0, 0]) != 0:
                            weight_matrix[i][j] = euclid_dist(point2, point1)
                            weight_matrix_coe[i][j] = 1

        sum_matrix = np.sum(weight_matrix, 0)
        sum_matrix_coe = np.sum(weight_matrix_coe, 0)
        weight_vector_ith = sum_matrix / sum_matrix_coe
        weight_vector_ith[markerwithgap[x]] = 0
        weight_vector[x] = weight_vector_ith

    weight_vector = np.min(weight_vector, 0)
    return weight_vector


def compute_weight_vect_norm_v2(markerwithgap, Data):
    weightScale = 200
    MMweight = 0.02
    AA = np.copy(Data)
    [frames, columns] = AA.shape
    marker = markerwithgap[0]
    frameindex = np.where(AA[:, marker * 3 + 1] == 0)[0]
    framewithgap = np.unique(frameindex)
    weight_vector = np.zeros((len(markerwithgap), columns // 3))
    for x in range(len(markerwithgap)):
        weight_matrix = np.zeros((frames, columns // 3))
        weight_matrix_coe = np.zeros((frames, columns // 3))
        for i in range(frames):
            valid = True
            if euclid_dist([0, 0, 0], get_point(Data, i, markerwithgap[x])) == 0:
                valid = False
            if valid:
                for j in range(columns // 3):
                    if j != markerwithgap[x]:
                        point1 = get_point(Data, i, markerwithgap[x])
                        point2 = get_point(Data, i, j)
                        weight_matrix[i][j] = euclid_dist(point2, point1)
                        weight_matrix_coe[i][j] = 1

        sum_matrix = np.sum(weight_matrix, 0)
        sum_matrix_coe = np.sum(weight_matrix_coe, 0)
        weight_vector_ith = sum_matrix / sum_matrix_coe
        weight_vector_ith[markerwithgap[x]] = 0
        weight_vector[x] = weight_vector_ith

    weight_vector = np.min(weight_vector, 0)
    return weight_vector


def check_vector_overlapping(vector_check):
    if len(np.where(vector_check == 0)[0]) > 0:
        return True
    return False


def get_list_frameidx_patch(K_patch, fix_leng, frame_start):
    list_frameidx_patch = []
    for patch_number in range(K_patch):
        patch_start_frame = patch_number * fix_leng + 0
        patch_end_frame = patch_start_frame + fix_leng
        list_frameidx_patch.append([patch_start_frame, patch_end_frame])
    return list_frameidx_patch


def calculate_mae_matrix(X):
    error_sum = np.sum(np.abs(X))
    mse = np.sum(np.square(X))
    # print("debug")
    print("mse error: ", mse)
    print("mae error: ", error_sum)
    return error_sum / len(X)


def matrixConnect():
    joint_connection = [[0, 2], [1, 3], [2, 4], [3, 4], [9, 10], [10, 11], [11, 12], [12, 14], [14, 13], [14, 15],
                        [16, 17], [17, 18], [18, 19], [19, 21], [21, 20], [21, 22],
                        [23, 27], [27, 28], [28, 29], [29, 30], [30, 31], [30, 32], [30, 33],
                        [24, 34], [34, 35], [35, 36], [36, 37], [37, 38], [37, 39], [37, 40],
                        [4, 5], [9, 4], [16, 4], [25, 23], [24, 26], [5, 25], [
                            5, 26], [7, 23], [7, 24], [6, 16], [6, 9], [6, 7],
                        [4, 8], [5, 8]]
    # matrix  = np.zeros([40, 40])
    list_connect = []
    for marker in range(41):
        marker_connect = []
        for x in joint_connection:
            if (x[0] == marker):
                marker_connect.append(x[1])
            if (x[1] == marker):
                marker_connect.append(x[0])
        list_connect.append(marker_connect)
    return list_connect


def calDistMatrix(full_data):
    markers = full_data.shape[1] // 3
    frames = full_data.shape[0]
    dist_matrix = np.zeros([markers, markers])
    for markerX in range(markers):
        for markerY in range(markers):
            accumulateSum = 0
            if markerX != markerY:
                for frameIdx in range(frames):
                    point1 = get_point(full_data, frameIdx, markerX)
                    point2 = get_point(full_data, frameIdx, markerY)
                    accumulateSum += euclid_dist(point1, point2)
            dist_matrix[markerX, markerY] = accumulateSum / frames
    return dist_matrix


def calWeight(marker, full_data, dist_matrix):
    list_connect = matrixConnect()
    queue = [marker]
    currentQ = 0
    endQueue = 0
    checkList = [0] * len(list_connect)
    checkList[marker] = 1
    distArr = np.zeros(len(list_connect))
    while currentQ <= endQueue:
        current_pos = queue[currentQ]
        list_edge = list_connect[current_pos]
        for vertex in list_edge:
            if checkList[vertex] == 0:
                endQueue += 1
                queue.append(vertex)
                distArr[vertex] = distArr[current_pos] + dist_matrix[current_pos, vertex]
                # distArr[endQueue] = distArr[currentQ] + 0
                checkList[vertex] = 1

        currentQ += 1
    # distArr = distArr / full_data.shape[0]
    return distArr


def getFullMat(matrix):
    frameindex = np.where(matrix == 0)[0]
    framewithgap = np.unique(frameindex)

    fullMat = np.delete(np.copy(matrix), framewithgap, 0)
    return fullMat


def get_closest(marker, weight_vector):
    weight_vector[marker] = 9999
    current = marker
    for idx, _ in enumerate(weight_vector):
        if weight_vector[idx] < weight_vector[current]:
            current = idx
    weight_vector[marker] = 0
    return current

def get_overlap(marker, CJ, missing_matrix):
    missing_frame_M = np.where(missing_matrix[:, marker*3] == 0)[0]
    missing_frame_CJ = np.where(missing_matrix[:, CJ*3] == 0)[0]
    l = 0
    r = 0
    list_nonOverlap = []
    list_overlap = []
    while(l < len(missing_frame_M) and r < len(missing_frame_CJ)):
        if (missing_frame_M[l] == missing_frame_CJ[r]):
            list_overlap.append(missing_frame_M[l])
            l+=1
            r+=1
        elif (missing_frame_M[l] < missing_frame_CJ[r]):
            list_nonOverlap.append(missing_frame_M[l])
            l+=1
        else:
            r+=1
    while(l<len(missing_frame_M)):
        list_nonOverlap.append(missing_frame_M[l])
        l+=1
    return list_nonOverlap, list_overlap

def check_diag(matrix):
    xx, yy = matrix.shape
    counter = 0
    for x in range(min(xx, yy)):
        if matrix[x][x] == 1:
            counter += 1
    if counter == min(xx, yy): 
        return True
    return False



joint_connect = [[0, 2], [1, 3], [2, 4], [3, 4], [9, 10], [10, 11], [11, 12], [12, 14], [14, 13], [14, 15], 
    [16, 17], [17, 18], [18, 19], [19, 21], [21, 20], [21, 22],
    [23, 27], [27, 28], [28, 29], [29, 30], [30, 31], [30, 32], [30, 33], 
    [24, 34], [34, 35], [35, 36], [36, 37], [37, 38], [37, 39], [37, 40],
    [4, 5],  [9, 4], [16, 4], [25, 23], [24, 26], [5, 25], [5, 26],[7, 23], [7, 24], [6, 16], [6, 9], [6, 7],
    [4, 8], [5, 8]]


def distance(joint1, joint2, location):
    return euclid_dist(location[joint1*3:joint1*3+3], location[joint2*3:joint2*3+3])

def dist(matrix, marker):
    location = np.mean(matrix,0)
    d = [99999]*41
    visited = [False] * 41
    d[marker] = 0
    while 1 > 0:
        nextJoint = marker
        minDis = 9999999
        for joint in range(len(d)):
            if d[joint] < minDis:
                minDis = d[joint]
                nextJoint = joint
        if visited[nextJoint]:
            break
        visited[nextJoint] = True
        for joint in range(len(d)):
            if visited[joint] == False:
                d[joint] = d[nextJoint] + distance(nextJoint, joint, location)
    return np.asarray(d)