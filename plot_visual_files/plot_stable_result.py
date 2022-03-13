import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:\\Users\\nvmnghia\\Desktop\\weightPCA")
import data

list_matrix = data.test


ox = [x for x in range(50)]

fig = plt.figure()
# print("Ours: 		  ", method_03_March)
# # print("method_10_Feb:", method_10_Feb)
plt.plot( ox, list_matrix[0],  marker='.', color='black', linewidth=2, label="1.PCA_g")
plt.plot( ox, list_matrix[1],  marker='.', color='red', linewidth=2, label="2.WPCAG")
plt.plot( ox, list_matrix[2],  marker='.', color='green', linewidth=1, label="3.WPCA")
# plt.plot( ox, list_matrix[2],  marker='.', color='blue', linewidth=1, label="4.WPCA")
# plt.plot( ox, list_matrix[3][i],  marker='_', color='black', linestyle="--", label="4.PMA_no_data")
# plt.plot( ox, list_matrix[4][i],  marker='_', color='red', linestyle="--", label="5.PCA_no_data")
plt.ylabel(' Reconstruction Error Value', fontsize=20)
plt.xlabel(' Test index ', fontsize=20)
plt.title(' LWA in HDM dataset', fontsize=30)
# plt.show()
plt.legend(fontsize = 20)
fig.set_size_inches(10, 6)
# plt.ylim(0, 20)
# plt.savefig("C://Users//nvmnghia/Desktop/weightPCA" + './list_result/'+'cmu50.eps', format='eps')
plt.savefig('LWA_HDM_stb.eps', format='eps')
# plt.savefig('result'+'.png')
# plt.clf()