import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from data import data
def draw(data, figure, outer_grid, name, last = False, first = False):
	ax = figure.add_subplot(outer_grid)
	bp = ax.boxplot(data, 
	                notch ='True', vert = True) 
	color = ['green', 'blue', 'red']
	
	for count, whisker in enumerate(bp['whiskers']): 
	    ct = int(count / 2)
	    whisker.set(color =color[ct], 
	                linewidth = 1.5, 
	                linestyle ="--") 
	for ct, box in enumerate(bp['boxes']): 
		box.set(color =color[ct],linewidth = 1)  

	for count, cap in enumerate(bp['caps']):
		ct = int(count / 2)
		cap.set(color =color[ct],linewidth = 1)  

	for median in bp['medians']: 
	    median.set(color ='red', 
	               linewidth = 1) 
	  
	for flier in bp['fliers']: 
	    flier.set(marker ='+', 
	              markeredgecolor ='red', 
	              alpha = 1)
	      
	ax.label_outer()
	ax.axes.xaxis.set_visible(False)
	if first == False: ax.axes.yaxis.set_visible(False)
	ax.set_ylim(0,70)
	ax.set_title(name, y=-0.1)
	if first: ax.set_ylabel('Reconstruction Error')
	if last: 
		ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]], ['3 markers', '6 markers', '9 markers'], loc='upper right')

fig = plt.figure(figsize=(12, 5), constrained_layout=False)
outer_grid = gridspec.GridSpec(ncols=4, nrows=1, figure=fig, wspace=0)

data_1 = data[0]
data_2 = data[1]
data_3 = data[2]
data_4 = data[3]

draw(data_1, fig, outer_grid[0, 0], "PCA", first = True)   
draw(data_2, fig, outer_grid[0, 1], "Lis")
draw(data_3, fig, outer_grid[0, 2], "WPCA")
draw(data_4, fig, outer_grid[0, 3], "LWPCA", last = True)
plt.savefig('Boxplot.eps', format='eps')
# plt.show()
