import pandas as pd 
import numpy as np
from numpy import genfromtxt
import sys
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from scipy import stats

# Averages the slow down for all BG tasks.
# Each file corresponds to a BG task
# Run the scripts seperately for low and high util
# Output files dumped by the grep_data.sh script
file0 = sys.argv[1]
file1 = sys.argv[2]

file2 = sys.argv[3]
file3 = sys.argv[4]

data0 = pd.read_csv(file0,header=None)
data1 = pd.read_csv(file1,header=None)
data2 = pd.read_csv(file2,header=None)
data3 = pd.read_csv(file3,header=None)
bg_average=np.zeros((12, 3))
for i in range(0,12):
	bg_average[i][0]=(data0[0][i]+data1[0][i]+data2[0][i]+data3[0][i])/4
	bg_average[i][1]=(data0[1][i]+data1[1][i]+data2[1][i]+data3[1][i])/4
	bg_average[i][2]=(data0[2][i]+data1[2][i]+data2[2][i]+data3[2][i])/4
fin = np.zeros((3,3))
for i in range(0,3):
	fin[i][0]=(bg_average[i][0]+bg_average[i+3][0]+bg_average[i+6][0]+bg_average[i+9][0])/4
	fin[i][1]=(bg_average[i][1]+bg_average[i+3][1]+bg_average[i+6][1]+bg_average[i+9][1])/4
	fin[i][2]=(bg_average[i][2]+bg_average[i+3][2]+bg_average[i+6][2]+bg_average[i+9][2])/4
print("#	BG	Event	ANTT")
seq=["\"D\"","\"R\"","\"P\""]
for i in range(0,3):
	print("%s %f %f %f "%(seq[i],fin[i][0],fin[i][1],fin[i][2]))