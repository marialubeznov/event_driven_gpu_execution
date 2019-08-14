import pandas as pd 
import numpy as np
from numpy import genfromtxt
import sys
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from scipy import stats

file = sys.argv[1]
bg_task = int(sys.argv[2])

data = pd.read_csv(file,header=None)

#Isolation runtime of BG tasks and the event kernels
bg_task_lat = [917947,917947,698933,917947]
event_lat = [2300 ,2300 ,22936,24171]
seq=["\"D\"", "\"R\"" ,"\"P\""]
event_seq = ["IPv4" ,"IPv6", "memc" ,"ipsec"]
conv=917947
mm=917947#917025
bfs= 917947
bp=698933
ipv4=2300
ipv6=2300
memc=22936
des=24171
f= open("relative_"+file,"w")
#f.write("This is line \r\n" )
if True:
	if len(data[0])!=12:
		print("Incorrect number of lines in %s"%(file))
	else:
		
		for i in range(0,4):
			print("#	BG	%s	ANTT"%(event_seq[i]))
			for j in range(0,3):
				index=i*3+j
				#print(index)
				bg_sd = data[0][index]/bg_task_lat[bg_task]
				event_sd = data[1][index]/event_lat[i]
				antt= ((data[3][index]*event_sd)+bg_sd)/(data[3][index] + 1)
				print("%s %f %f %f %d"%(seq[j],bg_sd,event_sd,antt,index+i))
				f.write("%f,%f,%f\n"%(bg_sd,event_sd,antt))
			print("\n")
