#!/usr/bin/python    	
filename = "./results_temp.log"
with open(filename) as f:
	data = [int(line) for line in f]
max_data = max(data)
#print("The minimum value is ", min(data))
#print("The maximum value is ", max_data)
#print("The average value is ", sum(data)/len(data))
sum95=0
cnt95=0
for num in data:
	if num>(0.95*max_data):
		sum95 = sum95+num
		cnt95 = cnt95+1
#print("The p95 value is ", sum95/cnt95)
print min(data), max_data, sum(data)/len(data), sum95/cnt95, len(data)