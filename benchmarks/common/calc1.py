#!/usr/bin/python
import sys

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

filename = sys.argv[1]
with open(filename) as f:
	data = [num(line) for line in f]
max_data = max(data)
data.sort(reverse = True)
#print("The minimum value is ", min(data))
#print("The maximum value is ", max_data)
#print("The average value is ", sum(data)/len(data))
sum95=0
cnt95=0
#print data
for num in data:
	if cnt95>(0.05*len(data)):
		break;
	sum95 = sum95+num
	cnt95 = cnt95+1
#print sum95, cnt95
print sum(data)/len(data), min(data), max_data, len(data)

