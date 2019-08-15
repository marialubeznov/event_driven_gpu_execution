# file containing output of GetDataFig12.sh for low_util
# Make sure that the output contains results for all tasks- technique combination
file_list=("0_low_util" "1_low_util" "2_low_util" "3_low_util")
bg_task_list=("CONV" "MM" "BP" "BFS")

for i in {0..3}
do
	file=${file_list[$i]}
	# The Background task ID {0,1,2,3} 
	bg_task=$i
	name=${bg_task_list[$i]}
	touch sd_$file
	less $file | grep BACKGROUND -B 1000 > event_$file
	less $file | grep BACKGROUND -A 1000 > BG_$file
	less event_$file | cut -d " " -f 3 | grep -v avg | tr -s "\n" | grep -v EVENT > t2
	less BG_$file | cut -d " " -f 1 | grep -v ipv | grep -v memc | grep -v des_ | grep -v BACKGROUND | grep -v min > t1
	less event_$file | cut -d " " -f 4 | grep -v p95 |  tr -s "\n" | grep -v EVENT > t3
	less event_$file | cut -d " " -f 5 |  tr -s "\n" | grep -v EVENT > t4
	paste -d , t1 t2 t3 t4 > sd_$file
	python ../process_data.py sd_$file $bg_task > ${name}_low_util.dat
	rm -rf event_$file
	rm -rf BG_$file
done
python ../average_data.py relative_sd_0_low_util relative_sd_1_low_util relative_sd_2_low_util relative_sd_3_low_util > avg_low_util.dat

# file containing output of GetDataFig12.sh
# Make sure that the output contains results for all tasks- technique combination
file_list=("0_high_util" "1_high_util" "2_high_util" "3_high_util")
bg_task_list=("CONV" "MM" "BP" "BFS")

for i in {0..3}
do
	file=${file_list[$i]}
	# The Background task ID {0,1,2,3} 
	bg_task=$i
	touch sd_$file
	less $file | grep BACKGROUND -B 1000 > event_$file
	less $file | grep BACKGROUND -A 1000 > BG_$file
	less event_$file | cut -d " " -f 3 | grep -v avg | tr -s "\n" | grep -v EVENT > t2
	less BG_$file | cut -d " " -f 1 | grep -v ipv | grep -v memc | grep -v des_ | grep -v BACKGROUND | grep -v min > t1
	less event_$file | cut -d " " -f 4 | grep -v p95 |  tr -s "\n" | grep -v EVENT > t3
	less event_$file | cut -d " " -f 5 |  tr -s "\n" | grep -v EVENT > t4
	paste -d , t1 t2 t3 t4 > sd_$file
	python ../process_data.py sd_$file $bg_task > ${name}_high_util.dat
	rm -rf event_$file
	rm -rf BG_$file
done
python ../average_data.py relative_sd_0_high_util relative_sd_1_high_util relative_sd_2_high_util relative_sd_3_high_util > avg_high_util.dat

rm -rf relative_sd_*_util
rm -rf sd_*_util
rm -rf t1 t2 t3 t4 