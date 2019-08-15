# file containing output of GetDataFig12.sh
# Make sure that the output contains results for all tasks- technique combination
file=$1
# The Background task ID {0,1,2,3} 
bg_task=$2
touch sd_$file
less $file | grep BACKGROUND -B 1000 > event_$file
less $file | grep BACKGROUND -A 1000 > BG_$file
less event_$file | cut -d " " -f 3 | grep -v avg | tr -s "\n" | grep -v EVENT > t2
less BG_$file | cut -d " " -f 1 | grep -v ipv | grep -v memc | grep -v des_ | grep -v BACKGROUND | grep -v min > t1
less event_$file | cut -d " " -f 4 | grep -v p95 |  tr -s "\n" | grep -v EVENT > t3
less event_$file | cut -d " " -f 5 |  tr -s "\n" | grep -v EVENT > t4
paste -d , t1 t2 t3 t4 > sd_$file
python ../process_data.py sd_$file $bg_task

