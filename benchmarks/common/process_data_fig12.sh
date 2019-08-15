# List of files containing output of GetDataFig11.sh
# Make sure that the output contains results for all tasks- technique combination
file_list=("0_no_overlap" "1_no_overlap" "2_no_overlap" "3_no_overlap")
#order of BG task should match the order of the file list
bg_task=("CONV" "MM" "BP" "BFS")


echo "	draining			preemption		"
echo "#	AVG	MIN	MAX	AVG	MIN	MAX"
rm -rf t3
rm -rf t4
for i in {0..3}
do
	file=${file_list[$i]}

	bg_task=${bg_task[$i]}
	touch drain_$file
	touch preempt_$file


	less $file |  grep  ipv4_fwd -A 3 | grep drain -A 3 | grep -v drain  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "ipv4_fwd,draining,$av,$min,$max" > drain_$file

	less $file |  grep  ipv4_fwd -A 3 | grep preempt -A 3 | grep -v preempt  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "ipv4_fwd,preemption,$av,$min,$max" > preempt_$file

	less $file |  grep  ipv6_fwd -A 3 | grep drain -A 3 | grep -v drain  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "ipv6_fwd,draining,$av,$min,$max" >> drain_$file

	less $file |  grep  ipv6_fwd -A 3 | grep preempt -A 3 | grep -v preempt  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "ipv6_fwd,preemption,$av,$min,$max" >> preempt_$file

	less $file |  grep  memc -A 3 | grep drain -A 3 | grep -v drain  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "memc,draining,$av,$min,$max" >> drain_$file

	less $file |  grep  memc -A 3 | grep preempt -A 3 | grep -v preempt  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "memc,preemption,$av,$min,$max" >> preempt_$file

	less $file |  grep  des_enc -A 3 | grep drain -A 3 | grep -v drain  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "des_encryption,draining,$av,$min,$max" >> drain_$file

	less $file |  grep  des_enc -A 3 | grep preempt -A 3 | grep -v preempt  > t2
	av=`awk -F " " 'BEGIN{sum=0}{sum=sum+$1}END{print sum/3}' t2`
	min=`awk -F " " 'BEGIN{sum=0}{sum=sum+$2}END{print sum/3}' t2`
	max=`awk -F " " 'BEGIN{sum=0}{sum=sum+$3}END{print sum/3}' t2`
	echo "des_encryption,preemption,$av,$min,$max" >> preempt_$file

	av=`awk -F "," 'BEGIN{sum=0}{sum=sum+$3}END{print sum/4}' drain_$file`
	min=`awk -F "," 'BEGIN{sum=0}{sum=sum+$4}END{print sum/4}' drain_$file`
	max=`awk -F "," 'BEGIN{sum=0}{sum=sum+$5}END{print sum/4}' drain_$file`
	echo "$av,$min,$max" >> t3

	av1=`awk -F "," 'BEGIN{sum=0}{sum=sum+$3}END{print sum/4}' preempt_$file`
	min1=`awk -F "," 'BEGIN{sum=0}{sum=sum+$4}END{print sum/4}' preempt_$file`
	max1=`awk -F "," 'BEGIN{sum=0}{sum=sum+$5}END{print sum/4}' preempt_$file`
	

	echo "\"$bg_task\" $av $min $max $av1 $min1 $max1"

	echo "$av1,$min1,$max1" >> t4
done

# draining average
av=`awk -F "," 'BEGIN{sum=0}{sum=sum+$1}END{print sum/4}' t3`
min=`awk -F "," 'BEGIN{sum=0}{sum=sum+$2}END{print sum/4}' t3`
max=`awk -F "," 'BEGIN{sum=0}{sum=sum+$3}END{print sum/4}' t3`

# preemption average
av1=`awk -F "," 'BEGIN{sum=0}{sum=sum+$1}END{print sum/4}' t4`
min1=`awk -F "," 'BEGIN{sum=0}{sum=sum+$2}END{print sum/4}' t4`
max1=`awk -F "," 'BEGIN{sum=0}{sum=sum+$3}END{print sum/4}' t4`

echo "\"AVERAGE\" $av $min $max $av1 $min1 $max1$"

rm -rf t1 t2 t3 t4
rm -rf preempt_*_no_overlap
rm -rf drain_*_no_overlap
