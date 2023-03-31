# export DATA_TEXT_LEN=128
cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
nproc=$(expr $cpus - 1)
echo "Using ${nproc} procs..."
datadir=$1
find -L ${datadir} -name "*.txt" | xargs --max-args=1 --max-procs=${nproc} bash gen_dataset_from_text.sh
