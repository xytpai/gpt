# export DATA_TEXT_LEN=128
cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
echo "Using ${cpus} CPU cores..."
datadir=$1
find -L ${datadir} -name "*.txt" | xargs --max-args=1 --max-procs=${cpus} bash gen_dataset_from_text.sh
