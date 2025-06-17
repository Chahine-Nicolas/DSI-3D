#!/bin/bash

title="Hey that's a super idea!!!"
sbatch_file="sbatch/eval.sh"

#sbatch ${sbatch_file}
#sleep 1
# Store the most recently created directory name in a variable
log_dir=$(ls -t -d -- ./logs/* | head -n 1)

echo "The most recently created directory is: $log_dir"
echo "$tile" > ${log_dir}/notes.txt
cp -rf ${sbatch_file} ${log_dir}
