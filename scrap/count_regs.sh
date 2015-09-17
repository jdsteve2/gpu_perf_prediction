for file in ptx_files/*
do
    echo $file
    /exthome/andreas/cuda-7.5/bin/ptxas -v --gpu-name sm_35 $file
done
