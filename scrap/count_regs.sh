for file in ptx_files/*
do
    echo $file
    perl -pi -e 's/.version 4.3/.version 4.1/g' $file
    tac $file | tail -n +10 | tac > _$file
    ptxas -v --gpu-name sm_20 _$file
done

#/exthome/andreas/cuda-7.5/bin/ptxas -v --gpu-name sm_35 $file
