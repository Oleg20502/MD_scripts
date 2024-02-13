#!/bin/bash

for n in {0..63}
do
    cd s${n}/
    python ../calc_acf_int.py 
    echo Finished $n
    cd ../
done
