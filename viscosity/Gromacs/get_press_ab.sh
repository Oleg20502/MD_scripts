#!/bin/bash

for n in {0..63}
do
    cd s${n}/
    gmx -quiet energy -f ener.edr -o press_ab.xvg <<< "27 28 31 0"
    echo Finished $n
    cd ../
done
