#!/bin/bash

function print_array {
    declare -a arr=($@)
    declare -a sizes=($2)
    for i in ${arr[@]}; do
        for j in ${sizes[@]}; do
            echo $i $j;
        done
    done
}

array=('mobilenet_v3' 'vgg16' 'resnet34' )
sizes=('100' '10_000' '100_000' )

for i in ${array[@]}; do
    for j in ${sizes[@]}; do
        python3 model_evaluate.py $i $j 
    done
done