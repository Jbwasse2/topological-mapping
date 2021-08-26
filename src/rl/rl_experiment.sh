#!/usr/bin/env bash
## declare an array variable

function run_experiment() {
    a=("$@")
    ((last_idx=${#a[@]} - 1))
    copy=${a[last_idx]}
    unset a[last_idx]
    gpu=${a[last_idx-1]}
    unset a[last_idx-1]
    echo "******************"
    echo "copy: $copy"
    echo "gpu: $gpu"
    echo "******************"

    cp sim_model.py sim_model$copy.py
    sed -i "s|set_GPU =.*|set_GPU = '$gpu'|g" sim_model$copy.py
    sed -i "s|MAX_NUMBER_OF_STEPS =.*|MAX_NUMBER_OF_STEPS = 30|g" sim_model$copy.py
    for env in "${a[@]}" ; do
        sed -i "s|env =.*|env = '$env'|g" sim_model$copy.py
        mkdir -p ./results/test50/$env
        while read p; do
            echo "$p"
            sed -i "s|map_type_test =.*|map_type_test = '$p'|g" sim_model$copy.py
            python sim_model$copy.py > ./results/test50/$env/$p 2>&1
        done < ./map_types.txt
    done
    rm sim_model$copy.py
}

x1=("Gilbert" "Harkeyville" "Holcut" "Stokes")
GPU=0
name=1
run_experiment "${x1[@]}" "$GPU" "$name" &

x2=("Ackermanville" "Aloha" "Bountiful" "Kankakee" "Maugansville" )
GPU=0
name=2
run_experiment "${x2[@]}" "$GPU" "$name" &

x3=("Bowlus" "Browntown" "Checotah" "Nuevo")
GPU=0
name=3
run_experiment "${x3[@]}" "$GPU" "$name" &
