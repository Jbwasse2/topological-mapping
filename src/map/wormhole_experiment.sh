#!/usr/bin/env bash
function run_experiment() {
    a=("$@")
    ((last_idx=${#a[@]} - 1))
    namel=${a[last_idx]}
    unset a[last_idx]
    gpu=${a[last_idx-1]}
    unset a[last_idx-1]

    echo "******************"
    echo "namel: $namel"
    echo "gpu: $gpu"
    echo "envs"
    for i in "${a[@]}" ; do
        echo "$i"
    done
    echo "******************"
    cp make_map_worm_experiment.py make_map_worm_experiment_$namel.py
    sed -i "s|set_GPU =.*|set_GPU = '$gpu'|g" make_map_worm_experiment_$namel.py
    for env in "${a[@]}" ; do
        echo "$env"
        mkdir -p ./results/test50/$env
        sed -i "s|env =.*|env = '$env'|g" make_map_worm_experiment_$namel.py
        while read p; do
            echo "$p"
            sed -i "s|map_type_test =.*|map_type_test = '$p'|g" make_map_worm_experiment_$namel.py
            python make_map_worm_experiment_$namel.py > ./results/test50/$env/$p 2>&1
        done < ./map_types.txt
    done
}

#./worm_model/test_env.txt
x1=("Browntown" "Harkeyville"  "Gilbert" "Bowlus" "Maugansville" "Holcut" "McCloud" "Shelbiana" "Spencerville" "Hambleton" "Milford" "Maida" "Deatsville" "Sweatman" "Inkom" "Chesterbrook" "Waimea" "Kremlin" "Springerville" "Kangley" "Azusa" "Pasatiempo" )
#x1=("Harkeyville" "Browntown" "Gilbert" "Bowlus" "Maugansville" "Holcut")
GPU=0
name=1
run_experiment "${x1[@]}" "$GPU" "$name" &

x2=("Ackermanville" "Nuevo" "Checotah" "Aloha" "Bountiful" "Kankakee" "Stokes" "Poyen" "Scandinavia" "Lynxville" "Ophir" "Seeley" "Weleetka" "Goodwine" "Mammoth" "Pocopson" "Mifflintown" "Chrisney" "Sanctuary" "Mazomanie" "Nemacolin" "Cullison" "Chireno")
#x2=("Ackermanville" "Nuevo" "Checotah" "Aloha" "Bountiful" "Kankakee" "Stokes" "Poyen")
#x2=("Scandinavia" "Lynxville" "Ophir" "Seeley" "Weleetka" "Goodwine" "Mammoth" "Pocopson" "Mifflintown" "Chrisney" "Sanctuary" "Mazomanie" "Nemacolin" "Cullison" "Chireno")
GPU=1
name=2
run_experiment "${x2[@]}" "$GPU" "$name" &
