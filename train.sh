mkdir -p train_logs
for dataset in "didi9" "didi20"
do
    # # real price, with fuel
    nohup python \
        -u main.py \
        --scenario ecr \
        --expt_name realprice \
        --dataset ${dataset} \
        --algo grl \
        --algo_mode train \
        --epi_num 20000 \
        --log_level 50 \
        --seed 10 \
        --fuel \
        --use_wandb \
        > train_logs/${dataset}_realprice_withfuel.log 2>&1 &

    # real price, no fuel
    nohup python \
        -u main.py \
        --scenario ecr \
        --expt_name realprice \
        --dataset ${dataset} \
        --algo grl \
        --algo_mode train \
        --epi_num 20000 \
        --log_level 50 \
        --seed 10 \
        --use_wandb \
        > train_logs/${dataset}_realprice_nofuel.log 2>&1 &

    # # unit price, nofuel
    nohup python \
        -u main.py \
        --scenario ecr \
        --expt_name unitprice \
        --dataset ${dataset} \
        --algo grl \
        --algo_mode train \
        --epi_num 20000 \
        --log_level 50 \
        --seed 10 \
        --use_wandb \
        > train_logs/${dataset}_unitprice_nofuel.log 2>&1 &
done