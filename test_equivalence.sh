for dataset in "1f2s"  "1f3s" "1f10s"
do
    python baseline/graphrl/supplychain/main.py \
        --dataset ${dataset} \
        --test \
        --log_level 20 \
        --max_episodes 30 \
        --seed 1010 \
        > test_logs/equiv_scim_${dataset}_old.log 2>&1

    python main.py \
        --scenario scim \
        --dataset ${dataset} \
        --algo grl \
        --algo_mode test \
        --log_level 30 \
        --epi_num 30 \
        --seed 1010 \
        --demand_pattern cosine \
        --use_pretrained_ckpt \
        > test_logs/equiv_scim_${dataset}_new.log 2>&1
done

for dataset in  "nyc" "sz"
do
    python baseline/graphrl/mobility/main.py \
        --dataset ${dataset} \
        --test \
        --log_level 50 \
        --max_episodes 30 \
        --seed 10 \
        > test_logs/equiv_ecr_${dataset}_old.log 2>&1

    python main.py \
        --scenario ecr \
        --expt_name realprice \
        --dataset ${dataset} \
        --algo grl \
        --algo_mode test \
        --log_level 50 \
        --epi_num 30 \
        --seed 10 \
        --fuel \
        --use_pretrained_ckpt \
        > test_logs/equiv_ecr_${dataset}_new.log 2>&1
done