generate_commands() {
    for dataset in "didi9" "didi20"; do
        echo "python main.py --scenario ecr --expt_name unitprice --dataset ${dataset} --algo dmfc --algo_mode test --log_level 50 --epi_num 30 --seed 10"
        echo "python main.py --scenario ecr --expt_name unitprice --dataset ${dataset} --algo grl --algo_mode test --log_level 50 --epi_num 30 --seed 10"
        echo "python main.py --scenario ecr --expt_name realprice --dataset ${dataset} --algo dmfc --algo_mode test --log_level 50 --epi_num 30 --seed 10 --fuel"
        echo "python main.py --scenario ecr --expt_name realprice --dataset ${dataset} --algo grl --algo_mode test --log_level 50 --epi_num 30 --seed 10 --fuel"
        echo "python main.py --scenario ecr --expt_name realprice --dataset ${dataset} --algo grl --algo_mode test --log_level 50 --epi_num 30 --seed 10"
    done # dmfc: it doesn't matter whether have or not fuel
    for dataset in "nyc" "sz"; do
        echo "python main.py --scenario ecr --expt_name realprice --dataset ${dataset} --algo dmfc --algo_mode test --log_level 50 --epi_num 30 --seed 10 --fuel"
        echo "python main.py --scenario ecr --expt_name realprice --dataset ${dataset} --algo grl --algo_mode test --log_level 50 --epi_num 30 --seed 10 --fuel --use_pretrained_ckpt"
    done
    for dataset in "1f2s"  "1f3s" "1f10s"; do
        echo "python main.py --scenario scim --dataset ${dataset} --algo grl --algo_mode test --log_level 50 --epi_num 30 --seed 1010 --demand_pattern cosine --use_pretrained_ckpt"
        echo "python main.py --scenario scim --dataset ${dataset} --algo dmfc --algo_mode test --log_level 50 --epi_num 30 --seed 1010 --demand_pattern cosine"
    done
}
generate_commands | xargs -I {} -P 4 bash -c "{}"