Code of the paper "Dynamic Mean-Field Control for Network MDPs with Exogenous Demand".

# Installation
[Gurobi](https://www.gurobi.com/) is used as the optimization solver. Please install Gurobi and set the license before running the code.

For consistant performance on some edge cases, some [IBM CPLEX](https://www.ibm.com/cn-zh/products/ilog-cplex-optimization-studio) module components from the original codebases have been retained.

> Python version: 3.10
> 
> Gurobi version: 12
> 
> CPLEX version: 22.1.1

Install the required packages:
```bash
pip install requirements.txt -r
```

Update config file.
```toml
wandb_key="your_wandb_key"  # set wandb if needed
ibm_cplex="your_cplex_path" 
```


# Usage
1. run evaluations:
```bash
bash test.sh
```

2. visualize results within jupyter notebooks:
```bash
scim_visualization.ipynb
ecr_visualization.ipynb
``` 

# Baseline
## Graph-RL 
> baseline repo: https://github.com/DanieleGammelli/graph-rl-for-network-optimization
>
> paper: https://proceedings.mlr.press/v202/gammelli23a.html

We refactored the code environment with minor modifications to support more detailed and flexible customization.

The modified environment remains functionally equivalent to the original baseline implementation.

Equivalence check:
```bash
bash test_equivalence.sh
```

> However, differences in solver strategies can yield distinct feasible solutions with the same objective value, causing discrepancies—especially in vehicle routing for mobility-on-demand systems. Additionally, numerical precision issues can further contribute to result variability. Although these differences may introduce fluctuations in the final results, the overall environment remains equivalent.

# citation
```text
@inproceedings{
    ye2025dynamic,
    title={Dynamic Mean-Field Control for Network {MDP}s with Exogenous Demand},
    author={Botao Ye and Hengquan Guo and Weichang Wang and Xin Liu},
    booktitle={Second Coordination and Cooperation in Multi-Agent Reinforcement Learning Workshop},
    year={2025},
    url={https://openreview.net/forum?id=Z9gxUpJgaN}
}
```