<p float="left">
    <img src="data/img/icon_kit.png" width="10%" hspace="20"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.9.18-blue?logo=python)](https://www.python.org/downloads/release/python-3918/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource)](./LICENSE)
[![Flake8](https://img.shields.io/badge/Linting-Flake8-blueviolet?logo=python)](https://flake8.pycqa.org/)


<h1 align="center">Linear and Nonlinear Model Predictive Control for Distributed Energy Resources in Power Grids</h1>

**⚠️ Note**: *Last update on 27.06.2025*

<div align="left"> This repository is the official code of the paper <strong>"Linear and Nonlinear Model Predictive Control for Distributed Energy Resources in Power Grids"</strong>, featuring a low-voltage grid simulation environment with control of photovoltaic (PV) inverters, battery energy storage systems (BES), and heat pumps (HP). It includes several generation and load profiles, dynamic EPEX spot market prices, and tools for evaluating linear and nonlinear MPC control strategies. </div> 

## 1. Introduction eGridLVOpt

<details>
  <summary>Click to expand/collapse</summary>

### 1.1 Description

We present **[eGridLVOpt](https://github.com/KIT-IAI/eGridLVOpt)**, an open source MPC framework implemented in nonlinear program (NLP) and mixed-integer linear programming (MILP) formulations. 
The nonlinear MPC variant is based on **[Pyomo](https://github.com/Pyomo/pyomo)** and uses the **[Interior Point OPTimizer (IPOPT)](https://github.com/coin-or/Ipopt)** solver (version 3.13.4), while the linear MPC variant uses Google’s **[Operations Research Tools (OR-Tools)](https://github.com/google/or-tools)** with the `pywraplp` interface and the **[Coin-or Branch and Cut (CBC)](https://github.com/coin-or/Cbc)** solver.


#### Key Features


- Linear MPC: OR-Tools (`pywraplp`) with `CBC` solver
- Nonlinear MPC: `Pyomo` with `IPOPT`
- Flexible prediction horizons: 3h to 24h
- Configurable noise levels: 0–1%
- Objective weighting: trade-off between energy exchange and economic profit (`w1`, `w2`)
- Compatible with SimBench LV grids
- Historical spot market prices
- 15-minute simulation resolution
- Battery SoC constraints: 0–100%
- Input profile noise injection for robustness
- Slurm cluster support for large-scale runs

</details>

## 2. Installation and Environment Setup

<details>
  <summary>Click to expand/collapse</summary>

#### 2.1 Clone the Repository
```bash
git clone https://github.com/KIT-IAI/eGridLVOpt
cd eGridLVOpt
python3.9 -m venv eGridLVOpt_env
source eGridLVOpt_env/bin/activate
```
 
#### 2.2 Install Dependencies
```bash
pip install -r NLPMILP_requirements.txt
```

#### 2.3 Make Scripts Executable
```bash
chmod +x run_all_LPscenarios.sh \
        run_one_LPscenario.sh \
        run_one_NLPscenario.sh \
        run_main_slurmscript.sh
```

#### 2.4 Basic Usage Example (Linear MPC with pywraplp)
Run the main script **[__main__.py](__main__.py)** using a linear solver:
```bash
# Example with linear solver (pywraplp):
python __main__.py \
    --solver_library pywraplp \
    --simulation_step_start 0 \
    --simulation_steps 96 \
    --horizon_length 96 \
    --noise_percent 1 \
    --initial_soc_solver 50 \
    --initial_soc_sim 50 \
    --target_soc_solver 50 \
    --omega_1 0.5 \
    --omega_2 0.5 \
    --pywraplp_solver CBC \
    --solving_method model_predictive
```
</details>

## 3. Command-line Arguments

<details>
  <summary>Click to expand/collapse</summary>


| Argument | Type | Default | Choices&nbsp;/&nbsp;Range | Description |
|----------|------|---------|---------------------------|-------------|
| `--solver_library` | `str` | **required** | `pywraplp`, `pyomo` | Optimization backend: OR-Tools or Pyomo. |
| `--simulation_step_start` | `int` | **required** | >= 0 | Index of the first time step (15-min resolution). |
| `--simulation_steps` | `int` | **required** | >= 1 | Total number of time steps to simulate. |
| `--horizon_length` | `int` | **required** | Divisor or multiple of 96 | MPC prediction horizon (time steps). |
| `--noise_percent` | `int` | **required** | 0 – 100 | Gaussian noise level added to load / generation profiles (%). |
| `--initial_soc_solver` | `int` | 50 | 0 – 100 | Initial battery SoC assumed by the solver (%). |
| `--initial_soc_sim` | `int` | 50 | 0 – 100 | Initial battery SoC in the simulation (%). |
| `--target_soc_solver` | `int` | 50 | 0 – 100 | Target battery SoC at the end of each horizon (%). |
| `--omega_1` | `float` | 0.5 | any | Weight for the cost (energy expenditure) term. |
| `--omega_2` | `float` | 0.5 | any | Weight for the power-exchange term. |
| `--pywraplp_solver` | `str` | `"CBC"` | e.g. `CBC`, `SCIP` | OR-Tools solver (used only when `--solver_library pywraplp`). |
| `--solving_method` | `str` | `"model_predictive"` | `model_predictive`, `rule_based`, `no_storage` | Control strategy to apply. |
| `--transformer_limit_enabled` | `bool` | `False` | `True`, `False` | Activate transformer loading constraint (Pyomo only). |
| `--transformer_limit_percentage` | `float` | 100 | 0 – 100 | Transformer loading limit (%) if enabled. |
| `--energy_price_year` | `int` | 2024 | >= 2016 | Year of spot-market price data (`data/{year}_spotmarket.csv`). |

</details>

<h2>4. Citation &#128221;</h2>
<p>
If you use this framework in your research, please consider citing our paper &#128221; and giving the repository a star &#11088;:
</p>

```bibTeX
@inproceedings{Demirel2025,
      title={Linear and Nonlinear Model Predictive Control for Distributed Energy Resources in Power Grids}, 
      author={Demirel, Gökhan and Mu, Xuanhao and Sari, Tolgahan and De Carne, Giovanni and Förderer, Kevin and Hagenmeyer, Veit},
      year={2025},
      booktitle={2025 IEEE 13th International Conference on Smart Energy Grid Engineering (SEGE)}, 
      pages={1--7}
}
```
## License
This code is licensed under the **[MIT License](LICENSE)**.
For any issues or any intention of cooperation, please feel free to contact me at **[goekhan.demirel@kit.edu](goekhan.demirel@kit.edu)**.
