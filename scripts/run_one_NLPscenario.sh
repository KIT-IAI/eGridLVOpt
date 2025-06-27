#!/bin/bash
# NLP - pyomo (IPOPT solver)
# chmod +x run_main_slurmscript.sh run_one_NLPscenario.sh  # Make both scripts executable
# ./run_one_NLPscenario.sh                                 # Launches a single SLURM job

# Base call with default parameters (24h horizon, noise enabled, omega = 0.5 / 0.5)
sbatch run_main_slurmscript.sh \
  pyomo \               # Solver library (Nonlinear optimization with Pyomo)
  10080 \               # Total simulation timesteps (7 days)
  672 \                 # Evaluation starts after 672 timesteps (day 1)
  96 \                  # Prediction horizon (24h = 96 timesteps at 15-minute intervals)
  1 \                   # Noise enabled (1 = yes, 0 = no)
  50 \                  # Initial SoC for solver
  50 \                  # Initial SoC for simulation
  50 \                  # Target SoC for solver
  0.5 \                 # Weight for objective 1
  0.5 \                 # Weight for objective 2
  IPOPT \               # Solver placeholder
  model_predictive      # Control strategy 
# End of script