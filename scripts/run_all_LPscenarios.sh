#!/bin/bash
# chmod +x run_main_slurmscript.sh run_all_LPscenarios.sh  # Make both scripts executable
# ./run_all_LPscenarios.sh                                 # Launches all 91 SLURM jobs

# Iterate over all parameter combinations
for noise in 0 1; do                                        # Run with (1) and without (0) noise
  for hours in 3 6 12 24 48; do                             # Set prediction horizons: 3h to 48h
    for i in {1..9}; do  # i=1 (0.1/0.9) to i=9 (0.9/0.1)    # Generate omega pairs from 0.1/0.9 to 0.9/0.1
      # Compute omega weights
      omega1=$(echo "scale=1; $i/10" | bc)
      omega2=$(echo "scale=1; 1 - $omega1" | bc)
      horizon_length=$(( hours * 4 ))  # Convert hours to 15-minute intervals (4 steps per hour)

      # Submit job to SLURM
      sbatch run_main_slurmscript.sh \
        pywraplp \                    # Solver library
        10080 \                       # Total simulation length in timesteps (7 days = 96 * 7)
        672 \                         # Evaluation starts after 672 steps (1 day)
        $horizon_length \             # Prediction horizon (in timesteps)
        $noise \                      # Noise setting: 0 = off, 1 = on
        50 \                          # Initial SoC (solver)
        50 \                          # Initial SoC (simulation)
        50 \                          # Target SoC (solver)
        $omega1 \                     # Weight for first objective
        $omega2 \                     # Weight for second objective
        CBC \                         # Solver backend (e.g., CBC for pywraplp)
        model_predictive              # Control strategy to use (e.g., MPC)
    done
  done
done
# End of script