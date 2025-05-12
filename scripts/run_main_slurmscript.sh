#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --time=72:00:00
#SBATCH --output=slurm/lp_nlp_job_%j.out
#SBATCH --error=slurm/lp_nlp_job_%j.err
#SBATCH --job-name=lpnlp-run-%j
#SBATCH --gres=gpu:full:1

# chmod +x run_main_slurmscript.sh run_all_LPscenarios.sh

# Check if the correct number of arguments is provided
if [ "$#" -ne 12 ]; then
    echo "Error: Expected 12 arguments, got $#"
    echo "Usage: $0 solver_library simulation_step_start simulation_steps horizon_length noise_percent initial_soc_solver initial_soc_sim target_soc_solver omega_1 omega_2 pywraplp_solver solving_method"
    exit 1
fi

# Read command-line arguments
SOLVER_LIBRARY=$1
SIMULATION_STEP_START=$2
SIMULATION_STEPS=$3
HORIZON_LENGTH=$4
NOISE_PERCENT=$5
INITIAL_SOC_SOLVER=$6
INITIAL_SOC_SIM=$7
TARGET_SOC_SOLVER=$8
OMEGA_1=$9
OMEGA_2=${10}
PYWRAPLP_SOLVER=${11}
SOLVING_METHOD=${12}

# Activate the Python environment
PYTHON_ENV="../lp_nlp_env"
echo "Activating Python environment: $PYTHON_ENV"
source $PYTHON_ENV/bin/activate

# Define the solver script path
SOLVER_SCRIPT="/hkfs/home/haicore/iai/ii6824/eGridLVOpt/solver_manager.py"

# Display job parameters
echo "Starting job with parameters:"
echo "  --solver_library $SOLVER_LIBRARY"
echo "  --simulation_step_start $SIMULATION_STEP_START"
echo "  --simulation_steps $SIMULATION_STEPS"
echo "  --horizon_length $HORIZON_LENGTH"
echo "  --noise_percent $NOISE_PERCENT"
echo "  --initial_soc_solver $INITIAL_SOC_SOLVER"
echo "  --initial_soc_sim $INITIAL_SOC_SIM"
echo "  --target_soc_solver $TARGET_SOC_SOLVER"
echo "  --omega_1 $OMEGA_1"
echo "  --omega_2 $OMEGA_2"
echo "  --pywraplp_solver $PYWRAPLP_SOLVER"
echo "  --solving_method $SOLVING_METHOD"

# Run the solver script with all parameters
python3.9 "$SOLVER_SCRIPT" \
  --solver_library "$SOLVER_LIBRARY" \
  --simulation_step_start "$SIMULATION_STEP_START" \
  --simulation_steps "$SIMULATION_STEPS" \
  --horizon_length "$HORIZON_LENGTH" \
  --noise_percent "$NOISE_PERCENT" \
  --initial_soc_solver "$INITIAL_SOC_SOLVER" \
  --initial_soc_sim "$INITIAL_SOC_SIM" \
  --target_soc_solver "$TARGET_SOC_SOLVER" \
  --omega_1 "$OMEGA_1" \
  --omega_2 "$OMEGA_2" \
  --pywraplp_solver "$PYWRAPLP_SOLVER" \
  --solving_method "$SOLVING_METHOD"
# End of script