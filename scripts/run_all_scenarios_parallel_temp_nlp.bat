@echo off
setlocal enabledelayedexpansion

:: Loops for all combinations
for %%h in (12 24 48 96) do (
        :: Run Python script with current parameters
    start "pyomo --horizon_length %%h --omega_1 0.0 --omega_2 1.0" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.0 --omega_2 1.0 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.1 --omega_2 0.9" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.1 --omega_2 0.9 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.2 --omega_2 0.8" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.2 --omega_2 0.8 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.3 --omega_2 0.7" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.3 --omega_2 0.7 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.4 --omega_2 0.6" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.4 --omega_2 0.6 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.5 --omega_2 0.5" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.5 --omega_2 0.5 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.6 --omega_2 0.4" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.6 --omega_2 0.4 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.7 --omega_2 0.3" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.7 --omega_2 0.3 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.8 --omega_2 0.2" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.8 --omega_2 0.2 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    start "pyomo --horizon_length %%h --omega_1 0.9 --omega_2 0.1" cmd /k "python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 0.9 --omega_2 0.1 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 & exit"
    timeout 5
    python solver_manager.py --solver_library pyomo --simulation_step_start 7584 --simulation_steps 672 --horizon_length %%h --noise_percent 0 --initial_soc_solver 50 --initial_soc_sim 50 --target_soc_solver 0 --omega_1 1.0 --omega_2 0.0 --solving_method model_predictive --transformer_limit_enabled False --transformer_limit_percentage 100 --energy_price_year 2024 
    timeout 5
)

endlocal
