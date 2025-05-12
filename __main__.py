from nonlinear_model import *
from enum import Enum
import argparse
from ortools.linear_solver import pywraplp
import pickle
import os
import time

# Create a struct to set the human preference such as target SoC, horizon length, omega, etc
class HumanPreference:
    def __init__(self, horizon_length=96, omega_1=0.5, omega_2=0.5, pywraplp_solver="CBC", initial_soc_solver=50, target_soc_solver=0, initial_soc_sim=50, noise_percent=1, solving_method="model_predictive"):
        self.horizon_length = horizon_length
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.pywraplp_solver = pywraplp_solver
        self.initial_soc_solver = initial_soc_solver
        self.target_soc_solver = target_soc_solver
        self.initial_soc_sim = initial_soc_sim
        self.noise_percent = noise_percent
        self.solving_method = solving_method


class solver_library(Enum):
    pyomo = "pyomo"
    pywraplp = "pywraplp"
    
class manager_solver:

    def __init__(self, 
                 solver_library: solver_library = solver_library.pywraplp, 
                 net=None, 
                 simulation_step_start=0, 
                 simulation_steps=96, 
                 energy_prices=None, 
                 control_heatpump=True, 
                 human_preference=None,
                 transformer_limit_enabled=False,
                 transformer_limit_percentage=100,
                 energy_price_year=2024):
       
        import pandapower as pp
        import copy
        import numpy as np

        # Create new instance
        if net == None:
            self.create_sample_net()
        else:
            self.net = net

        if human_preference is None:
            human_preference = HumanPreference()

        # Set energy prices
        if energy_prices is None:
            self.inp_energy_prices = np.empty(0, dtype=float)	
        else:
            self.inp_energy_prices = energy_prices

        self.energy_price_year = energy_price_year
        self.step = 0
        self.step_global = simulation_step_start
        self.simulation_step_start = simulation_step_start
        self.simulation_steps = simulation_steps

        self.transformer_limit_enabled = transformer_limit_enabled
        self.transformer_limit_percentage = transformer_limit_percentage

        self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)

        self.profiles_load_p = self.profiles[("load", "p_mw")]
        self.profiles_load_q = self.profiles[("load", "q_mvar")]
        self.profiles_sgen_p = self.profiles[("sgen", "p_mw")]
        self.profiles_storage = self.profiles[("storage", "p_mw")]

        # simulate network without storage usage to get the power exchange and energy expenditures
        print("Simulating network to create scalers... ", end="")
        self.net_temp = pp.pandapowerNet(self.net)
        self.temp_net_sum_p_mw = np.empty(shape=self.simulation_steps, dtype=float)
        self.temp_money_spend = np.empty(shape=self.simulation_steps, dtype=float)
        for step_temp in range(self.simulation_steps):
            step_global_temp = self.step_global + step_temp
            for i in self.net_temp.load.index:
                self.net_temp.load["p_mw"][i] = self.profiles_load_p[i][step_global_temp]
                self.net_temp.load["q_mvar"][i] = self.profiles_load_q[i][step_global_temp]
            for i in self.net_temp.sgen.index:
                self.net_temp.sgen["p_mw"][i] = self.profiles_sgen_p[i][step_global_temp]
            for i in self.net_temp.storage.index:
                self.net_temp.storage["p_mw"][i] = 0
            
            pp.runpp(self.net_temp)

            self.temp_net_sum_p_mw[step_temp] = 0
            for i in self.net_temp.sgen.index:
                self.temp_net_sum_p_mw[step_temp] -= self.net_temp.sgen.p_mw[i]
            for i in self.net_temp.load.index:
                self.temp_net_sum_p_mw[step_temp] += self.net_temp.load.p_mw[i]
            
            # You can also use self.net.res_bus[0] instead of sum of sgens and loads
            self.temp_money_spend[step_temp] = self.temp_net_sum_p_mw[step_temp] * self.inp_energy_prices[step_global_temp] * 1000 * 15 / 60

        # Create min max scaler for the power exchange and energy expenditures
        self.temp_money_spend_std = np.std(self.temp_money_spend)
        self.temp_money_spend_mean = np.mean(self.temp_money_spend)
        self.temp_net_sum_p_mw_std = np.std(self.temp_net_sum_p_mw)
        self.temp_net_sum_p_mw_mean = np.mean(self.temp_net_sum_p_mw)
        print("Finish")

        print("adding noise to profiles")
        self.noise_factor = human_preference.noise_percent/100
        self.prediction_profiles_load_p_noisy = self.profiles_load_p.copy().to_numpy().transpose()
        self.prediction_profiles_load_q_noisy = self.profiles_load_q.copy().to_numpy().transpose()
        self.prediction_profiles_sgen_p_noisy = self.profiles_sgen_p.copy().to_numpy().transpose()

        for i in self.net.load.index:
            self.prediction_profiles_load_p_noisy[i] += np.random.normal(0, np.std(self.prediction_profiles_load_p_noisy[i])*self.noise_factor, self.prediction_profiles_load_p_noisy[i].shape)
            self.prediction_profiles_load_q_noisy[i] += np.random.normal(0, np.std(self.prediction_profiles_load_q_noisy[i])*self.noise_factor, self.prediction_profiles_load_q_noisy[i].shape)
            self.prediction_profiles_load_p_noisy[i] = np.max([self.prediction_profiles_load_p_noisy[i], np.zeros(self.prediction_profiles_load_p_noisy[i].shape)], axis=0)
            self.prediction_profiles_load_q_noisy[i] = np.max([self.prediction_profiles_load_q_noisy[i], np.zeros(self.prediction_profiles_load_q_noisy[i].shape)], axis=0)
            
        for i in self.net.sgen.index:
            self.prediction_profiles_sgen_p_noisy[i] += np.random.normal(0, np.std(self.prediction_profiles_sgen_p_noisy[i])*self.noise_factor, self.prediction_profiles_sgen_p_noisy[i].shape)
            self.prediction_profiles_sgen_p_noisy[i] = np.max([self.prediction_profiles_sgen_p_noisy[i], np.zeros(self.prediction_profiles_sgen_p_noisy[i].shape)], axis=0)

        self.prediction_energy_prices_noisy = self.inp_energy_prices + np.random.normal(0, np.std(self.inp_energy_prices)*self.noise_factor, self.inp_energy_prices.shape)

        # Set solver library
        self.solver_library = solver_library
        self.horizon_length = human_preference.horizon_length
        self.net.ext_grid['vm_pu'] = 1
        
        self.initial_soc_sim = human_preference.initial_soc_sim
        self.omega_1 = human_preference.omega_1 # Weight for selecting objective function
        self.omega_2 = human_preference.omega_2 # Weight for selecting objective function
        # Raise error if horizon length is not a divisor of 96
        if 96 % self.horizon_length != 0 and self.horizon_length % 96 != 0:
            raise ValueError("Horizon length must be a divisor of 96")
        if self.solver_library == solver_library.pyomo:

            """Preprocessing from pyomo code"""

            # 2. Dimensionierung der Batteriespeicher beibehalten
            self.net.storage["max_p_mw"] = -self.net.storage["min_p_mw"]
            self.net.storage["max_p_mw"] = self.net.storage["max_e_mwh"] /4 /2
            self.net.storage["min_p_mw"] = -self.net.storage["max_e_mwh"] /4 /2

            for i in self.net.storage.index:
                self.net.storage["soc_percent"][i] = self.initial_soc_sim
               
            net["profiles_load_p"]=self.prediction_profiles_load_p_noisy
            net["profiles_load_q"]=self.prediction_profiles_load_q_noisy
            net["profiles_pv_p"]=self.prediction_profiles_sgen_p_noisy
            net["prediction_energy_prices_noisy"] = self.prediction_energy_prices_noisy

            self.net_backup = pp.pandapowerNet(net)
            self.net = pp.pandapowerNet(self.net_backup)
            i_ka = np.zeros((self.net.line.shape[0],self.simulation_steps))
            trafo_loading = np.zeros((self.net.trafo.shape[0],self.simulation_steps))

            # Adjust line limits if they are exceeded such that the NLP solver does not have infeasible solutions
            # This section can be deleted for simulations where the line limits are not exceeded
            # ==============================================================================================================
            print("Simulating network for line current calculation")
            net_copy = copy.deepcopy(self.net)
            for step_net_copy in range(self.simulation_step_start, self.simulation_step_start + self.simulation_steps):
                for i in self.net.load.index:
                    net_copy.load["p_mw"][i] = self.profiles_load_p[i][step_net_copy]
                    net_copy.load["q_mvar"][i] = self.profiles_load_q[i][step_net_copy]
                for i in self.net.sgen.index:
                    net_copy.sgen["p_mw"][i] = self.profiles_sgen_p[i][step_net_copy]
                for i in self.net.storage.index:
                    net_copy.storage["p_mw"][i] = self.profiles_storage[i][step_net_copy]
                pp.runpp(net_copy)
                i_ka[:,step_net_copy-self.simulation_step_start] = net_copy.res_line["i_ka"].values
                trafo_loading[:,step_net_copy-self.simulation_step_start] = net_copy.res_trafo["loading_percent"].values
            
            max_i_ka = i_ka.max(axis=1)
            for i in self.net.line.index:
                self.net.line["max_i_ka"][i] = np.max([max_i_ka[i], self.net.line["max_i_ka"][i]])
            # Find all lines that are connected to the transformer
            hv_bus = self.net.trafo.at[0, 'hv_bus']
            lv_bus = self.net.trafo.at[0, 'lv_bus']
            self.connected_lines = self.net.line[(self.net.line['from_bus'].isin([hv_bus, lv_bus])) | 
                               (self.net.line['to_bus'].isin([hv_bus, lv_bus]))].index
            self.connected_lines_cap = 1.05 # Line capacity limit: 1.05 = 105% for all lines that are connected to the trafo
            for i in self.connected_lines:
                self.net.line["max_i_ka"][i] = max_i_ka[i] * self.connected_lines_cap
            print("Finished simulating network")
            # ==============================================================================================================

            if self.transformer_limit_enabled:
                flow_constraint = "both"
            else:
                flow_constraint = "current"
            self.model = NonLinearModel(self.net, flow_constraint=flow_constraint,solver="ipopt", v_min=0.9, v_max=1.1, H=self.horizon_length, H_all=simulation_steps, dt_min=15, objective="pywraplp_multiobjective_power_price", solver_options = {'verbose': True, 'tee': False}, timestep_start=simulation_step_start, omega_1=self.omega_1, omega_2=self.omega_2, temp_money_spend_std=self.temp_money_spend_std, temp_money_spend_mean=self.temp_money_spend_mean, temp_net_sum_p_mw_mean=self.temp_net_sum_p_mw_mean, temp_net_sum_p_mw_std=self.temp_net_sum_p_mw_std, trafo_limit_percent=self.transformer_limit_percentage)


            self.mins = []
            self.maxs = []
            self.means = []
            self.Pstos = []
            self.soc_opt = []
            self.P = []
            self.Q = []
            self.P_Auslastung_mins = []
            self.P_Auslastung_maxs = []
            self.P_Auslastung_means = []
            self.i_Auslastung_mins = []
            self.i_Auslastung_maxs = []
            self.i_Auslastung_means = [] = []


            self.i_Auslastung_mpc, self.Strafo_mpc =[],[]
            self.i_Auslastung_pyomo_voll, self.V_pyomo = [],[]
            self.pf,self.pt = [],[]

            self.solving_method = "model_predictive"

            self.horizon_length = human_preference.horizon_length

        elif self.solver_library == solver_library.pywraplp:

            """Preprocessing for pywraplp code"""
            
            self.solving_method = human_preference.solving_method
            self.pywraplp_solver = human_preference.pywraplp_solver

            self.initial_soc_solver = human_preference.initial_soc_solver
            self.target_soc_solver = human_preference.target_soc_solver

            self.max_p_mw_hp = {}

            self.money_spend_sum = 0
            self.p_subnet_abs_sum = 0

        else:
            raise ValueError("Solver library not implemented")

        self.nue = 0.001 # Battery drain loss


        self.out_battery_p_mw = np.empty((len(self.net.storage.index), self.simulation_steps), dtype=float)
        self.out_battery_soc = np.empty((len(self.net.storage.index), self.simulation_steps), dtype=float)
        self.out_pv_p_mw = np.empty((len(self.net.sgen.index), self.simulation_steps), dtype=float)
        self.out_load_p_mw = np.empty((len(self.net.load.index), self.simulation_steps), dtype=float)
        self.out_bus_m_p_mw = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_bus_p_mw = np.empty((len(self.net.bus.index), self.simulation_steps), dtype=float)
        self.out_net_sum_p_mw = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_energy_prices = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_money_spend = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_money_spend_sum = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_res_trafo_loading_percent = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_res_trafo_p_hv_mw = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_res_trafo_p_lv_mw = np.empty(shape=self.simulation_steps, dtype=float)
        self.out_res_bus_vm_pu = np.empty((len(self.net.bus.index), self.simulation_steps), dtype=float)
        self.out_res_line_loading = np.empty((len(self.net.line.index), self.simulation_steps), dtype=float)

        self.id_res_bus = 0
        
        self.index_hp = np.where(net.load.profile.str.startswith(('Soil', 'Air')))[0]
        self.max_p_mw_hp = {}
        for i in self.index_hp:
            self.max_p_mw_hp[i] = max(self.profiles_load_p[i])
        self.p_heatpump_solutions = {}

        # Disable heatpump constraint control
        self.control_heatpump = control_heatpump

    def simulate_step(self):

        # This code does a simulation step for the pywraplp solver simulating the pandapower net
        self.__update_environment()
        
        # Run power flow simulation after control solution for storage and heatpump are attained
        pp.runpp(self.net)

        """Output processing"""
        self.out_net_sum_p_mw[self.step] = 0
        
        for i in self.net.storage.index:
            self.out_battery_soc[i][self.step] = self.net.storage.soc_percent[i]
            self.out_battery_p_mw[i][self.step] = self.net.storage.p_mw[i]
            self.out_net_sum_p_mw[self.step] += self.net.storage.p_mw[i]

        self.out_bus_m_p_mw[self.step] = self.net.res_bus.p_mw[self.id_res_bus]
        self.out_energy_prices[self.step] = self.inp_energy_prices[self.step_global]
        self.out_money_spend[self.step] = -self.net.res_bus.p_mw[self.id_res_bus] * self.out_energy_prices[self.step] * 1000 * 15 / 60
        if self.step >= 1:
            self.out_money_spend_sum[self.step] = self.out_money_spend_sum[self.step-1] + self.out_money_spend[self.step]
        else:
            self.out_money_spend_sum[0] = self.out_money_spend[0]

        self.out_res_trafo_loading_percent[self.step] = self.net.res_trafo.loading_percent[0]
        self.out_res_trafo_p_hv_mw[self.step] = self.net.res_trafo.p_hv_mw[0]
        self.out_res_trafo_p_lv_mw[self.step] = self.net.res_trafo.p_lv_mw[0]

        for i in self.net.bus.index:
            self.out_res_bus_vm_pu[i][self.step] = self.net.res_bus.vm_pu[i]

        for i in self.net.load.index:
            self.out_load_p_mw[i][self.step] = self.net.load.p_mw[i]
            self.out_net_sum_p_mw[self.step] += self.net.load.p_mw[i]

        for i in self.net.sgen.index:
            self.out_pv_p_mw[i][self.step] = self.net.sgen.p_mw[i]
            self.out_net_sum_p_mw[self.step] -= self.net.sgen.p_mw[i]
        
        for i in self.net.line.index:
            self.out_res_line_loading[i][self.step] = self.net.res_line.loading_percent[i]

        print(self.step)

    def __update_environment(self):
        """Update all components"""

        if self.solver_library == solver_library.pywraplp:
            # Update battery SOC
            for i in self.net.storage.index:
                self.net.storage.soc_percent[i] = self.next_soc(self.net.storage.soc_percent[i], self.net.storage.p_mw[i], self.net.storage.max_e_mwh[i])
    
        # Load
        for i in self.net.load.index:
            self.net.load.p_mw[i] = self.profiles_load_p[i][self.step_global]
            self.net.load.q_mvar[i] = self.profiles_load_q[i][self.step_global]

        # Storage
        for i in self.net.storage.index:
             self.net.storage.p_mw[i] = self.profiles_storage[i][self.step_global]
 
        # PV
        for i in self.net.sgen.index:
            self.net.sgen.p_mw[i] = self.profiles_sgen_p[i][self.step_global]

        if self.solver_library == solver_library.pyomo:
            self.model.fix_values_timestep(self.step_global)
            try:
                result = self.model.run_opt()

                solver_condition = result.solver.termination_condition.value

                res_bus, res_line, res_obj_fcn = self.model.extract_results()
                
                # self.means.append(res_bus["V_pu"][1:,0].mean())
                # self.maxs.append(res_bus["V_pu"][1:,0].max())
                # self.mins.append(res_bus["V_pu"][1:,0].min())
                # self.V_pyomo.append(res_bus["V_pu"][1:,0])

                # self.P.append(res_bus['P_mw'][:,0])
                # self.Q.append(res_bus['Q_mvar'][:,0])
                
                # Ptrafo_lv = res_bus['P_mw'][4,0]
                # Qtrafo_lv = res_bus['Q_mvar'][4,0]
                
                # self.Strafo_mpc.append([Ptrafo_lv,Qtrafo_lv,np.sqrt(Ptrafo_lv**2+Qtrafo_lv**2)])

                # P_Auslastung = np.sqrt((res_line['pf_mw'][:,0])**2+(res_line['pt_mw'][:,0]**2))
                # self.pf.append(res_line['pf_mw'][:,0])
                # self.pt.append(res_line['pt_mw'][:,0])
                # self.P_Auslastung_means.append(P_Auslastung.mean())
                # self.P_Auslastung_maxs.append(P_Auslastung.max())
                # self.P_Auslastung_mins.append(P_Auslastung.min())

                # if_Auslastung = np.sqrt(((res_line['i_f_real_pu'][:,0])**2+(res_line['i_f_imag_pu'][:,0]**2))*(1/(0.4*np.sqrt(3)))**2)
                # it_Auslastung = np.sqrt(((res_line['i_t_real_pu'][:,0])**2+(res_line['i_t_imag_pu'][:,0]**2))*(1/(0.4*np.sqrt(3)))**2)
                # i_Auslastung = np.array([np.max(np.array([if_Auslastung,it_Auslastung])[:,i]) for i in range(0, self.net.line.shape[0])])
                # self.i_Auslastung_mpc.append(i_Auslastung)
                # self.i_Auslastung_means.append(i_Auslastung.mean())
                # self.i_Auslastung_maxs.append(i_Auslastung.max())
                # self.i_Auslastung_mins.append(i_Auslastung.min())
                # self.i_Auslastung_pyomo_voll.append(i_Auslastung)

                self.Pstos.append(res_bus['P_sto'][:,0])
                self.soc_opt.append(res_bus['E_sto'][:,0]/self.net.storage['max_e_mwh'].values*100)
                
                self.model.update_storages()
                self.model.clear_model()

                for i in self.net.storage.index:
                    self.net.storage.soc_percent[i] = self.soc_opt[self.step][i]
                    self.net.storage.p_mw[i] = self.Pstos[self.step][i]
                for i in self.index_hp:
                    self.net.load.p_mw[i] = res_bus["P_load"][i][0]
            except Exception as e:
                print(f"Error: {e}")
                
                # # error handling
                # self.means.append(self.means[-1])
                # self.maxs.append(self.maxs[-1])
                # self.mins.append(self.mins[-1])
                # self.V_pyomo.append(self.V_pyomo[-1])

                # self.P.append(self.P[-1])
                # self.Q.append(self.Q[-1])
                
                # self.Strafo_mpc.append(self.Strafo_mpc[-1])

                # self.pf.append(self.pf[-1])
                # self.pt.append(self.pt[-1])
                # self.P_Auslastung_means.append(self.P_Auslastung_means[-1])
                # self.P_Auslastung_maxs.append(self.P_Auslastung_maxs[-1])
                # self.P_Auslastung_mins.append(self.P_Auslastung_mins[-1])

                # self.i_Auslastung_mpc.append(self.i_Auslastung_mpc[-1])
                # self.i_Auslastung_means.append(self.i_Auslastung_means[-1])
                # self.i_Auslastung_maxs.append(self.i_Auslastung_maxs[-1])
                # self.i_Auslastung_mins.append(self.i_Auslastung_mins[-1])
                # self.i_Auslastung_pyomo_voll.append(self.i_Auslastung_pyomo_voll[-1])

                self.Pstos.append(0)
                self.soc_opt.append(self.soc_opt[-1])
                
                self.model.clear_model()

                for i in self.index_hp:
                    self.model.p_hp_history[i].append(0)

                #if False and solver_condition == "infeasible": # potential error handling
                for i in self.net.storage.index:
                    self.net.storage.soc_percent[i] = self.next_soc(self.net.storage.soc_percent[i], self.net.storage.p_mw[i], self.net.storage.max_e_mwh[i])
                    if self.net.storage.soc_percent[i] > 100.:
                        self.net.storage.soc_percent[i] = 100.
                    elif self.net.storage.soc_percent[i] < 0.:
                        self.net.storage.soc_percent[i] = 0.
                    self.net.storage.p_mw[i] = 0
                for i in self.index_hp:
                    self.net.load.p_mw[i] = 0
        elif self.solver_library == solver_library.pywraplp:
            p_storage_solutions, p_heatpump_solutions = self.solve()
            for i in self.net.storage.index:
                self.net.storage.p_mw[i] = p_storage_solutions[i]
                        
            for i in self.index_hp:
                if self.control_heatpump:
                    self.net.load.p_mw[i] = p_heatpump_solutions[i]
                else:
                    self.net.load.p_mw[i] = self.profiles_load_p[i][self.step_global]
                    self.net.load.q_mvar[i] = self.profiles_load_q[i][self.step_global]
        
            # Cap Battery p_mw to not go below 0% or above 100% SoC
            for i in self.net.storage.index:
                if self.next_soc(self.net.storage.soc_percent[i], self.net.storage.p_mw[i], self.net.storage.max_e_mwh[i]) > 100.:
                    self.net.storage.p_mw[i] = self.max_p_mw_for_100_soc(self.net.storage.soc_percent[i], self.net.storage.max_e_mwh[i])
                elif self.next_soc(self.net.storage.soc_percent[i], self.net.storage.p_mw[i], self.net.storage.max_e_mwh[i]) < 0.:
                    self.net.storage.p_mw[i] = self.min_p_mw_for_0_soc(self.net.storage.soc_percent[i], self.net.storage.max_e_mwh[i])
            for i in self.net.storage.index:
                if self.net.storage.soc_percent[i] > 100.:
                    self.net.storage.soc_percent[i] = 100.
                elif self.net.storage.soc_percent[i] < 0.:
                    self.net.storage.soc_percent[i] = 0.
        else:
            raise ValueError("Solver library not implemented")

        
    # Calculate the soc after one timestep of 15min
    def next_soc(self, soc, p_mw, max_e_mwh):
        return soc + (p_mw * 15 / 60) / max_e_mwh * 100 - self.nue * soc * max_e_mwh * 15/60

    def max_p_mw_for_100_soc(self, soc, max_e_mwh):
        return (100 - soc) * max_e_mwh * (60 / 15) / 100
        
    def min_p_mw_for_0_soc(self, soc, max_e_mwh):
        return (-soc) * max_e_mwh * (60 / 15) / 100

    def solve(self):
        """Solve network using the selected method"""
        self.solver = pywraplp.Solver.CreateSolver(self.pywraplp_solver)

        if self.solving_method == "model_predictive":
            time_steps_solver = self.horizon_length

            power_sgen = {}
            for i in self.net.sgen.index:
                power_sgen[i] = np.array(self.prediction_profiles_sgen_p_noisy[i][self.step_global:self.step_global+self.horizon_length])
            
            power_load = {}
            for i in self.net.load.index:
                power_load[i] = np.array(self.prediction_profiles_load_p_noisy[i][self.step_global:self.step_global+self.horizon_length])

            energy_prices = self.prediction_energy_prices_noisy[self.step_global:self.step_global+self.horizon_length]
            run_solver = 1

            initial_soc = {}
            for i in self.net.storage.index:
                if self.step == 0:
                    initial_soc[i] = self.initial_soc_sim
                else:
                    initial_soc[i] = max(min(self.next_soc(self.net.storage.soc_percent[i], self.net.storage.p_mw[i], self.net.storage.max_e_mwh[i]), 100.), 0.)


            # heatpumps will be controlled in 6h blocks
            e_hp_blocks_start_global = {}
            e_hp_blocks_start_local = {}
            e_hp_profile = {}
            e_hp_used = {}
            # Calculate the energy draw of heatpumps
            for i in self.index_hp:
                e_hp_blocks_start_global[i] = self.step_global // 24 * 24
                e_hp_blocks_start_local[i] = self.step // 24 * 24
                e_hp_used[i] = 0
                e_hp_profile[i] = 0
                for t in range(e_hp_blocks_start_local[i], self.step):
                    e_hp_used[i] += self.out_load_p_mw[i][t] * 15 / 60
                for t in range(e_hp_blocks_start_global[i], e_hp_blocks_start_global[i] + 24):
                    e_hp_profile[i] += self.prediction_profiles_load_p_noisy[i][t] * 15 / 60
        elif self.solving_method == "no_storage":
                run_solver = 0
        
        else:
            raise ValueError(f"Solving method {self.solving_method} not implemented")

        p_storage_solution = {}
        p_heatpump_solution = {}

        if run_solver:
            
            p_storage_max_change = 0.8

            # Storage power variables
            p_storage = {}
            for i in self.net.storage.index:
                p_storage[i] = [self.solver.NumVar(-self.net.storage.max_e_mwh[i]/2/4, self.net.storage.max_e_mwh[i]/2/4, f'p_storage_index{i}_{t}') for t in range(time_steps_solver)]
            b_heatpump = {}
            for i in self.index_hp:
                b_heatpump[i] = [self.solver.BoolVar(f"b_heatpump_index{i}_{t}") for t in range(max(time_steps_solver, 24))]
            p_subnet = [self.solver.NumVar(-self.solver.infinity(), self.solver.infinity(), f'p_subnet_{t}') for t in range(time_steps_solver)]
            money_spend = [self.solver.NumVar(-self.solver.infinity(), self.solver.infinity(), f'money_spend_{t}') for t in range(time_steps_solver)]
            p_subnet_pos = [self.solver.NumVar(0, self.solver.infinity(), f'p_subnet_pos_{t}') for t in range(time_steps_solver)]
            p_subnet_neg = [self.solver.NumVar(-self.solver.infinity(), 0, f'p_subnet_neg_{t}') for t in range(time_steps_solver)]

            p_storage_change = {}
            for i in self.net.storage.index:
                p_storage_change[i] = [self.solver.NumVar(-self.solver.infinity(), self.solver.infinity(), f'p_storage_change_index{i}_{t}') for t in range(2, time_steps_solver)]

            # State of charge variables
            soc_storage = {}
            for i in self.net.storage.index:
                soc_storage[i] = [self.solver.NumVar(0, 100, f"soc_index{i}_{t}") for t in range(time_steps_solver)]

            if self.control_heatpump:
                # Heatpump energy consumption needs to be within a certain tolerance to the real consumption from the profiles
                constraints_heatpump = {} 
                for i in self.index_hp:
                    constraints_heatpump[i] = self.solver.Constraint(e_hp_profile[i]-0.51*self.max_p_mw_hp[i]*15/60-e_hp_used[i], e_hp_profile[i]+0.51*self.max_p_mw_hp[i]*15/60-e_hp_used[i])
                    for t in range(0, e_hp_blocks_start_global[i]+24-self.step_global):
                        constraints_heatpump[i].SetCoefficient(b_heatpump[i][t], self.max_p_mw_hp[i] * 15 / 60)
                    # Remaining values do not necessarily need to be set, they can be minimized either way

    
            # Constraints for state of charge dynamics
            for t in range(time_steps_solver):

                p_subnet_without_storage = 0
                if self.control_heatpump:
                    for i in self.net.load.index:
                        if not i in self.index_hp:
                            p_subnet_without_storage += power_load[i][t]
                else:
                    for i in self.net.load.index:
                        p_subnet_without_storage += power_load[i][t]
                for i in self.net.sgen.index:
                    p_subnet_without_storage -= power_sgen[i][t]
                constraint_subnet = self.solver.Constraint(p_subnet_without_storage, p_subnet_without_storage)
                for i in self.net.storage.index:
                    constraint_subnet.SetCoefficient(p_storage[i][t], -1)
                if self.control_heatpump:
                    for i in self.index_hp:
                        constraint_subnet.SetCoefficient(b_heatpump[i][t], -self.max_p_mw_hp[i])
                constraint_subnet.SetCoefficient(p_subnet[t], 1)

                self.solver.Add(money_spend[t] == p_subnet[t] * energy_prices[t] * 1000 * 15 / 60)

                self.solver.Add(p_subnet_pos[t] >= p_subnet[t])
                self.solver.Add(p_subnet_neg[t] <= p_subnet[t])
                
            # SoC change in every step
            for i in self.net.storage.index:
                for t in range(1, time_steps_solver):
                    self.solver.Add(soc_storage[i][t] == soc_storage[i][t-1] + (p_storage[i][t-1] * 15 / 60) / self.net.storage.max_e_mwh[i] * 100)

            # Initial state of charge constraint
            if self.solving_method == "model_predictive":
                for i in self.net.storage.index:
                    self.solver.Add(soc_storage[i][0] == initial_soc[i])

            # Norm variables
            p_subnet_normed = [self.solver.NumVar(-self.solver.infinity(), self.solver.infinity(), f'p_subnet_normed_{t}') for t in range(time_steps_solver)]
            for t in range(time_steps_solver):
                self.solver.Add(p_subnet_normed[t] == ( p_subnet_pos[t] - p_subnet_neg[t] - self.temp_net_sum_p_mw_mean ) / self.temp_net_sum_p_mw_std)
            money_spend_normed = [self.solver.NumVar(-self.solver.infinity(), self.solver.infinity(), f'money_spend_normed_{t}') for t in range(time_steps_solver)]
            for t in range(time_steps_solver):
                self.solver.Add(money_spend_normed[t] == 4*( money_spend[t] - self.temp_money_spend_mean ) / self.temp_money_spend_std)
            
            # Objective to minimize the total cost
            objective = self.solver.Objective()
            for t in range(time_steps_solver):
                objective.SetCoefficient(money_spend_normed[t], self.omega_1)
                objective.SetCoefficient(p_subnet_normed[t], self.omega_2)
            objective.SetMinimization()

            # Solve the optimization problem
            self.solver.set_time_limit(15*1000) # 15 seconds time limit for solver
            status = self.solver.Solve()

            self.money_spend_sum += money_spend[0].solution_value()
            self.p_subnet_abs_sum += p_subnet_pos[0].solution_value() - p_subnet_neg[0].solution_value()

            p_storage_solutions = {}
            p_heatpump_solutions = {}
            self.p_storage_solutions = p_storage_solutions
            self.p_heatpump_solutions = p_heatpump_solutions
            if status == pywraplp.Solver.OPTIMAL:
                pass
                for i in self.net.storage.index:
                    p_storage_solutions[i] = np.empty(shape=time_steps_solver)
                    for t in range(time_steps_solver):
                        p_storage_solutions[i][t] = p_storage[i][t].solution_value()
                for i in self.index_hp:
                    p_heatpump_solutions[i] = np.empty(shape=time_steps_solver)
                    for t in range(time_steps_solver):
                        p_heatpump_solutions[i][t] = b_heatpump[i][t].solution_value() * self.max_p_mw_hp[i]
            else:
                print("ERROR: Solver had no success, status: ", status)
                for i in self.net.storage.index:
                    p_storage_solutions[i] = np.zeros(shape=time_steps_solver)
                for i in self.index_hp:
                    p_heatpump_solutions[i] = np.zeros(shape=time_steps_solver)
            
        if self.solving_method == "model_predictive":
            for i in self.net.storage.index:
                p_storage_solution[i] = self.p_storage_solutions[i][0]
            for i in self.index_hp:
                p_heatpump_solution[i] = self.p_heatpump_solutions[i][0]
        elif self.solving_method == "no_storage":
            for i in self.net.storage.index:
                p_storage_solution[i] = 0
            for i in self.index_hp:
                p_heatpump_solution[i] = self.profiles_load_p[i][self.step_global]


        return p_storage_solution, p_heatpump_solution


    def simulate(self):

        # Save
        for i in self.net.storage.index:
            self.net.storage.soc_percent[i] = self.initial_soc_sim # SOCs are initialized at weird values otherwise

        while self.step < self.simulation_steps:

            # Execute a simulation step
            self.simulate_step()

            self.step += 1
            self.step_global = self.step + self.simulation_step_start

        print("Money spent: ", self.out_money_spend_sum[-1]/100)

        output_dir = f"results/{self.solver_library.value}/O1_{self.omega_1}andO2_{self.omega_2}"
        os.makedirs(output_dir, exist_ok=True)

        output_data = {"out_battery_p_mw": self.out_battery_p_mw,
        "out_battery_soc": self.out_battery_soc,
        "out_pv_p_mw": self.out_pv_p_mw,
        "out_load_p_mw": self.out_load_p_mw,
        "out_bus_m_p_mw": self.out_bus_m_p_mw,
        "out_bus_p_mw": self.out_bus_p_mw,
        "out_net_sum_p_mw": self.out_net_sum_p_mw,
        "out_energy_prices": self.out_energy_prices,
        "out_money_spend": self.out_money_spend,
        "out_money_spend_sum": self.out_money_spend_sum,
        "out_res_trafo_loading_percent": self.out_res_trafo_loading_percent,
        "out_res_trafo_p_hv_mw": self.out_res_trafo_p_hv_mw,
        "out_res_trafo_p_lv_mw": self.out_res_trafo_p_lv_mw,
        "out_res_bus_vm_pu": self.out_res_bus_vm_pu,
        "out_res_line_loading": self.out_res_line_loading}

        # Final data save
        filename = (
            f"{output_dir}/final_{self.energy_price_year}_{self.solving_method}_"
            f"{self.simulation_step_start}-{self.simulation_step_start + self.simulation_steps}_"
            f"{self.noise_factor*100}percentnoise_"
            f"{self.horizon_length/4:.0f}h.pkl"
        )
        with open(filename, 'wb') as f:
            pickle.dump(output_data, f)
        print("Saved to ", filename)

def get_energy_prices(year):
    # Load energy prices
    import pandas as pd
    data = pd.read_csv(f'./data/{year}_spotmarket.csv', index_col=0, parse_dates=True, sep=';')
    data.index = data.index + ' ' + data['von']
    data.index = pd.to_datetime(data.index, format='%d.%m.%Y %H:%M')
    data['price'] = data['Spotmarktpreis in ct/kWh']
    data = data.drop(['von', 'Spotmarktpreis in ct/kWh', 'bis', 'Zeitzone von', 'Zeitzone bis'], axis=1)
    data.to_csv('./data/spotmarket_reduced.csv')
    data = pd.read_csv('./data/spotmarket_reduced.csv', index_col=0, parse_dates=True)
    data = data[~data.index.duplicated(keep='first')]
    data['price'] = data['price'].str.replace(',', '.').astype(float)
    data = data.resample('15min').interpolate(method='pad')
    data.to_csv('./data/spotmarket_reduced_quarters.csv')
    data = pd.read_csv('./data/spotmarket_reduced_quarters.csv', index_col=0, parse_dates=True)
    data = data[f'{year}-01-01':] #f'{year}-12-31']
    data.to_csv(f'./data/spotmarket_reduced_quarters_{year}.csv')
    data = pd.read_csv(f'./data/spotmarket_reduced_quarters_{year}.csv', index_col=0, parse_dates=True)
    #data['price'].plot(title='Original Price Data')
    energy_prices = data["price"].to_numpy()
    return energy_prices

def main():

    debug = False
    if debug:
        import simbench as sb
        net = sb.get_simbench_net("1-LV-rural1--1-sw")
        energy_prices = get_energy_prices(2016)
        human_preference = HumanPreference(horizon_length=int(96/1), omega_1=0.5, omega_2=0.5, pywraplp_solver="CBC", initial_soc_solver=50, target_soc_solver=50, initial_soc_sim=50, solving_method="no_storage", noise_percent=0)
        instance =  manager_solver(solver_library=solver_library.pywraplp,
                                    net=net, 
                                    simulation_steps=96*7,
                                    simulation_step_start=7104, #7584, #7104, #10080,
                                    energy_prices=energy_prices,
                                    human_preference=human_preference)
        instance.simulate()
    else:
        # Create the parser
        parser = argparse.ArgumentParser(description="Solver Manager")

        parser.add_argument('--solver_library', type=str, choices=["pywraplp", "pyomo"],
                            help='Library to use for solving', required=True)
        parser.add_argument('--simulation_step_start', type=int,
                            help='Simulation step index to start at', required=True)
        parser.add_argument('--simulation_steps', type=int,
                            help='Number of simulation time steps (each = 15 min)', required=True)
        parser.add_argument('--horizon_length', type=int,
                            help='Number of time steps in the solver horizon', required=True)
        parser.add_argument('--noise_percent', type=int,
                            help='Percentage of noise to add to load/generation profiles', required=True)
        parser.add_argument('--initial_soc_solver', type=int, choices=range(0, 100), default=50,
                            help='SoC the solver will assume at the beginning of each horizon')
        parser.add_argument('--initial_soc_sim', type=int, choices=range(0, 100), default=50,
                            help='SoC of storages at the beginning of the entire simulation')
        parser.add_argument('--target_soc_solver', type=int, choices=range(0, 100), default=50,
                            help='SoC the solver will aim for at the end of each horizon')
        parser.add_argument('--omega_1', type=float, default=0.5,
                            help='Weighting factor omega_1 for the optimization objective (default=0.5)')
        parser.add_argument('--omega_2', type=float, default=0.5,
                            help='Weighting factor omega_2 for the optimization objective (default=0.5)')
        parser.add_argument('--pywraplp_solver', type=str, default="CBC",
                            help='Solver name when solver_library=pywraplp (e.g. "CBC", "SCIP")')
        parser.add_argument('--solving_method', type=str, default="model_predictive",
                            help='Which solving method to use (e.g. "rule_based", "model_predictive")')
        parser.add_argument('--transformer_limit_enabled', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='Whether to enable the transformer limit constraint (only for nonlinear MPC) [True, False]')
        parser.add_argument('--transformer_limit_percentage', type=float, default=100,
                            help='Percentage of the transformer limit to use (only for nonlinear MPC, only used if transformer_limit_enabled=True)')
        parser.add_argument('--energy_price_year', type=int, default=2024,
                            help='Which year energy price data to use. Loaded from data/{year}_spotmarket.csv')

        args = parser.parse_args()

        # Access the arguments
        if args.solver_library == "pywraplp":
            arg_solver_library = solver_library.pywraplp
        elif args.solver_library == "pyomo":
            arg_solver_library = solver_library.pyomo
        else:
            raise ValueError("Unknown solver library specified.")

        print("=== Solver Manager Configuration ===")
        print(f"Solver Library:         {args.solver_library}")
        print(f"Simulation Step Start:  {args.simulation_step_start}")
        print(f"Simulation Steps:       {args.simulation_steps}")
        print(f"Horizon Length:         {args.horizon_length}")
        print(f"Noise Percent:          {args.noise_percent}")
        print(f"Initial SoC (solver):   {args.initial_soc_solver}")
        print(f"Initial SoC (sim):      {args.initial_soc_sim}")
        print(f"Target SoC (solver):    {args.target_soc_solver}")
        print(f"omega_1:                {args.omega_1}")
        print(f"omega_2:                {args.omega_2}")
        print(f"solving_method:         {args.solving_method}")
        if args.solver_library == "pywraplp":
            print(f"pywraplp_solver:        {args.pywraplp_solver}")
        elif args.solver_library == "pyomo":
            print(f"pyomo_solver:        IPOPT")
            print(f"transformer_limit_enabled: {args.transformer_limit_enabled}")
            if args.transformer_limit_enabled:
                print(f"transformer_limit_percentage: {args.transformer_limit_percentage}")
        print(f"energy price year:         {args.energy_price_year}")
        print("====================================")

        import simbench as sb

        net = sb.get_simbench_net("1-LV-rural1--1-sw")
        energy_prices = get_energy_prices(args.energy_price_year)

        prefs = HumanPreference(
            horizon_length=args.horizon_length,
            omega_1=args.omega_1,
            omega_2=args.omega_2,
            pywraplp_solver=args.pywraplp_solver,
            initial_soc_solver=args.initial_soc_solver,
            target_soc_solver=args.target_soc_solver,
            initial_soc_sim=args.initial_soc_sim,
            solving_method=args.solving_method,
            noise_percent=args.noise_percent
        )

        start = time.time()
        instance = manager_solver(
            solver_library=arg_solver_library,
            net=net,
            simulation_steps=args.simulation_steps,
            simulation_step_start=args.simulation_step_start,
            energy_prices=energy_prices,
            human_preference=prefs,
            transformer_limit_enabled=args.transformer_limit_enabled,
            transformer_limit_percentage=args.transformer_limit_percentage,
            energy_price_year=args.energy_price_year,
        )
        instance.simulate()
        stop = time.time()
        print(f"Simulation took {stop - start} seconds")
        command_line = (
            f"python solver_manager.py "
            f"--solver_library {args.solver_library} "
            f"--simulation_step_start {args.simulation_step_start} "
            f"--simulation_steps {args.simulation_steps} "
            f"--horizon_length {args.horizon_length} "
            f"--noise_percent {args.noise_percent} "
            f"--initial_soc_solver {args.initial_soc_solver} "
            f"--initial_soc_sim {args.initial_soc_sim} "
            f"--target_soc_solver {args.target_soc_solver} "
            f"--omega_1 {args.omega_1} "
            f"--omega_2 {args.omega_2} "
            f"--solving_method {args.solving_method}"
        )
        with open("times.log", "a") as f:
            f.write(f"{command_line}\n{stop - start}\n\n\n")

if __name__ == "__main__":
    main()
