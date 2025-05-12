"""
This module provides the NonLinearModel class for formulating and solving 
a non-linear (quadratic) AC optimal power flow (OPF) problem with optional 
battery storage, heat pump control, and time series data. The current 
implementation uses IPOPT as the non-linear solver. 
"""

import pyomo.environ as pe
import pyomo.opt as po
import pandapower as pp
import numpy as np
import pandas as pd
from collections import defaultdict
import simbench as sb
from pyomo.util.infeasible import log_infeasible_constraints
import logging

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

class NonLinearModel:
    def __init__(
        self, 
        Enet: pp.auxiliary.pandapowerNet, 
        flow_constraint=False,
        v_min=0.9,
        v_max=1.1,
        H: int=96,
        H_all: int=96,
        dt_min: float=15,
        objective='quadratic_exchange',                
        solver_options: dict = {'verbose': True, 'tee': False},
        withTimeseries=True,
        solver="ipopt",
        with_storage_ramp="False",
        timestep_start=0,
        omega_1=0.5,
        omega_2=0.5,
        temp_money_spend_mean=0,
        temp_money_spend_std=1,
        temp_net_sum_p_mw_mean=0,
        temp_net_sum_p_mw_std=1,
        trafo_limit_percent=100
    ):
        """
        Initializes the NonLinearModel class to set up a non-linear AC OPF problem
        using Pyomo. The user can choose different objective functions, specify
        time-series data (including heat pump profiles), include or exclude flow 
        constraints, and optionally handle ramp-rate limits for battery storage. 
        The model is solved with IPOPT, a non-linear solver, supporting
        quadratic objective terms.

        Parameters
        ----------
        Enet : pp.auxiliary.pandapowerNet
            A pandapower network instance. Must have sn_mva = 1 for p.u. consistency.
        flow_constraint : str, default False
            Indicates which flow constraints to enforce:
            - 'power': Enforce power (apparent power) flow limits (for transformers).
            - 'current': Enforce current flow limits (for lines).
            - 'both': Enforce both of the above.
            - False: No flow constraints are enforced.
        v_min : float, default 0.9
            Default minimum per-unit voltage if no bus constraints are set in Enet.
        v_max : float, default 1.1
            Default maximum per-unit voltage if no bus constraints are set in Enet.
        H : int, default 96
            Time horizon for the OPF (e.g., number of 15-minute intervals in one day).
        H_all : int, default 96
            Extended horizon for specialized calculations, particularly for heat pump scheduling.
        dt_min : float, default 15
            Duration of each time step in minutes.
        objective : str, default 'quadratic_exchange'
            Objective function type. Options:
            - 'quadratic_exchange'
            - 'transmission_losses'
            - 'min_cost'
            - 'variable_cost'
            - 'binary_cost'
        solver_options : dict, default {'verbose': True, 'tee': False}
            Solver options passed to Pyomo solver (e.g., "ipopt").
        withTimeseries : bool, default True
            Whether time-series data is included in the pandapower network.
        solver : str, default "ipopt"
            Name of the solver to use (IPOPT, etc.). This solver must be able to handle
            non-linear (quadratic) formulations.
        with_storage_ramp : str, default "False"
            If set to "True", enforces ramp-rate constraints on battery storage.
        energy_prices : array-like or None
            Energy price data for each time step when using certain objective functions.
        timestep_start : int, default 0
            The initial time index from which the time horizon is considered.
        omega_1 : float, default 0.5
            Weighting factor for the first part of a multi-objective function (e.g., cost).
        omega_2 : float, default 0.5
            Weighting factor for the second part of a multi-objective function (e.g., power).

        Notes
        -----
        - The code relies on pandapower’s internal data structures, including
          ppc['internal']['Yf'], ppc['internal']['Yt'] to retrieve admittance matrices.
        - If the pandapower network does not already contain certain columns or
          time series, default values/assumptions are used.
        - Heat pumps are identified in the 'load' table based on their profile names
          and are subject to additional energy constraints over specified blocks of time.
        """


        self.nue = 0.001 # Battery drain loss

        self.omega_1 = omega_1
        self.omega_2 = omega_2

        self.trafo_limit_percent = trafo_limit_percent

        # Allowed deviation from the target energy for heat pumps
        self.e_hp_tolerance = 0.1  # 10% deviation allowed

        self.timestep_start = timestep_start

        self.Enet = pp.pandapowerNet(Enet)

        self.temp_money_spend_mean=temp_money_spend_mean
        self.temp_money_spend_std=temp_money_spend_std
        self.temp_net_sum_p_mw_mean=temp_net_sum_p_mw_mean
        self.temp_net_sum_p_mw_std=temp_net_sum_p_mw_std

        # Ensure the nominal power base in p.u. is set to 1 MVA
        assert self.Enet.sn_mva == 1, (
            f"Nominal power base sn_mva is {self.Enet.sn_mva}, but it must be 1."
        )

        # Validate objective argument
        allowed_objectives = [
            'quadratic_exchange',
            'transmission_losses',
            'min_cost',
            'variable_cost',
            'binary_cost',
            "pywraplp_multiobjective_power_price"
        ]
        assert objective in allowed_objectives, (
            f'Invalid objective "{objective}". Must be one of {allowed_objectives}.'
        )

        # If objective is 'min_cost' but the Enet has an empty poly_cost, revert to 'quadratic_exchange'
        if objective == "min_cost" and self.Enet.poly_cost.empty:
            print('No cost data in Enet.poly_cost. Objective set to "quadratic_exchange".')
            self.objective = 'quadratic_exchange'
        else:
            self.objective = objective

        # Validate flow_constraint argument
        allowed_flowconstr = ['power', 'current', False, 'both']
        assert flow_constraint in allowed_flowconstr, (
            f'Invalid flow_constraint "{flow_constraint}". Must be one of {allowed_flowconstr}.'
        )
        self.flow_constraint = flow_constraint

        # Check presence and consistency of time series if requested
        self.withTimeseries = withTimeseries
        necessary_time_series_profiles = ['profiles_load_p','profiles_load_q']
        necessary_time_series_sgen_profiles = ['profiles_pv_p','profiles_pv_q']
        necessary_time_series_gen_profiles = ['profiles_gen_p','profiles_gen_q']
        necessary_time_series_sto_profiles = ['profiles_sto_soc']

        if self.withTimeseries:
            self.H = H
            self.dt_min = dt_min

            # Verify that all required columns for time-series exist
            if set(necessary_time_series_profiles).issubset(self.Enet.keys()):
                assert (
                    self.Enet['profiles_load_p'].shape[1] ==
                    self.Enet['profiles_load_q'].shape[1]
                ), "Lengths of P and Q for load profiles do not match."
                assert (
                    self.Enet['load'].shape[0] ==
                    self.Enet['profiles_load_p'].shape[0]
                ), "Number of loads does not match the number of load profile rows."

                self.t_all = self.Enet['profiles_load_p'].shape[1]

                if not self.Enet.sgen.empty and set(necessary_time_series_sgen_profiles).issubset(self.Enet.keys()):
                    assert (
                        self.Enet['profiles_load_p'].shape[1] ==
                        self.Enet['profiles_pv_p'].shape[1]
                    ), "Mismatch in length of sgen (PV) and load profiles."
                    assert (
                        self.Enet['sgen'].shape[0] ==
                        self.Enet['profiles_pv_p'].shape[0]
                    ), "Number of sgen elements does not match the shape of PV profiles."

                if not self.Enet.gen.empty and set(necessary_time_series_gen_profiles).issubset(self.Enet.keys()):
                    assert (
                        self.Enet['profiles_load_p'].shape[1] ==
                        self.Enet['profiles_gen_p'].shape[1]
                    ), "Mismatch in length of generator and load profiles."
                    assert (
                        self.Enet['gen'].shape[0] ==
                        self.Enet['profiles_gen_p'].shape[0]
                    ), "Number of generators does not match the shape of generator profiles."

                if not self.Enet.storage.empty and set(necessary_time_series_sto_profiles).issubset(self.Enet.keys()):
                    assert (
                        self.Enet['profiles_load_p'].shape[1] ==
                        self.Enet['profiles_sto_soc'].shape[1]
                    ), "Mismatch in length of storage and load profiles."
                    assert (
                        self.Enet['storage'].shape[0] ==
                        self.Enet['profiles_sto_soc'].shape[0]
                    ), "Number of storage elements does not match the shape of SOC profiles."

                assert (
                    self.H <= self.t_all
                ), f"Horizon H={self.H} exceeds available profile length t_all={self.t_all}."

        else:
            # No time series: default to single-step
            self.t_all = 0
            self.H = 1
            self.dt_min = dt_min

        self.H_all = H_all

        # Identify heat pumps by name in load.profile
        self.index_hp = np.where(self.Enet.load.profile.str.startswith(('Soil', 'Air')))[0]
        self.max_p_mw_hp = {}
        for i in self.index_hp:
            self.max_p_mw_hp[i] = max(self.Enet["profiles_load_p"][i])

        # Number of blocks used for heat pump scheduling (e.g., daily blocks)
        self.e_hp_blocks = max(1, self.H_all // 24)

        # Store total required energy for each heat pump in each block
        self.e_hp = {}
        # Keep track of heat pump power usage history
        self.p_hp_history = {}
        for i in self.index_hp:
            self.p_hp_history[i] = []

        print(self.t_all)

        # If switch data is present in the network, add a small series resistance for consistency
        if 'switch' in self.Enet.keys():
            if self.Enet.switch.shape[0] != 0:
                print('Switches detected. Adding small resistance for consistency.')
                self.Enet.switch['z_ohm'] = 0.1

        # If necessary, run a power flow to populate the _ppc data structure
        if ('_ppc' not in self.Enet.keys()) or ('switch' in self.Enet.keys()):
            print('No previous power flow solution found. Running initial power flow.')
            pp.runpp(self.Enet, numba=False)

        self.solver_options = solver_options
        self.solver = solver

        # Default voltage bounds if they are not explicitly defined in the pandapower network
        self.v_min_default = v_min
        self.v_max_default = v_max

        # Store nominal voltage for each bus
        self.vn = {i: k for i, k in enumerate(self.Enet.bus['vn_kv'].values)}

        # Ramping constraints for battery storage (True/False)
        self.with_storage_ramp = with_storage_ramp

        # Construct the Pyomo model
        self.model = self.transform_pandapower_grid_to_model()

        # Build mappings for loads, generators, storage, etc.
        self._attach_nodemappings()

        # Create and add constraints to the Pyomo model
        self.make_constraints()

        # Initialize the model with either time series data or static data
        if self.withTimeseries:
            self.fix_values_timestep(self.timestep_start)
        else:
            self.fix_values_static()

        # Define the objective function in the model
        self.make_objective()

    def _get_buses(self):
        """
        Returns
        -------
        bus_idx : ndarray
            Array of bus indices as integers.
        """
        bus_idx = self.Enet._ppc['internal']['bus'][:, 0].astype(int)
        return bus_idx

    def _get_edges(self):
        """
        Returns
        -------
        ft : list of tuples
            Each tuple contains (from_bus, to_bus) as internal indices.
        ft_lines : list of tuples
            Edges corresponding to network lines.
        ft_trafo : list of tuples
            Edges corresponding to transformers.
        """
        ft = np.real(self.Enet._ppc['internal']['branch'][:, [0, 1]]).astype(int)
        ft = [(f, t) for f, t in ft]
        ft_lines = [(f, t) for f, t in self.Enet.line[['from_bus', 'to_bus']].values]
        ft_trafo = [(f, t) for f, t in self.Enet.trafo[['hv_bus', 'lv_bus']].values]
        return (ft, ft_lines, ft_trafo)

    def _get_pq_buses(self):
        """Returns the set of bus indices where loads (pq) are located."""
        pq_idx = self.Enet.load['bus'].values.astype(int)
        return pq_idx

    def _get_sgen_buses(self):
        """Returns the set of bus indices where sgen elements are located."""
        sgen_idx = self.Enet.sgen['bus'].values.astype(int)
        return sgen_idx

    def _get_storage_buses(self):
        """Returns the set of bus indices where storage elements are located."""
        storage_idx = self.Enet.storage['bus'].values.astype(int)
        return storage_idx

    def _get_slack_buses(self):
        """Returns the set of bus indices where external grids (slack buses) are located."""
        slack_idx = self.Enet.ext_grid['bus'].values.astype(int)
        return slack_idx

    def _get_gen_buses(self):
        """Returns the set of bus indices where generators are located."""
        gen_idx = self.Enet.gen['bus'].values.astype(int)
        return gen_idx

    def _net_to_lineparameters(self):
        """
        Extracts the line admittance at the "from" and "to" buses from the Yf and Yt matrices.

        Returns
        -------
        g_ff, b_ff, g_ft, b_ft, g_tt, b_tt, g_tf, b_tf : dict
            Conductance (g_xx) and susceptance (b_xx) dictionaries indexed by edge (from_bus, to_bus).
            Each key is a tuple (from_bus, to_bus), and each value is the respective real/imag component.
        """
        Yf = self.Enet._ppc['internal']['Yf']
        Yt = self.Enet._ppc['internal']['Yt']

        g_ff, b_ff = {}, {}
        g_ft, b_ft = {}, {}
        g_tt, b_tt = {}, {}
        g_tf, b_tf = {}, {}

        ft, ft_lines, ft_trafo = self._get_edges()
        i = 0
        for f, t in ft:
            g_ff[(f, t)] = np.real(Yf[i, f])
            b_ff[(f, t)] = np.imag(Yf[i, f])
            g_ft[(f, t)] = np.real(Yf[i, t])
            b_ft[(f, t)] = np.imag(Yf[i, t])

            g_tt[(f, t)] = np.real(Yt[i, t])
            b_tt[(f, t)] = np.imag(Yt[i, t])
            g_tf[(f, t)] = np.real(Yt[i, f])
            b_tf[(f, t)] = np.imag(Yt[i, f])
            i += 1

        return (g_ff, b_ff, g_ft, b_ft, g_tt, b_tt, g_tf, b_tf)

    def _get_inout(self, edges):
        """
        Constructs adjacency sets that record which buses flow into and out of each bus.

        Parameters
        ----------
        edges : iterable of tuples
            Each tuple is (from_bus, to_bus).

        Returns
        -------
        inflow_set, outflow_set : dict of sets
            inflow_set[node] = set of all buses that directly feed into `node`.
            outflow_set[node] = set of all buses that `node` feeds into directly.
        """
        inflow_set = defaultdict(set)
        outflow_set = defaultdict(set)

        for i, j in edges:
            inflow_set[j].add(i)
            outflow_set[i].add(j)

        return inflow_set, outflow_set

    def _attach_nodemappings(self):
        """
        Creates mappings from a bus index to the indices of connected
        loads, sgens, storage units, or generators in that bus.
        """
        mapping_load = {}
        mapping_sgen = {}
        mapping_storage = {}
        mapping_gen = {}

        for i in self.Enet.bus.index.values:
            nodes_load = list(self.Enet.load[self.Enet.load['bus'] == i].index.values)
            nodes_sgen = list(self.Enet.sgen[self.Enet.sgen['bus'] == i].index.values)
            nodes_storage = list(self.Enet.storage[self.Enet.storage['bus'] == i].index.values)
            nodes_gen = list(self.Enet.gen[self.Enet.gen['bus'] == i].index.values)

            mapping_load[i] = nodes_load
            mapping_sgen[i] = nodes_sgen
            mapping_storage[i] = nodes_storage
            mapping_gen[i] = nodes_gen

        self.model.mapping_load = mapping_load
        self.model.mapping_sgen = mapping_sgen
        self.model.mapping_gen = mapping_gen
        self.model.mapping_storage = mapping_storage

    def transform_pandapower_grid_to_model(self):
        """
        Creates a Pyomo ConcreteModel and defines the sets, parameters, and variables
        required for representing the network in an AC OPF formulation.

        Returns
        -------
        model : pyomo.environ.ConcreteModel
            A Pyomo ConcreteModel ready for constraint definitions.
        """
        model = pe.ConcreteModel()

        # Retrieve edges
        ft, ft_lines, ft_trafo = self._get_edges()

        # Pyomo Sets
        model.nodes = pe.Set(initialize=self._get_buses())
        model.nodes_no_hp = pe.Set(
            initialize=np.array([i for i in self._get_buses() if i not in self.index_hp])
        )
        model.edges = pe.Set(initialize=ft)
        model.lines = pe.Set(initialize=ft_lines)
        model.trafo = pe.Set(initialize=ft_trafo)
        model.heatpumps = pe.Set(initialize=self.index_hp)

        # Time indices
        model.T = pe.RangeSet(0, self.H - 1)
        model.T2minus1 = pe.RangeSet(-1, self.H - 2)
        model.T2 = pe.RangeSet(-1, self.H - 1)
        # For battery storage SoC indexing
        model.Te = pe.RangeSet(0, self.H)

        # Bus sets
        model.slack_nodes = pe.Set(initialize=self._get_slack_buses())
        model.pq_nodes = pe.Set(initialize=self.Enet.load.index.values)
        model.pq_nodes_no_hp = pe.Set(
            initialize=np.array([i for i in self.Enet.load.index.values if i not in self.index_hp])
        )
        model.pq_nodes_only_hp = pe.Set(initialize=self.index_hp)
        model.sgen_nodes = pe.Set(initialize=self.Enet.sgen.index.values)
        model.generator_nodes = pe.Set(initialize=self.Enet.gen.index.values)
        model.storage_nodes = pe.Set(initialize=self.Enet.storage.index.values)

        # Line parameters
        (g_ff, b_ff, g_ft, b_ft,
         g_tt, b_tt, g_tf, b_tf) = self._net_to_lineparameters()

        model.g_ff = pe.Param(g_ff.keys(), initialize=g_ff)
        model.b_ff = pe.Param(b_ff.keys(), initialize=b_ff)
        model.g_ft = pe.Param(g_ft.keys(), initialize=g_ft)
        model.b_ft = pe.Param(b_ft.keys(), initialize=b_ft)
        model.g_tt = pe.Param(g_tt.keys(), initialize=g_tt)
        model.b_tt = pe.Param(b_tt.keys(), initialize=b_tt)
        model.g_tf = pe.Param(g_tf.keys(), initialize=g_tf)
        model.b_tf = pe.Param(b_tf.keys(), initialize=b_tf)

        model.sn_mva = pe.Param(initialize=self.Enet.sn_mva)
        model.vn = pe.Param(model.nodes, initialize=self.vn)

        inflow_set, outflow_set = self._get_inout(model.edges)
        model.inflow_set = pe.Param(model.nodes, within=pe.Any, default=set(), initialize=inflow_set)
        model.outflow_set = pe.Param(model.nodes, within=pe.Any, default=set(), initialize=outflow_set)

        # Additional constraints for lines/trafo if specified
        if self.flow_constraint in ['power', 'both']:
            # For transformers: sn_mva is used as the rating
            self.trafo_limits = {
                (int(f), int(t)): s
                for f, t, s in self.Enet.trafo[['hv_bus', 'lv_bus', 'sn_mva']].values
            }
            for key in self.trafo_limits.keys():
                self.trafo_limits[key]*= self.trafo_limit_percent/100
            model.line_limit_s = pe.Param(model.trafo, initialize=self.trafo_limits)

        if self.flow_constraint in ['current', 'both']:
            self.line_limits = {
                (int(f), int(t)): i
                for f, t, i in self.Enet.line[['from_bus', 'to_bus', 'max_i_ka']].values
            }
            model.line_limit_i = pe.Param(model.lines, initialize=self.line_limits)

        # Define Variables
        # Node-level variables
        model.P = pe.Var(model.nodes, model.T, domain=pe.Reals)
        model.P_pos = pe.Var(model.nodes, model.T, domain=pe.NonNegativeReals)
        model.P_neg = pe.Var(model.nodes, model.T, domain=pe.NonNegativeReals)
        model.Q = pe.Var(model.nodes, model.T, domain=pe.Reals)
        model.V = pe.Var(model.nodes, model.T, domain=pe.Reals)
        model.Theta = pe.Var(model.nodes, model.T, domain=pe.Reals)

        # Load, sgen, storage, generator variables
        model.Pload = pe.Var(model.pq_nodes, pe.RangeSet(0, max(model.T.at(-1), 23)), domain=pe.Reals)
        model.Qload = pe.Var(model.pq_nodes, model.T, domain=pe.Reals)
        model.Psgen = pe.Var(model.sgen_nodes, model.T, domain=pe.Reals)
        model.Qsgen = pe.Var(model.sgen_nodes, model.T, domain=pe.Reals, bounds=(0.0, 0.0))
        model.Psto = pe.Var(model.storage_nodes, model.T2, domain=pe.Reals)
        # By default, Qsto is set to zero. If needed, remove the bounds.
        model.Qsto = pe.Var(model.storage_nodes, model.T, domain=pe.Reals, bounds=(0.0, 0.0))
        model.Pgen = pe.Var(model.generator_nodes, model.T, domain=pe.Reals)
        model.Qgen = pe.Var(model.generator_nodes, model.T, domain=pe.Reals)

        # Storage SoC
        model.E = pe.Var(model.storage_nodes, model.Te, domain=pe.Reals)
        model.Storage_Pmax = pe.Param(
            model.storage_nodes,
            initialize=self.Enet.storage['max_p_mw'].values
        )

        lbs = self.Enet.storage['min_e_mwh'].values
        ubs = self.Enet.storage['max_e_mwh'].values
        for t in model.Te:
            for i, lb, ub in zip(model.storage_nodes, lbs, ubs):
                model.E[i, t].setub(ub)
                model.E[i, t].setlb(lb)

        # Edge-level variables
        model.p_f = pe.Var(model.edges, model.T, domain=pe.Reals)
        model.q_f = pe.Var(model.edges, model.T, domain=pe.Reals)
        model.i_f_real = pe.Var(model.edges, model.T, domain=pe.Reals)
        model.i_f_imag = pe.Var(model.edges, model.T, domain=pe.Reals)

        model.p_t = pe.Var(model.edges, model.T, domain=pe.Reals)
        model.q_t = pe.Var(model.edges, model.T, domain=pe.Reals)
        model.i_t_real = pe.Var(model.edges, model.T, domain=pe.Reals)
        model.i_t_imag = pe.Var(model.edges, model.T, domain=pe.Reals)

        return model

    def nodal_power_injection_P(self, model, i, t):
        """
        Defines the net active power injection into bus `i` at time `t`.
        This is the algebraic sum of loads, generation, and storage on bus `i`.
        """
        if i in self.model.slack_nodes:
            return pe.Constraint.Skip
        else:
            lhs = model.P[i, t]
            rhs = (
                -sum(model.Pgen[k, t] for k in model.mapping_gen[i])
                - sum(model.Psgen[k, t] for k in model.mapping_sgen[i])
                + sum(model.Psto[k, t] for k in model.mapping_storage[i])
                + sum(model.Pload[k, t] for k in model.mapping_load[i])
            )
            return rhs == lhs

    def nodal_power_injection_Q(self, model, i, t):
        """
        Defines the net reactive power injection into bus `i` at time `t`.
        Similar to the active power injection but for Q.
        """
        if i in self.model.slack_nodes:
            return pe.Constraint.Skip
        else:
            lhs = model.Q[i, t]
            rhs = (
                -sum(model.Qgen[k, t] for k in model.mapping_gen[i])
                - sum(model.Qsgen[k, t] for k in model.mapping_sgen[i])
                + sum(model.Qsto[k, t] for k in model.mapping_storage[i])
                + sum(model.Qload[k, t] for k in model.mapping_load[i])
            )
            return rhs == lhs

    def nodal_power_balance_P(self, model, i, t):
        """
        Active power balance at bus `i` in time `t`.
        Sums flows entering and leaving the bus to ensure zero net flow.
        """
        return (
            model.P[i, t]
            + sum(model.p_t[k, i, t] for k in model.inflow_set[i])
            + sum(model.p_f[i, l, t] for l in model.outflow_set[i])
            == 0
        )

    def nodal_power_balance_Q(self, model, i, t):
        """
        Reactive power balance at bus `i` in time `t`.
        """
        return (
            model.Q[i, t]
            + sum(model.q_t[k, i, t] for k in model.inflow_set[i])
            + sum(model.q_f[i, l, t] for l in model.outflow_set[i])
            == 0
        )

    def edge_flow_p_from(self, model, i, j, t):
        """
        Active power flow from bus i->j. 
        P_f = G_ff * V_i^2 + V_i*V_j * [ G_ft*cos(theta_i - theta_j) + B_ft*sin(theta_i - theta_j) ].
        """
        return (
            model.p_f[i, j, t] ==
            model.g_ff[i, j] * model.V[i, t]**2
            + model.V[i, t] * model.V[j, t] * (
                model.g_ft[i, j]*pe.cos(model.Theta[i, t] - model.Theta[j, t]) +
                model.b_ft[i, j]*pe.sin(model.Theta[i, t] - model.Theta[j, t])
            )
        )

    def edge_flow_p_to(self, model, i, j, t):
        """
        Active power flow from bus j->i (the "to" part):
        P_t = G_tt * V_j^2 + ...
        """
        return (
            model.p_t[i, j, t] ==
            model.g_tt[i, j] * model.V[j, t]**2
            + model.V[i, t] * model.V[j, t] * (
                model.g_tf[i, j]*pe.cos(model.Theta[j, t] - model.Theta[i, t]) +
                model.b_tf[i, j]*pe.sin(model.Theta[j, t] - model.Theta[i, t])
            )
        )

    def edge_flow_q_from(self, model, i, j, t):
        """
        Reactive power flow from bus i->j.
        Q_f = -B_ff * V_i^2 + ...
        """
        return (
            model.q_f[i, j, t] ==
            -model.b_ff[i, j] * model.V[i, t]**2
            + model.V[i, t] * model.V[j, t] * (
                model.g_ft[i, j]*pe.sin(model.Theta[i, t] - model.Theta[j, t]) -
                model.b_ft[i, j]*pe.cos(model.Theta[i, t] - model.Theta[j, t])
            )
        )

    def edge_flow_q_to(self, model, i, j, t):
        """
        Reactive power flow from bus j->i.
        Q_t = -B_tt * V_j^2 + ...
        """
        return (
            model.q_t[i, j, t] ==
            -model.b_tt[i, j] * model.V[j, t]**2
            + model.V[i, t] * model.V[j, t] * (
                model.g_tf[i, j]*pe.sin(model.Theta[j, t] - model.Theta[i, t]) -
                model.b_tf[i, j]*pe.cos(model.Theta[j, t] - model.Theta[i, t])
            )
        )

    def line_limit_s_from(self, model, i, j, t):
        """
        Apparent power limit (s^2 = p^2 + q^2) for the from-side of a transformer.
        """
        return 0 >= model.p_f[i, j, t]**2 + model.q_f[i, j, t]**2 - model.line_limit_s[i, j]**2

    def line_limit_s_to(self, model, i, j, t):
        """
        Apparent power limit for the to-side of a transformer.
        """
        return 0 >= model.p_t[i, j, t]**2 + model.q_t[i, j, t]**2 - model.line_limit_s[i, j]**2

    def edge_flow_i_from_real(self, model, i, j, t):
        """
        Real component of current from bus i->j.
        """
        return (
            model.i_f_real[i, j, t] ==
            model.g_ff[i, j] * model.V[i, t]*pe.cos(model.Theta[i, t])
            - model.b_ff[i, j] * model.V[i, t]*pe.sin(model.Theta[i, t])
            + model.g_ft[i, j] * model.V[j, t]*pe.cos(model.Theta[j, t])
            - model.b_ft[i, j] * model.V[j, t]*pe.sin(model.Theta[j, t])
        )

    def edge_flow_i_from_imag(self, model, i, j, t):
        """
        Imag component of current from bus i->j.
        """
        return (
            model.i_f_imag[i, j, t] ==
            model.b_ff[i, j] * model.V[i, t]*pe.cos(model.Theta[i, t])
            + model.g_ff[i, j] * model.V[i, t]*pe.sin(model.Theta[i, t])
            + model.b_ft[i, j] * model.V[j, t]*pe.cos(model.Theta[j, t])
            + model.g_ft[i, j] * model.V[j, t]*pe.sin(model.Theta[j, t])
        )

    def edge_flow_i_to_real(self, model, i, j, t):
        """
        Real component of current from bus j->i.
        """
        return (
            model.i_t_real[i, j, t] ==
            model.g_tt[i, j] * model.V[j, t]*pe.cos(model.Theta[j, t])
            - model.b_tt[i, j] * model.V[j, t]*pe.sin(model.Theta[j, t])
            + model.g_tf[i, j] * model.V[i, t]*pe.cos(model.Theta[i, t])
            - model.b_tf[i, j] * model.V[i, t]*pe.sin(model.Theta[i, t])
        )

    def edge_flow_i_to_imag(self, model, i, j, t):
        """
        Imag component of current from bus j->i.
        """
        return (
            model.i_t_imag[i, j, t] ==
            model.b_tt[i, j] * model.V[j, t]*pe.cos(model.Theta[j, t])
            + model.g_tt[i, j] * model.V[j, t]*pe.sin(model.Theta[j, t])
            + model.b_tf[i, j] * model.V[i, t]*pe.cos(model.Theta[i, t])
            + model.g_tf[i, j] * model.V[i, t]*pe.sin(model.Theta[i, t])
        )

    def line_limit_i_from(self, model, i, j, t):
        """
        Current (in kA) limit for the from side of a line. 
        Conversion from p.u. to kA is done within the constraint.
        """
        return 0 >= (
            (model.i_f_real[i, j, t]**2 + model.i_f_imag[i, j, t]**2)
            * (model.sn_mva/(pe.sqrt(3)*model.vn[j]))**2
            - model.line_limit_i[i, j]**2
        )

    def line_limit_i_to(self, model, i, j, t):
        """
        Current (in kA) limit for the to side of a line.
        """
        return 0 >= (
            (model.i_t_real[i, j, t]**2 + model.i_t_imag[i, j, t]**2)
            * (model.sn_mva/(pe.sqrt(3)*model.vn[j]))**2
            - model.line_limit_i[i, j]**2
        )

    def storage_equation(self, model, i, t):
        """
        Battery storage energy balance equation:
        E[i, t+1] = E[i, t] + Psto[i, t]*(dt_min/60).
        """
        return (
            model.E[i, t+1] 
            - model.E[i, t]
            - model.Psto[i, t] * (self.dt_min / 60.) + self.nue * model.E[i, t] * 15 / 60.
            == 0.
        )

    def storage_ramp1(self, model, i, t):
        """
        Ramping constraint (1) for battery storage:
        P[i,t] - P[i,t+1] <= 20% of maximum storage power.
        """
        return model.Psto[i, t] - model.Psto[i, t+1] <= model.Storage_Pmax[i]*0.2

    def storage_ramp2(self, model, i, t):
        """
        Ramping constraint (2) for battery storage:
        P[i,t+1] - P[i,t] <= 20% of maximum storage power.
        """
        return model.Psto[i, t+1] - model.Psto[i, t] <= model.Storage_Pmax[i]*0.2

    def make_constraints(self):
        """
        Constructs and attaches the constraints to the Pyomo model,
        including power balance, voltage bounds, flow limits, and storage relations.
        """

        # Default voltage limits, if not provided in the pandapower bus data
        if "min_vm_pu" not in self.Enet.bus.columns:
            self.model.V.setlb(self.v_min_default)
        else:
            for i in range(self.Enet.bus.shape[0]):
                vm_pu_lb = (
                    self.v_min_default
                    if np.isnan(self.Enet.bus["min_vm_pu"][i])
                    else self.Enet.bus["min_vm_pu"][i]
                )
                self.model.V[i, :].setlb(vm_pu_lb)

        if "max_vm_pu" not in self.Enet.bus.columns:
            self.model.V.setub(self.v_max_default)
        else:
            for i in range(self.Enet.bus.shape[0]):
                vm_pu_ub = (
                    self.v_max_default
                    if np.isnan(self.Enet.bus["max_vm_pu"][i])
                    else self.Enet.bus["max_vm_pu"][i]
                )
                self.model.V[i, :].setub(vm_pu_ub)

        # Power balance constraints
        self.model.nodal_power_balance_P = pe.Constraint(
            self.model.nodes,
            self.model.T,
            rule=self.nodal_power_balance_P
        )
        self.model.nodal_power_balance_Q = pe.Constraint(
            self.model.nodes,
            self.model.T,
            rule=self.nodal_power_balance_Q
        )
        self.model.nodal_power_inj_P = pe.Constraint(
            self.model.nodes,
            self.model.T,
            rule=self.nodal_power_injection_P
        )
        self.model.nodal_power_inj_Q = pe.Constraint(
            self.model.nodes,
            self.model.T,
            rule=self.nodal_power_injection_Q
        )

        # Branch flow constraints
        self.model.edge_flow_p_from = pe.Constraint(
            self.model.edges, 
            self.model.T,
            rule=self.edge_flow_p_from
        )
        self.model.edge_flow_p_to = pe.Constraint(
            self.model.edges, 
            self.model.T,
            rule=self.edge_flow_p_to
        )
        self.model.edge_flow_q_from = pe.Constraint(
            self.model.edges, 
            self.model.T,
            rule=self.edge_flow_q_from
        )
        self.model.edge_flow_q_to = pe.Constraint(
            self.model.edges, 
            self.model.T,
            rule=self.edge_flow_q_to
        )
        self.model.edge_flow_i_from_real = pe.Constraint(
            self.model.edges,
            self.model.T,
            rule=self.edge_flow_i_from_real
        )
        self.model.edge_flow_i_from_imag = pe.Constraint(
            self.model.edges,
            self.model.T,
            rule=self.edge_flow_i_from_imag
        )
        self.model.edge_flow_i_to_real = pe.Constraint(
            self.model.edges,
            self.model.T,
            rule=self.edge_flow_i_to_real
        )
        self.model.edge_flow_i_to_imag = pe.Constraint(
            self.model.edges,
            self.model.T,
            rule=self.edge_flow_i_to_imag
        )

        # Flow or current constraints
        if self.flow_constraint in ['power', 'both']:
            self.model.line_limit_s_from = pe.Constraint(
                self.model.trafo,
                self.model.T,
                rule=self.line_limit_s_from
            )
            self.model.line_limit_s_to = pe.Constraint(
                self.model.trafo,
                self.model.T,
                rule=self.line_limit_s_to
            )

        if self.flow_constraint in ['current', 'both']:
            self.model.line_limit_i_from_con = pe.Constraint(
                self.model.lines,
                self.model.T,
                rule=self.line_limit_i_from
            )
            self.model.line_limit_i_to_con = pe.Constraint(
                self.model.lines,
                self.model.T,
                rule=self.line_limit_i_to
            )

        # Storage energy balance constraints
        if not self.Enet.storage.empty:
            self.model.storage_eq = pe.Constraint(
                self.model.storage_nodes,
                self.model.T,
                rule=self.storage_equation
            )

        # Ramping constraints for storage if specified
        if self.with_storage_ramp == "True":
            self.model.storage_ramp1 = pe.Constraint(
                self.model.storage_nodes,
                self.model.T2minus1,
                rule=self.storage_ramp1
            )
            self.model.storage_ramp2 = pe.Constraint(
                self.model.storage_nodes,
                self.model.T2minus1,
                rule=self.storage_ramp2
            )

        # Additional objectives
        if self.objective == 'variable_cost' or self.objective == "pywraplp_multiobjective_power_price":
            self.model.preis = pe.Var(self.model.T)
        elif self.objective == 'binary_cost':
            self.model.p_grid_pos = pe.Var(self.model.T, domain=pe.PositiveReals)
            self.model.p_grid_neg = pe.Var(self.model.T, domain=pe.PositiveReals)
            self.model.p_grid_pos_con = pe.Constraint(
                self.model.T,
                rule=lambda m, t: m.p_grid_pos[t] >= 0.0
            )
            self.model.p_grid_neg_con = pe.Constraint(
                self.model.T,
                rule=lambda m, t: m.p_grid_neg[t] >= 0.0
            )
            self.model.p_grid_abs_con = pe.Constraint(
                self.model.T,
                rule=lambda m, t: m.p_grid_pos[t] - m.p_grid_neg[t] == sum(m.P[i, t] for i in self.model.slack_nodes)
            )

        # Positive/negative decomposition for the slack bus injection
        self.model.P_balance = pe.Constraint(
            self.model.nodes,
            self.model.T,
            rule=lambda m, i, t: m.P[i, t] == m.P_pos[i, t] - m.P_neg[i, t]
        )
        self.model.P_pos_nonneg = pe.Constraint(
            self.model.nodes,
            self.model.T,
            rule=lambda m, i, t: m.P_pos[i, t] >= 0
        )
        self.model.P_neg_nonneg = pe.Constraint(
            self.model.nodes,
            self.model.T,
            rule=lambda m, i, t: m.P_neg[i, t] >= 0
        )

        # Pre-compute total heat demand in blocks (e_hp).
        for i in self.index_hp:
            self.e_hp[i] = np.zeros(shape=self.e_hp_blocks, dtype=float)
            for block in range(self.e_hp_blocks):
                for t in range(block*24, (block+1)*24):
                    self.e_hp[i][block] += (
                        self.Enet["profiles_load_p"][i][t + self.timestep_start] * 15 / 60
                    )

    def make_objective(self):
        """
        Defines the objective function of the Pyomo model based on the selected 'self.objective'.
        """
        if self.objective == 'quadratic_exchange':
            # Minimizes the square of P and Q at the slack bus
            self.model.obj_func = sum(
                self.model.P[i, t]**2 + self.model.Q[i, t]**2
                for i in self.model.slack_nodes
                for t in self.model.T
            )
        elif self.objective == 'pywraplp_multiobjective_power_price':
            # Weighted sum of cost of power exchange and total power magnitude
            objective_1 = sum(
                (sum(-self.model.P[i, t]*self.model.preis[t] * 1000 * 15 / 60
                for i in self.model.slack_nodes) - self.temp_money_spend_mean) / self.temp_money_spend_std
                for t in self.model.T
            )
            objective_2 = sum(
                (sum([self.model.P[i, t]
                for i in self.model.slack_nodes]) ** 2 - self.temp_net_sum_p_mw_mean) / self.temp_net_sum_p_mw_std
                for t in self.model.T
            )

            #print(f"price weight: {self.omega_1}, power weight: {self.omega_2}")
            self.model.obj_func = self.omega_1 * objective_1 + self.omega_2 * objective_2

        elif self.objective == 'transmission_losses':
            # Minimizes sum of P_f + P_t over all edges
            self.model.obj_func = sum(
                self.model.p_f[i, j, t] + self.model.p_t[i, j, t]
                for i, j in self.model.edges
                for t in self.model.T
            )
        elif self.objective == 'variable_cost':
            # Minimizes cost of power usage (negative sign because we define minimization)
            self.model.obj_func = -sum(
                self.model.P[i, t]*self.model.preis[t]
                for i in self.model.slack_nodes
                for t in self.model.T
            )
        elif self.objective == 'binary_cost':
            # Illustrative binary cost: different prices for import/export
            self.model.obj_func = sum(
                self.model.p_grid_pos[t]*(-0.08) + 0.3*self.model.p_grid_neg[t]
                for t in self.model.T
            )

        # If an old objective component exists, remove it
        if hasattr(self.model, 'obj'):
            self.model.del_component(self.model.obj)
        self.model.obj = pe.Objective(expr=self.model.obj_func, sense=pe.minimize)

    def fix_values_timestep(self, t):
        """
        Fixes (or bounds) model variables for the time steps in the Pyomo model,
        aligning them with the real data from the time step `t`.
        Called in each iteration of an MPC or a rolling horizon approach.

        Parameters
        ----------
        t : int
            Current global time index (e.g., from 0 to t_all - H).
        """
        assert self.t_all >= t + self.H, "Requested time horizon extends beyond available time series data."

        t_local = t - self.timestep_start

        # ----------------------------------------------------------------------
        # Heatpump block constraints:
        # Enforce the total energy consumption in each block for heat pumps,
        # allowing a specified tolerance for partial distribution.
        # ----------------------------------------------------------------------
        def constraint_heatpump_lower_energy(model, i):
            block = t_local // 24
            e_hp_history_in_block = 0
            for tt in range(block*24, t_local):
                e_hp_history_in_block += self.p_hp_history[i][tt] * 15 / 60
            return (
                self.e_hp[i][block] * (1 - self.e_hp_tolerance)
                <= e_hp_history_in_block
                   + sum(model.Pload[i, tt] * 15 / 60 for tt in range(0, (24 - (t_local % 24))))
            )

        def constraint_heatpump_upper_energy(model, i):
            block = t_local // 24
            e_hp_history_in_block = 0
            for tt in range(block*24, t_local):
                e_hp_history_in_block += self.p_hp_history[i][tt] * 15 / 60
            return (
                e_hp_history_in_block
                + sum(model.Pload[i, tt] * 15 / 60 for tt in range(0, (24 - (t_local % 24))))
                <= self.e_hp[i][block] * (1 + self.e_hp_tolerance)
            )
        def constraint_heatpump_zero(model, i, tt):
            return model.Pload[i, tt] == 0

        # Remove old constraints if they exist, then add new ones
        if self.model.find_component("constraints_heatpump_lower_energy"):
            self.model.del_component(self.model.constraints_heatpump_lower_energy)
        if self.model.find_component("constraints_heatpump_upper_energy"):
            self.model.del_component(self.model.constraints_heatpump_upper_energy)
        if self.model.find_component("constraints_heatpump_zero"):
            self.model.del_component(self.model.constraints_heatpump_zero)

        # if heatpump energy is used, set it to 0 in the current block, because otherwise infeasible
        # currently this acts only correctly if one heatpump is there, otherwise it would block all heatpumps if one is above
        heatpump_zero = False
        for i in self.index_hp:
            block = t_local // 24
            e_hp_history_in_block = 0
            for tt in range(block*24, t_local):
                e_hp_history_in_block += self.p_hp_history[i][tt] * 15 / 60
            if e_hp_history_in_block >= self.e_hp[i][block] and e_hp_history_in_block > 0:
                heatpump_zero = True
        if heatpump_zero:
            self.model.constraints_heatpump_zero = pe.Constraint(
                self.model.heatpumps,
                self.model.T,
                rule=constraint_heatpump_zero
            )
        else:
            self.model.constraints_heatpump_lower_energy = pe.Constraint(
                self.model.heatpumps,
                rule=constraint_heatpump_lower_energy
            )
            self.model.constraints_heatpump_upper_energy = pe.Constraint(
                self.model.heatpumps,
                rule=constraint_heatpump_upper_energy
            )

        # ----------------------------------------------------------------------
        # Limit the heatpump power at each time step between 0 and P_max.
        # ----------------------------------------------------------------------
        def constraint_heatpump_lower_power(model, i, tt):
            return model.Pload[i, tt] >= 0.

        def constraint_heatpump_upper_power(model, i, tt):
            return model.Pload[i, tt] <= self.max_p_mw_hp[i]

        if self.model.find_component("constraints_heatpump_lower_power"):
            self.model.del_component(self.model.constraints_heatpump_lower_power)
        if self.model.find_component("constraints_heatpump_upper_power"):
            self.model.del_component(self.model.constraints_heatpump_upper_power)

        self.model.constraints_heatpump_lower_power = pe.Constraint(
            self.model.heatpumps,
            self.model.T,
            rule=constraint_heatpump_lower_power
        )
        self.model.constraints_heatpump_upper_power = pe.Constraint(
            self.model.heatpumps,
            self.model.T,
            rule=constraint_heatpump_upper_power
        )

        # For Q load of heat pumps, we fix Q = 0; if needed, this can be relaxed
        def constraint_set_heatpump_qload(model, i, tt):
            return model.Qload[i, tt] == 0

        # If you need P at the bus to match the heat pump's Pload, uncomment
        def constraint_set_heatpump_p(model, i, tt):
            return model.P[i, tt] == model.Pload[i, tt]

        if self.model.find_component("qload_heatpump_constraint"):
            self.model.del_component(self.model.qload_heatpump_constraint)
        self.model.qload_heatpump_constraint = pe.Constraint(
            self.model.heatpumps,
            self.model.T,
            rule=constraint_set_heatpump_qload
        )

        if self.model.find_component("p_heatpump_constraint"):
            self.model.del_component(self.model.p_heatpump_constraint)
        # self.model.p_heatpump_constraint = pe.Constraint(
        #     self.model.heatpumps,
        #     self.model.T,
        #     rule=constraint_set_heatpump_p
        # )

        # ----------------------------------------------------------------------
        # Fix or bound other variables for each time step.
        # ----------------------------------------------------------------------
        for tt in self.model.T:
            # Slack bus: fix voltage magnitude and angle
            for i in self.model.slack_nodes:
                self.model.V[i, tt].fix(1.0)
                self.model.Theta[i, tt].fix(0.0)
                self.model.P[i, tt].fixed = False
                self.model.Q[i, tt].fixed = False

            # Storage: fix the SoC for the first time step, set bounds for P
            if not self.Enet.storage.empty:
                for i, soc, emx in zip(
                    self.model.storage_nodes,
                    self.Enet.storage['soc_percent'].values/100,
                    self.Enet.storage['max_e_mwh'].values
                ):
                    self.model.E[i, 0].fix(soc * emx)

                for i in self.Enet.storage.index.values:
                    self.model.Psto[i, -1].fix(self.Enet.storage["p_mw"][i], skip_validation=True)
                    if "min_p_mw" in self.Enet.storage.columns:
                        if not np.isnan(self.Enet.storage["min_p_mw"][i]):
                            self.model.Psto[i, :].setlb(self.Enet.storage["min_p_mw"][i])
                    if "max_p_mw" in self.Enet.storage.columns:
                        if not np.isnan(self.Enet.storage["max_p_mw"][i]):
                            self.model.Psto[i, :].setub(self.Enet.storage["max_p_mw"][i])

            # Fix loads at each non-HP bus using time-series data
            if not self.Enet.load.empty:
                if 'profiles_load_p' in self.Enet.keys():
                    for i, p in zip(self.model.pq_nodes_no_hp, self.Enet['profiles_load_p'][:, t+tt]):
                        self.model.Pload[i, tt].fix(p)
                if 'profiles_load_q' in self.Enet.keys():
                    for i, q in zip(self.model.pq_nodes_no_hp, self.Enet['profiles_load_q'][:, t+tt]):
                        self.model.Qload[i, tt].fix(q)

            # Fix generator outputs if time-series data is present
            if not self.Enet.gen.empty:
                if 'profiles_gen_p' in self.Enet.keys():
                    for i, p in zip(self.model.generator_nodes, self.Enet['profiles_gen_p'][:, t+tt]):
                        self.model.Pgen[i, tt].fix(p)
                if 'profiles_gen_q' in self.Enet.keys():
                    for i, q in zip(self.model.generator_nodes, self.Enet['profiles_gen_p'][:, t+tt]):
                        self.model.Qgen[i, tt].fix(q)

                # Bounds for generators
                for i, v in enumerate(self.Enet.gen.bus.values):
                    if "min_p_mw" in self.Enet.gen.columns:
                        if not np.isnan(self.Enet.gen["min_p_mw"][i]):
                            self.model.Pgen[i, :].setlb(self.Enet.gen["min_p_mw"][i])
                    if "max_p_mw" in self.Enet.gen.columns:
                        if not np.isnan(self.Enet.gen["max_p_mw"][i]):
                            self.model.Pgen[i, :].setub(self.Enet.gen["max_p_mw"][i])

            # Fix sgen outputs (PV) if time-series data is present
            if not self.Enet.sgen.empty:
                if 'profiles_pv_p' in self.Enet.keys():
                    for i, p in zip(self.model.sgen_nodes, self.Enet['profiles_pv_p'][:, t+tt]):
                        self.model.Psgen[i, tt].fix(p)
                if 'profiles_pv_q' in self.Enet.keys():
                    for i, q in zip(self.model.sgen_nodes, self.Enet['profiles_pv_q'][:, t+tt]):
                        self.model.Qsgen[i, tt].fix(q)
                else:
                    # If no Q data for sgen, fix to 0
                    for i in self.model.sgen_nodes:
                        self.model.Qsgen[i, tt].fix(0)

                # Bounds for sgen
                for i, v in enumerate(self.Enet.sgen.bus.values):
                    if "min_p_mw" in self.Enet.sgen.columns:
                        if not np.isnan(self.Enet.sgen["min_p_mw"][i]):
                            self.model.Psgen[i, :].setlb(self.Enet.sgen["min_p_mw"][i])
                    if "max_p_mw" in self.Enet.sgen.columns:
                        if not np.isnan(self.Enet.sgen["max_p_mw"][i]):
                            self.model.Psgen[i, :].setub(self.Enet.sgen["max_p_mw"][i])

            # If objective requires variable cost, fix price at each time step
            if self.objective == 'variable_cost' or self.objective == "pywraplp_multiobjective_power_price":
                self.model.preis[tt].fix(self.Enet["prediction_energy_prices_noisy"][t + tt])

        if self.objective in ['variable_cost', 'pywraplp_multiobjective_power_price']:
            self.make_objective()

    def fix_values_static(self):
        """
        Fixes variable values for a single-step scenario (i.e., no time series).
        This is useful if you want to solve a snapshot OPF without dynamic or
        temporal elements.
        """
        # Slack node: fix voltage and angle
        for i in self.model.slack_nodes:
            self.model.V[i, :].fix(1.0)
            self.model.Theta[i, :].fix(0.0)

        # Storage initialization
        if not self.Enet.storage.empty:
            for i, soc, emx in zip(
                self.model.storage_nodes,
                self.Enet.storage['soc_percent'].values/100,
                self.Enet.storage['max_e_mwh'].values
            ):
                self.model.E[i, 0].fix(soc * emx)
            for i in self.Enet.storage.index.values:
                if "min_p_mw" in self.Enet.storage.columns:
                    if not np.isnan(self.Enet.storage["min_p_mw"][i]):
                        self.model.Psto[i, :].setlb(self.Enet.storage["min_p_mw"][i])
                if "max_p_mw" in self.Enet.storage.columns:
                    if not np.isnan(self.Enet.storage["max_p_mw"][i]):
                        self.model.Psto[i, :].setub(self.Enet.storage["max_p_mw"][i])

        # Fix loads
        if not self.Enet.load.empty:
            for i in self.model.pq_nodes:
                self.model.Pload[i, :].fix(
                    self.Enet.load['p_mw'].loc[self.Enet.load.index == i].values[0]
                )
                self.model.Qload[i, :].fix(
                    self.Enet.load['q_mvar'].loc[self.Enet.load.index == i].values[0]
                )

        # Fix generators
        if not self.Enet.gen.empty:
            for i in self.model.generator_nodes:
                if 'p_mw' in self.Enet.gen.columns:
                    self.model.Pgen[i, :].fix(
                        self.Enet.gen['p_mw'].loc[self.Enet.gen.index == i].values[0]
                    )
                if 'q_mvar' in self.Enet.gen.columns:
                    self.model.Qgen[i, :].fix(
                        self.Enet.gen['q_mvar'].loc[self.Enet.gen.index == i].values[0]
                    )
            for i in self.Enet.gen.index.values:
                if "min_p_mw" in self.Enet.gen.columns:
                    if not np.isnan(self.Enet.gen["min_p_mw"][i]):
                        self.model.Pgen[i, :].setlb(self.Enet.gen["min_p_mw"][i])
                if "max_p_mw" in self.Enet.gen.columns:
                    if not np.isnan(self.Enet.gen["max_p_mw"][i]):
                        self.model.Pgen[i, :].setub(self.Enet.gen["max_p_mw"][i])

        # Fix sgens
        if not self.Enet.sgen.empty:
            for i in self.model.sgen_nodes:
                self.model.Psgen[i, :].fix(
                    self.Enet.sgen['p_mw'].loc[self.Enet.sgen.index == i].values[0]
                )
                self.model.Qsgen[i, :].fix(
                    self.Enet.sgen['q_mvar'].loc[self.Enet.sgen.index == i].values[0]
                )
            for i in self.Enet.sgen.index.values:
                if "min_p_mw" in self.Enet.sgen.columns:
                    if not np.isnan(self.Enet.sgen["min_p_mw"][i]):
                        self.model.Psgen[i, :].setlb(self.Enet.sgen["min_p_mw"][i])
                if "max_p_mw" in self.Enet.sgen.columns:
                    if not np.isnan(self.Enet.sgen["max_p_mw"][i]):
                        self.model.Psgen[i, :].setub(self.Enet.sgen["max_p_mw"][i])

    def run_opt(self):
        """
        Solves the Pyomo model with the user-specified solver.

        Returns
        -------
        results_obj : pyomo.opt.SolverResults
            Solver results object, containing solver status and other info.
        """
        solver = po.SolverFactory(self.solver)
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options['tol'] = 1e-9
        solver.options['max_iter'] = 30000
        results_obj = solver.solve(self.model, tee=self.solver_options['tee'])
        solver_condition = results_obj.solver.termination_condition.value

        if self.solver_options['verbose'] and solver_condition != 'optimal':
            log_infeasible_constraints(self.model)
            print(f"Solver termination condition: {solver_condition}")


        return results_obj

    def extract_results(self):
        """
        Extracts and returns the solution values from the model.

        Returns
        -------
        res_bus : dict of ndarrays
            Keys include 'V_pu', 'Theta_rad', 'P_mw', 'Q_mvar', 'P_sto', 'P_load', 'Q_sto', 'E_sto'.
            Each item is shaped (#elements, #time_steps).
        res_line : dict of ndarrays
            Flow variables shaped (number_of_edges, #time_steps).
            Keys include 'pf_mw', 'qf_mvar', 'pt_mw', 'qt_mvar', etc.
        res_obj_fcn : float
            Final objective value.
        """
        V, Theta, P, Pload, Q = [], [], [], [], []
        Psto, Qsto, Esto = [], [], []
        pf, pt, qf, qt = [], [], [], []
        if_real, if_imag, it_real, it_imag = [], [], [], []
        if_ka, it_ka = [], []

        # Retrieve results over the time horizon
        for t in self.model.T:
            for i in self.model.nodes:
                V.append(self.model.V[i, t].value)
                Theta.append(self.model.Theta[i, t].value)
                P.append(self.model.P[i, t].value)
                Q.append(self.model.Q[i, t].value)

            for i in self.model.storage_nodes:
                Psto.append(self.model.Psto[i, t].value)
                Qsto.append(self.model.Qsto[i, t].value)

            for i, j in self.model.edges:
                pf.append(self.model.p_f[i, j, t].value)
                pt.append(self.model.p_t[i, j, t].value)
                qf.append(self.model.q_f[i, j, t].value)
                qt.append(self.model.q_t[i, j, t].value)

                if_real.append(self.model.i_f_real[i, j, t].value)
                if_imag.append(self.model.i_f_imag[i, j, t].value)
                it_real.append(self.model.i_t_real[i, j, t].value)
                it_imag.append(self.model.i_t_imag[i, j, t].value)

                if_ka.append(
                    np.sqrt(
                        self.model.i_f_real[i, j, t].value**2 +
                        self.model.i_f_imag[i, j, t].value**2
                    )
                    * self.Enet.sn_mva / (np.sqrt(3)*self.vn[i])
                )
                it_ka.append(
                    np.sqrt(
                        self.model.i_t_real[i, j, t].value**2 +
                        self.model.i_t_imag[i, j, t].value**2
                    )
                    * self.Enet.sn_mva / (np.sqrt(3)*self.vn[j])
                )

            for i in self.Enet.load.index:
                Pload.append(self.model.Pload[i, t].value)

        # Storage SoC is one index longer in time
        for tt in self.model.Te:
            for i in self.model.storage_nodes:
                Esto.append(self.model.E[i, tt].value)

        # Format results into dictionaries
        res_bus = {}
        res_bus['V_pu'] = np.array(V).reshape(self.H, self.Enet.bus.shape[0]).T
        res_bus['Theta_rad'] = np.array(Theta).reshape(self.H, self.Enet.bus.shape[0]).T
        res_bus['P_mw'] = np.array(P).reshape(self.H, self.Enet.bus.shape[0]).T
        res_bus['Q_mvar'] = np.array(Q).reshape(self.H, self.Enet.bus.shape[0]).T
        res_bus['P_sto'] = np.array(Psto).reshape(self.H, self.Enet.storage.shape[0]).T
        res_bus['P_load'] = np.array(Pload).reshape(self.H, self.Enet.load.shape[0]).T
        res_bus['Q_sto'] = np.array(Qsto).reshape(self.H, self.Enet.storage.shape[0]).T
        res_bus['E_sto'] = np.array(Esto).reshape(self.H+1, self.Enet.storage.shape[0]).T

        number_edges = (
            self.Enet.line.shape[0]
            + self.Enet.switch[self.Enet.switch['et'] == 'b'].shape[0]
            + self.Enet.trafo.shape[0]
        )
        res_line = {}
        res_line['pf_mw'] = np.array(pf).reshape(self.H, number_edges).T
        res_line['qf_mvar'] = np.array(qf).reshape(self.H, number_edges).T
        res_line['pt_mw'] = np.array(pt).reshape(self.H, number_edges).T
        res_line['qt_mvar'] = np.array(qt).reshape(self.H, number_edges).T
        res_line['i_f_real_pu'] = np.array(if_real).reshape(self.H, number_edges).T
        res_line['i_f_imag_pu'] = np.array(if_imag).reshape(self.H, number_edges).T
        res_line['i_t_real_pu'] = np.array(it_real).reshape(self.H, number_edges).T
        res_line['i_t_imag_pu'] = np.array(it_imag).reshape(self.H, number_edges).T
        res_line['if_kA'] = np.array(if_ka).reshape(self.H, number_edges).T
        res_line['it_kA'] = np.array(it_ka).reshape(self.H, number_edges).T

        # Update heat pump usage history for the first time step
        for i in self.index_hp:
            self.p_hp_history[i].append(self.model.Pload[i, 0].value)

        res_obj_fcn = pe.value(self.model.obj)

        return res_bus, res_line, res_obj_fcn

    def update_storages(self):
        """
        Updates the 'soc_percent' in self.Enet.storage based on the solution
        from the most recent time step of the Pyomo model. 
        This can be called after each optimization step in a rolling horizon approach.
        """
        # SoC is updated as E[i,1]/Emax. The factor 100 is for converting to percent.
        self.Enet.storage['soc_percent'] = (
            pe.value(self.model.E[:, 1])
            / self.Enet.storage['max_e_mwh'].values
            * 100.
        )
        self.Enet.storage['p_mw'] = pe.value(self.model.Psto[:, 0])

    def update_TimeSeries(self, profiles_load_p, profiles_load_q, profiles_pv_p, profiles_pv_q):
        """
        Updates the time series in self.Enet with new data.
        """
        self.Enet['profiles_load_p'] = profiles_load_p
        self.Enet['profiles_load_q'] = profiles_load_q
        self.Enet['profiles_pv_p'] = profiles_pv_p
        self.Enet['profiles_pv_q'] = profiles_pv_q
        print("Time series updated.")

    def clear_model(self):
        """
        Clears the primary model variables for a fresh start or for re-initializing
        with new data. This sets their values to None.
        """
        self.model.P[:, :] = None
        self.model.Q[:, :] = None
        self.model.V[:, :] = None
        self.model.Theta[:, :] = None
