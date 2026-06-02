% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\main.m

clc
clear

%%
params = config();

%%
% MC_model = RW_markow_model(params);
% demand_obj = Demand(params.T, params.P_d0, MC_model.levels, MC_model.P);
% demand_trend = demand_obj.P_d;

%%
[MC_model] = CSOV_MC_model(params);

fn = 'data/SOV_08-Feb-2023.mat';
Ds = SOV_demand(params, fn);

%%
Pd = Ds.Pd.';

%%
% ZOH vectors (separate from downsampling)
tL = Ds.left_edges; yb = Ds.Pd; T = Ds.Tsec;
tzoh = reshape([tL.'; (tL+T).'], [], 1);
yzoh = reshape([yb.'; yb.'], [], 1);

% Plot (ZOH)
figure(11); 
plot(tzoh/3600, yzoh, 'LineWidth', 1.5);
xlabel('Time [h]');
% ylabel('Power [kW]');
title(sprintf('Downsampled mean, ZOH (T = %d s)', Ds.Tsec));
grid on;
saveas(gcf, 'figures/sov_demand_zoh.fig');
saveas(gcf, 'figures/sov_demand_zoh.png');

%%
ctrls = create_controllers(params, MC_model);

%%
results = run_simulation(params, Pd, ctrls);

%%
plot_results(params, results, ctrls, Pd);


%% ==================================================================
%% FUNCTIONS
%% ==================================================================
function controllers = create_controllers(params, model)
    % Create controller instances based on the Markov chain model
    % Controllers are created from the model, not the demand
    
    controllers = {
        ConstantControl(), ...
        StochasticControl(params.k_s, params.p_star, model.levels, model.P, params.n_vals), ...
        ThresholdControl(params.k_s, params.p_star)
    };
end

function results = run_simulation(params, P_d, controllers)
    % Run simulation using demand trend and controllers
    % Returns structured results for analysis
    
    num_controllers = length(controllers);
    results = cell(num_controllers, 1);
    
    for i = 1:num_controllers
        % Create simulator instance
        simulator = Simulator(params.p_star, P_d, params.n0, params.k_s);
        
        % Run simulation
        total_cost = simulator.run(controllers{i});
        
        % Store results
        results{i} = struct();
        results{i}.total_cost = total_cost;
        results{i}.operating_cost = sum(simulator.C_o);
        results{i}.switching_cost = sum(simulator.C_s);
        results{i}.demand = simulator.P_d;
        results{i}.control = simulator.n;
        results{i}.operating_costs = simulator.C_o;
        results{i}.switching_costs = simulator.C_s;
        results{i}.controller_name = class(controllers{i});
    end
end

function plot_results(params, results, controllers, demand_trend)
    % Plot simulation results
    % Unified plotting regardless of model type
    
    % Create simulator data structure for compatibility with existing plotting functions
    all_run_data = cell(size(results));
    for i = 1:length(results)
        simulator_data = struct();
        simulator_data.C_o = results{i}.operating_costs;
        simulator_data.C_s = results{i}.switching_costs;
        simulator_data.n = results{i}.control*params.p_star;
        simulator_data.P_d = results{i}.demand;
        all_run_data{i} = simulator_data;
    end
    
    % Use existing plotting functions
    plot_costs_and_control(all_run_data, controllers, demand_trend);
    plot_cost_comparison(all_run_data, controllers);
    
end
