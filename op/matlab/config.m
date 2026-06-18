% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\config.m
function params = config()
    % Basic simulation parameters
    params.k_s = 1;         % Switching cost coefficient
    params.p_nom = 200;   % Reference power
    params.num_runs = 1;    % Number of simulation runs
    params.ENABLE_PLOTTING = true;
    
    params.Ts = 300;    % Sample rate [s]

    params.n_states = 8;   % Number of power states
    params.n_vals = [1:10];
    params.n0 = 5;
    params.sigma = 0.5;    % Standard deviation for random walk
        

end