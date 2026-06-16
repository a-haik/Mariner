function model = RW_markow_model(params)
    % Create Markov chain model based on parameters
    
    model = struct();
    
    % Random walk: create appropriate state space and transition matrix
    % Use a reasonable range around the initial demand
    expected_range = 3/2 * params.sigma * sqrt(params.T);
    
    min_state = max(0, params.P_d0 - expected_range);
    max_state = params.P_d0 + expected_range;
    model.levels = linspace(min_state, max_state, params.n_states);
    model.P = create_transition_matrix('gaussian_random_walk', ...
        model.levels, params.sigma);
    model.sigma = params.sigma; % Keep for threshold calculations
end