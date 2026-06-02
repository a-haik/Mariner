% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Stochastic_programming\Demand.m
classdef Demand < handle
    properties
        T
        P_d0
        sigma  % Keep for backward compatibility with random walk
        P_d
        % Markov Chain properties (always used)
        states           % Discrete state space
        transition_matrix % Transition probability matrix
    end
    
    methods
        function obj = Demand(T, P_d0, varargin)
            obj.T = T;
            obj.P_d0 = P_d0;
            
            % Parse input arguments
            if length(varargin) == 1
                % Random walk mode: Demand(T, P_d0, sigma)
                % Convert random walk to discrete Markov chain
                obj.sigma = varargin{1};
                obj = obj.create_random_walk_markov_chain();
            elseif length(varargin) == 2
                % Direct Markov chain: Demand(T, P_d0, states, transition_matrix)
                obj.states = varargin{1};
                obj.transition_matrix = varargin{2};
                obj.sigma = [];
            else
                error('Invalid number of arguments');
            end
            
            obj.P_d = obj.simulate_demand();
        end
        
        function obj = create_random_walk_markov_chain(obj)
            % Convert random walk parameters to discrete Markov chain
            % Create a fine-grained state space around expected range
            expected_range = 3 * obj.sigma * sqrt(obj.T); % 3-sigma range over T steps
            state_step = obj.sigma / 2; % Fine discretization
            
            min_state = max(0, obj.P_d0 - expected_range);
            max_state = obj.P_d0 + expected_range;
            obj.states = min_state:state_step:max_state;
            
            % Create transition matrix based on Gaussian random walk
            n_states = length(obj.states);
            obj.transition_matrix = zeros(n_states, n_states);
            
            for i = 1:n_states
                current_state = obj.states(i);
                for j = 1:n_states
                    next_state = obj.states(j);
                    delta = next_state - current_state;
                    
                    % Gaussian transition probability
                    prob = exp(-0.5 * (delta / obj.sigma)^2) / (obj.sigma * sqrt(2 * pi));
                    obj.transition_matrix(i, j) = prob;
                end
                
                % Normalize row to sum to 1
                obj.transition_matrix(i, :) = obj.transition_matrix(i, :) / sum(obj.transition_matrix(i, :));
                
                % Handle boundary conditions (absorbing at 0)
                if obj.states(i) <= 0
                    obj.transition_matrix(i, :) = 0;
                    obj.transition_matrix(i, 1) = 1; % Stay at minimum state
                end
            end
        end
        
        function P_d = simulate_demand(obj)
            P_d = zeros(1, obj.T);
            
            % Find closest state to initial demand
            [~, current_state_idx] = min(abs(obj.states - obj.P_d0));
            P_d(1) = obj.states(current_state_idx);
            
            % Markov chain simulation (works for both random walk and general cases)
            for t = 2:obj.T
                % Sample next state based on transition probabilities
                probs = obj.transition_matrix(current_state_idx, :);
                cumprobs = cumsum(probs);
                r = rand();
                next_state_idx = find(r <= cumprobs, 1, 'first');
                current_state_idx = next_state_idx;
                P_d(t) = obj.states(current_state_idx);
            end
        end
    end
end