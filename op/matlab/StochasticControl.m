
classdef StochasticControl < ControlLaw
    properties
        k_s
        sigma  % Keep for backward compatibility and threshold calculations
        p_vals   % State space (always discrete)
        n_vals   % Action space
        V        % Value function
        policy   % Policy function
        transition_matrix  % Transition matrix (always used)
        p_star
    end
    
    methods
        function obj = StochasticControl(k_s, p_star, states, transition_matrix, n_vals)
            % Constructor can be called as:
            % StochasticDynamicProgrammingControl(k_s, sigma) - for random walk
            % StochasticDynamicProgrammingControl(k_s, states, transition_matrix) - for general Markov chain
            % Optional parameters for random walk: p_min, p_max, p_steps, n_steps
            
            obj.k_s = k_s;
            obj.p_star = p_star;
            
            % Markov chain mode: (k_s, states, transition_matrix, ...)
            obj.p_vals = states;  % states become p_vals
            obj.transition_matrix = transition_matrix;
            
            obj.n_vals = [];
            obj.V = [];
            obj.policy = [];
            
            obj.n_vals = n_vals;
        end
       
        function n_control = compute(obj, P_d, n0)
            T = length(P_d);
            p_size = length(obj.p_vals);
            n_size = length(obj.n_vals);
            
            % Use transition matrix (unified approach for all Markov chains)
            transition_probs = obj.transition_matrix;
            if size(transition_probs, 1) ~= p_size || size(transition_probs, 2) ~= p_size
                error('Transition matrix size must match the number of states');
            end
            
            % Initialize DP table
            V = inf(T, p_size, n_size);
            policy = zeros(T, p_size, n_size);
            
            % Terminal cost
            for i = 1:p_size
                p_val = obj.p_vals(i);
                for j = 1:n_size
                    n_val = obj.n_vals(j);
                    if n_val > 0
                        V(T, i, j) = ((p_val/obj.p_star - n_val)^2)/n_val;
                    end
                end
            end
            
            % Backward iteration
            for t = (T-1):-1:1
                for i = 1:p_size
                    p_val = obj.p_vals(i);
                    for j = 1:n_size
                        n_val = obj.n_vals(j);
                        if n_val <= 0, continue; end
                        
                        C_o = ((p_val/obj.p_star - n_val)^2)/n_val;
                        best_cost = inf;
                        best_action = 1;
                        
                        for a_idx = 1:n_size
                            n_next = obj.n_vals(a_idx);
                            C_s = obj.k_s * abs(n_next - n_val);
                            
                            exp_future = 0;
                            for i_next = 1:p_size
                                exp_future = exp_future + transition_probs(i, i_next) * V(t+1, i_next, a_idx);
                            end
                            
                            total = C_o + C_s + exp_future;
                            if total < best_cost
                                best_cost = total;
                                best_action = a_idx;
                            end
                        end
                        V(t, i, j) = best_cost;
                        policy(t, i, j) = best_action;
                    end
                end
            end
            
            % Forward pass
            n_control = zeros(1, T);
            [~, idx_p] = min(abs(obj.p_vals - P_d(1)));
            [~, idx_n] = min(abs(obj.n_vals - n0));
            n_control(1) = obj.n_vals(idx_n);
            
            for t = 2:T
                idx_n = policy(t-1, idx_p, idx_n);
                n_control(t) = obj.n_vals(idx_n);
                [~, idx_p] = min(abs(obj.p_vals - P_d(t)));
            end
        end
        
        function threshold = calculate_threshold(obj, current_n, remaining_time)
            if ~isempty(obj.sigma)
                % For random walk (converted to Markov chain), use analytical formula
                T_rem = remaining_time;
                V = obj.sigma^2 * ((T_rem - 1) * T_rem / 2);
                A = 1;
                B = 2 * current_n;
                C = -(current_n + (obj.k_s * current_n * (current_n + 1) - V) / T_rem);
                X = (-B + sqrt(B^2 - 4 * A * C)) / (2 * A);
                threshold = current_n + X;
            else
                % For general Markov chains, use numerical approximation
                % This could be improved with full DP, but provides a reasonable heuristic
                warning('Threshold calculation for general Markov chains is approximate');
                threshold = current_n + 1; % Simple heuristic
            end
        end
        
        function threshold = calculate_downward_threshold(obj, current_n, ~)
            if ~isempty(obj.sigma)
                % For random walk (converted to Markov chain), use analytical formula
                A = 1;
                B = -2 * current_n;
                C = current_n + obj.k_s * current_n * (current_n - 1);
                disc = B^2 - 4 * A * C;
                if disc < 0
                    threshold = NaN;
                    return;
                end
                Z = (2 * current_n - sqrt(disc)) / (2 * A);
                threshold = current_n - Z;
            else
                % For general Markov chains, use numerical approximation
                warning('Downward threshold calculation for general Markov chains is approximate');
                threshold = current_n - 1; % Simple heuristic
            end
        end
    end
end
