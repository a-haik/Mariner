% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\create_transition_matrix.m
function P = create_transition_matrix(type, varargin)
% CREATE_TRANSITION_MATRIX Creates various types of transition matrices for Markov chains
%
% Usage:
%   P = create_transition_matrix('persistent', n_states, persistence)
%   P = create_transition_matrix('drift', n_states, drift_prob, drift_direction)
%   P = create_transition_matrix('random_walk_like', n_states, step_prob)
%   P = create_transition_matrix('custom', transition_matrix)
%
% Parameters:
%   type: Type of transition matrix to create
%   n_states: Number of states in the Markov chain
%   persistence: Probability of staying in current state (0-1)
%   drift_prob: Probability of drift movement (0-1)
%   drift_direction: 1 for upward drift, -1 for downward drift
%   step_prob: Probability of taking a step (vs staying in place)
%   transition_matrix: Custom user-provided matrix

switch lower(type)
    case 'persistent'
        % States tend to persist with some random transitions
        n_states = varargin{1};
        persistence = varargin{2};
        P = create_persistent_matrix(n_states, persistence);
        
    case 'drift'
        % States have a drift tendency (upward or downward)
        n_states = varargin{1};
        drift_prob = varargin{2};
        drift_direction = varargin{3};
        P = create_drift_matrix(n_states, drift_prob, drift_direction);
        
    case 'random_walk_like'
        % Similar to random walk but with discrete states
        n_states = varargin{1};
        step_prob = varargin{2};
        P = create_random_walk_like_matrix(n_states, step_prob);
        
    case 'gaussian_random_walk'
        % Convert continuous Gaussian random walk to discrete Markov chain
        states = varargin{1};
        sigma = varargin{2};
        P = create_gaussian_random_walk_matrix(states, sigma);
        
    case 'custom'
        % User provides their own matrix
        P = varargin{1};
        validate_transition_matrix(P);
        
    otherwise
        error('Unknown transition matrix type: %s', type);
end

% Validate the resulting matrix
validate_transition_matrix(P);
end

function P = create_gaussian_random_walk_matrix(states, sigma)
% Creates a transition matrix that represents a discretized Gaussian random walk
n_states = length(states);
P = zeros(n_states, n_states);

for i = 1:n_states
    current_state = states(i);
    for j = 1:n_states
        next_state = states(j);
        delta = next_state - current_state;
        
        % Gaussian transition probability
        P(i, j) = exp(-0.5 * (delta / sigma)^2) / (sigma * sqrt(2 * pi));
    end
    
    % Normalize row to sum to 1
    P(i, :) = P(i, :) / sum(P(i, :));
    
    % Handle boundary conditions (absorbing at 0 or negative states)
    if states(i) <= 0
        P(i, :) = 0;
        P(i, 1) = 1; % Stay at minimum state
    end
end
end

function P = create_persistent_matrix(n_states, persistence)
% Creates a matrix where states tend to persist
P = zeros(n_states, n_states);
remaining_prob = 1 - persistence;

for i = 1:n_states
    P(i, i) = persistence;
    % Distribute remaining probability uniformly among other states
    other_prob = remaining_prob / (n_states - 1);
    for j = 1:n_states
        if i ~= j
            P(i, j) = other_prob;
        end
    end
end
end

function P = create_drift_matrix(n_states, drift_prob, drift_direction)
% Creates a matrix with drift tendency
P = zeros(n_states, n_states);
stay_prob = 0.5;  % Base probability to stay

for i = 1:n_states
    P(i, i) = stay_prob;
    
    if drift_direction > 0 && i < n_states
        % Upward drift
        P(i, i+1) = drift_prob;
        remaining = 1 - stay_prob - drift_prob;
    elseif drift_direction < 0 && i > 1
        % Downward drift
        P(i, i-1) = drift_prob;
        remaining = 1 - stay_prob - drift_prob;
    else
        % At boundary or no drift
        remaining = 1 - stay_prob;
    end
    
    % Distribute remaining probability
    if remaining > 0
        available_states = setdiff(1:n_states, find(P(i, :) > 0));
        if ~isempty(available_states)
            P(i, available_states) = remaining / length(available_states);
        end
    end
    
    % Normalize
    P(i, :) = P(i, :) / sum(P(i, :));
end
end

function P = create_random_walk_like_matrix(n_states, step_prob)
% Creates a matrix similar to discretized random walk
P = zeros(n_states, n_states);
stay_prob = 1 - step_prob;

for i = 1:n_states
    P(i, i) = stay_prob;
    
    % Equal probability to step left or right (if possible)
    step_left_prob = 0;
    step_right_prob = 0;
    
    if i > 1
        step_left_prob = step_prob / 2;
        P(i, i-1) = step_left_prob;
    end
    
    if i < n_states
        step_right_prob = step_prob / 2;
        P(i, i+1) = step_right_prob;
    end
    
    % If at boundary, redistribute step probability
    if i == 1
        P(i, i) = P(i, i) + step_prob / 2;  % Can't step left
    elseif i == n_states
        P(i, i) = P(i, i) + step_prob / 2;  % Can't step right
    end
    
    % Normalize
    P(i, :) = P(i, :) / sum(P(i, :));
end
end

function validate_transition_matrix(P)
% Validates that P is a proper stochastic matrix
if size(P, 1) ~= size(P, 2)
    error('Transition matrix must be square');
end

if any(P(:) < 0)
    error('Transition matrix elements must be non-negative');
end

row_sums = sum(P, 2);
if any(abs(row_sums - 1) > 1e-10)
    error('Each row of transition matrix must sum to 1');
end
end
