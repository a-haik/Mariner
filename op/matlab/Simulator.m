% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\Simulator.m
classdef Simulator < handle
    properties
        T
        n0
        k_s
        P_d
        n
        C_o
        C_s
        C
        p_star
    end
    
    methods
        function obj = Simulator(p_star, P_d, n0, k_s)
            obj.T = numel(P_d);
            obj.n0 = n0;
            obj.k_s = k_s;
            obj.P_d = P_d;
            obj.p_star = p_star;
        end
        
        function TC = run(obj, control_law)
            n = control_law.compute(obj.P_d, obj.n0);
            [C_o, C_s, C, TC] = obj.calculate_cost(obj.P_d, n, obj.k_s);
            
            % Store data for plotting
            obj.n = n;
            obj.C_o = C_o;
            obj.C_s = C_s;
            obj.C = C;
        end
        
        function [C_o, C_s, C, TC] = calculate_cost(obj, P_d, n, k_s)
            % Operating cost
            C_o = ((P_d/obj.p_star - n).^2) ./ n;

            % Switching cost for continuous n
            n_diff = [0, diff(n)];
            C_s = k_s * abs(n_diff);

            % Total cost
            C = C_o + C_s;
            TC = sum(C);
        end
    end
end