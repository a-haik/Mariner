% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\ThresholdStochasticControl.m
classdef ThresholdControl < ControlLaw
    properties
        k_s
        p_nom
    end
    
    methods
        function obj = ThresholdControl(k_s, p_nom)
            obj.k_s = k_s;
            obj.p_nom = p_nom;
        end
        
        function n_control = compute(obj, P_d, n0)
            T = length(P_d);
            n_control = zeros(1, T);
            n_control(1) = n0;
            
            for t = 2:T
                x = P_d(t)/obj.p_nom - n_control(t-1);
                x_thres = 2* 0.5 * (n_control(t-1) * obj.k_s / max(1, T-t) + 1);
                
                if x > x_thres
                    n_control(t) = n_control(t-1) + 1;
                elseif x < -x_thres
                    n_control(t) = n_control(t-1) - 1;
                else
                    n_control(t) = n_control(t-1);
                end
            end
        end
    end
end