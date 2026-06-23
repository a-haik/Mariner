% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\ConstantControl.m
classdef ConstantControl < ControlLaw
    methods
        function n = compute(obj, P_d, n0)
            n = repmat(n0, size(P_d));
        end
    end
end