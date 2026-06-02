% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\ControlLaw.m
classdef (Abstract) ControlLaw < handle
    methods (Abstract)
        n = compute(obj, P_d, n0)
    end
end