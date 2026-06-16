% filepath: c:\Users\ulaa\OneDrive - NORCE\Documents\Hyeff Project\code\Random Walk\matlab\plotting_functions.m
function plot_costs_and_control(all_run_data, control_laws, P_d)
    num_controllers = length(control_laws);
    
    % Calculate global limits
    C_o_values = cellfun(@(x) x.C_o, all_run_data, 'UniformOutput', false);
    C_s_values = cellfun(@(x) x.C_s, all_run_data, 'UniformOutput', false);
    
    C_o_min = min(cellfun(@min, C_o_values));
    C_o_max = max(cellfun(@max, C_o_values));
    C_s_min = min(cellfun(@min, C_s_values));
    C_s_max = max(cellfun(@max, C_s_values));
    P_min = min(P_d);
    P_max = max(P_d);
    
%     figure('Position', [100, 100, 400*num_controllers, 800]);
    figure(1); clf;
    
    for idx = 1:num_controllers
        run_data = all_run_data{idx};
        control_law = control_laws{idx};
        
        % Operating Cost
        subplot(3, num_controllers, idx);
        plot(run_data.C_o);
        title([class(control_law), ' - Operating Cost']);
        xlabel('Time');
        ylabel('Cost');
        ylim([C_o_min, C_o_max]);
        
        % Switching Cost
        subplot(3, num_controllers, idx + num_controllers);
        plot(run_data.C_s, 'Color', [1 0.5 0]);
        title([class(control_law), ' - Switching Cost']);
        xlabel('Time');
        ylabel('Cost');
        ylim([C_s_min, C_s_max+1]);
        
        % Demand vs Control
        subplot(3, num_controllers, idx + 2*num_controllers);
        plot(P_d, 'g', 'DisplayName', 'Power Demand');
        hold on;
        plot(run_data.n, 'm', 'DisplayName', 'Control (n)');
        title([class(control_law), ' - Demand vs Control']);
        xlabel('Time');
        ylabel('Value');
        legend('show');
        ylim([P_min/1.1, P_max*1.1]);
        hold off;
    end
    
    % Save figure
    saveas(gcf, 'figures/costs_and_control.fig');
    saveas(gcf, 'figures/costs_and_control.png');
end

