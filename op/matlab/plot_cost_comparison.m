function plot_cost_comparison(all_run_data, control_laws)
    total_op_costs = cellfun(@(x) sum(x.C_o), all_run_data);
    total_switch_costs = cellfun(@(x) sum(x.C_s), all_run_data);
    
    controllers = cellfun(@class, control_laws, 'UniformOutput', false);
    
    figure(2); clf;
    x = 1:length(controllers);
    width = 0.35;
    
    bar(x - width/2, total_op_costs, width, 'DisplayName', 'Operational Cost');
    hold on;
    bar(x + width/2, total_switch_costs, width, 'DisplayName', 'Switching Cost');
    
    ylabel('Total Cost');
    title('Total Operational and Switching Costs by Controller');
    set(gca, 'XTick', x, 'XTickLabel', controllers);
    legend('show');
    
    % Add value labels on bars
    for i = 1:length(x)
        text(x(i) - width/2, total_op_costs(i) + max(total_op_costs)*0.01, ...
             sprintf('%.2f', total_op_costs(i)), 'HorizontalAlignment', 'center');
        text(x(i) + width/2, total_switch_costs(i) + max(total_switch_costs)*0.01, ...
             sprintf('%.2f', total_switch_costs(i)), 'HorizontalAlignment', 'center');
    end
    
    % Save figure
    saveas(gcf, 'figures/cost_comparison.fig');
    saveas(gcf, 'figures/cost_comparison.png');
end