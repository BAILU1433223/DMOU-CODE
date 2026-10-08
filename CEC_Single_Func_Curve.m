function CEC_Single_Func_Curve()
% =========================================================================
% CEC Single Function Test Program
% Purpose: Test a single specified CEC benchmark function with all
%          comparison algorithms and display their convergence curves
%          (no images are saved).
% The user can freely modify the following in the "User Parameters" block:
%       - CEC test suite year (2017 or 2022)
%       - Test function number
%       - Population size (number of particles)
%       - Maximum number of iterations
%       - Problem dimension
% =========================================================================
    clear; clc; close all;

    %% ==================== User Parameters (edit freely) ====================
    CEC_Year   = 2022;   % Test suite: 2017 or 2022
    Func_Num   = 1;      % Function number (CEC2017: 1, 3:30; CEC2022: 1:12)
    N_init     = 50;     % Population size (number of particles)
    Max_iter   = 3000;   % Maximum number of iterations
    Dim        = 20;     % Problem dimension
    
    % List of algorithms to compare ('DMOU' is the proposed algorithm;
    % the rest are baselines).
    Algo_names = [{'DMOU'}, {'PSO','GWO','MVO','SCA','AO','DBO','INFO','NRBO','DP'}];
    % ========================================================================

    % Boundaries (official CEC standard bounds)
    LB = -100; UB = 100;

    % -------- Load the CEC benchmark function --------
    root_dir = pwd;
    addpath(root_dir);
    if CEC_Year == 2017
        suite_dir = 'CEC2017';
        fobj = @(x) cec17_func(x', Func_Num);
        fprintf('===== CEC2017 F%d | Dim=%d | N=%d | Iter=%d =====\n', ...
                Func_Num, Dim, N_init, Max_iter);
    else
        suite_dir = 'CEC2022';
        fobj = @(x) cec22_test_func(x', Func_Num);
        fprintf('===== CEC2022 F%d | Dim=%d | N=%d | Iter=%d =====\n', ...
                Func_Num, Dim, N_init, Max_iter);
    end
    suite_path = fullfile(root_dir, suite_dir);
    if ~exist(suite_path, 'dir')
        error('CEC function folder does not exist: %s', suite_path);
    end
    cd(suite_path);

    % -------- Run each algorithm --------
    num_algos   = length(Algo_names);
    curves_all  = zeros(num_algos, Max_iter);  % Store each convergence curve
    best_all    = zeros(num_algos, 1);         % Final best value
    time_all    = zeros(num_algos, 1);         % Runtime

    for a = 1:num_algos
        algo_name = Algo_names{a};
        fprintf('Running algorithm: %-6s ... ', algo_name);

        rng(1000 + a, 'twister');   % Fixed random seed for reproducibility
        t_start = tic;
        try
            if strcmp(algo_name, 'DMOU')
                [best_f, ~, curve] = DMOU(N_init, Max_iter, LB, UB, Dim, fobj);
            else
                [best_f, ~, curve] = Baseline_Algos_Library( ...
                    algo_name, N_init, Max_iter, LB, UB, Dim, fobj);
            end
        catch ME
            warning('Algorithm %s failed: %s', algo_name, ME.message);
            best_f = NaN;
            curve  = NaN(1, Max_iter);
        end
        time_all(a) = toc(t_start);

        % Unify curve length
        curve = curve(:)';
        if length(curve) < Max_iter
            curve = [curve, NaN(1, Max_iter - length(curve))];
        elseif length(curve) > Max_iter
            curve = curve(1:Max_iter);
        end
        curves_all(a, :) = curve;
        best_all(a)      = best_f;

        fprintf('best = %.6e, time = %.2f s\n', best_f, time_all(a));
    end

    cd(root_dir);
    rmpath(root_dir);

    %% ==================== Plot convergence curves ====================
    figure('Position', [200, 150, 900, 620], ...
           'Name', sprintf('CEC%d F%d Convergence Curves', CEC_Year, Func_Num));

    hold on;
    line_styles = {'-','--','-.',':','-','--','-.',':','-','--'};
    markers     = {'o','s','^','d','v','>','<','p','h','x'};
    colors      = lines(num_algos);

    for a = 1:num_algos
        y = curves_all(a, :);
        % Handle non-positive / NaN values for log-scale plotting
        y(y <= 0) = eps;
        y(isnan(y)) = NaN;
        plot(1:Max_iter, y, ...
             'LineStyle', line_styles{mod(a-1,length(line_styles))+1}, ...
             'Color', colors(a,:), ...
             'Marker', markers{mod(a-1,length(markers))+1}, ...
             'MarkerIndices', 1:max(1,floor(Max_iter/15)):Max_iter, ...
             'MarkerSize', 5, ...
             'LineWidth', 1.5, ...
             'DisplayName', Algo_names{a});
    end
    hold off;

    set(gca, 'YScale', 'log');
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Best Fitness (log scale)', 'FontSize', 12);
    title(sprintf('CEC%d F%d Convergence Curves  (Dim=%d, N=%d, Iter=%d)', ...
                  CEC_Year, Func_Num, Dim, N_init, Max_iter), 'FontSize', 13);
    legend('Location', 'northeastoutside', 'FontSize', 10);
    grid on; box on;

    %% ==================== Print result summary ====================
    fprintf('\n----------------- Result Summary -----------------\n');
    fprintf('%-8s %-18s %-10s\n', 'Algorithm', 'Final Best', 'Time (s)');
    fprintf('--------------------------------------------------\n');
    for a = 1:num_algos
        fprintf('%-8s %-18.6e %-10.2f\n', ...
                Algo_names{a}, best_all(a), time_all(a));
    end
    fprintf('--------------------------------------------------\n');
end