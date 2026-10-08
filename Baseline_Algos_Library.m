function [best_f, best_x, curve] = Baseline_Algos_Library(algo_name, N_init, Max_iter, LB, UB, Dim, fobj)
% Baseline_Algos_Library - Dispatcher for baseline metaheuristic algorithms.
%
% Inputs:
%   algo_name - name of the algorithm (case-insensitive)
%   N_init    - population size
%   Max_iter  - maximum number of iterations
%   LB, UB    - lower / upper bounds (scalar or vector)
%   Dim       - problem dimension
%   fobj      - objective function handle (accepts a row vector)
%
% Outputs:
%   best_f - best fitness value
%   best_x - best solution vector
%   curve  - convergence curve (1 x Max_iter)

    switch upper(algo_name)
        case 'PSO'   % 1995 - Particle Swarm Optimization
            [best_f, best_x, curve] = run_PSO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'GWO'   % 2014 - Grey Wolf Optimizer
            [best_f, best_x, curve] = run_GWO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'SCA'   % 2016 - Sine Cosine Algorithm
            [best_f, best_x, curve] = run_SCA(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'MVO'   % 2016 - Multi-Verse Optimizer
            [best_f, best_x, curve] = run_MVO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'AO'    % 2021 - Aquila Optimizer
            [best_f, best_x, curve] = run_AO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'INFO'  % 2022 - Weighted Mean of Vectors
            [best_f, best_x, curve] = run_INFO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'DBO'   % 2022 - Dung Beetle Optimizer
            [best_f, best_x, curve] = run_DBO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'DP'    % 2025 - Delta Plus
            [best_f, best_x, curve] = run_DP(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'NRBO'  % 2024 - Newton-Raphson-Based Optimizer
            [best_f, best_x, curve] = run_NRBO(N_init, Max_iter, LB, UB, Dim, fobj);
        otherwise
            error(['Unknown algorithm name! Choose from: ' ...
                   'PSO, GWO, SCA, MVO, AO, INFO, DBO, DP, NRBO']);
    end
end


%% ==================== GWO: Grey Wolf Optimizer ====================
function [Best_F, Best_X, Curve] = run_GWO(N_init, Max_iter, LB, UB, Dim, fobj)
% Grey Wolf Optimizer (GWO)
% Reference: Mirjalili et al., Advances in Engineering Software, 2014
% DOI: 10.1016/j.advengsoft.2013.12.007

    if isscalar(LB), LB = LB * ones(1, Dim); UB = UB * ones(1, Dim); end
    lb = LB; ub = UB;

    % ----- Initialize alpha, beta, delta -----
    Alpha_pos = zeros(1, Dim);
    Alpha_score = inf;   % minimization
    Beta_pos = zeros(1, Dim);
    Beta_score = inf;
    Delta_pos = zeros(1, Dim);
    Delta_score = inf;

    % ----- Initialize positions -----
    Positions = rand(N_init, Dim) .* (ub - lb) + lb;

    Curve = zeros(1, Max_iter);

    % ----- Main loop -----
    for l = 1:Max_iter
        % Boundary check and fitness evaluation
        for i = 1:N_init
            Positions(i, :) = max(min(Positions(i, :), ub), lb);
            fitness = fobj(Positions(i, :));

            if fitness < Alpha_score
                Alpha_score = fitness;
                Alpha_pos = Positions(i, :);
            elseif fitness < Beta_score && fitness > Alpha_score
                Beta_score = fitness;
                Beta_pos = Positions(i, :);
            elseif fitness < Delta_score && fitness > Beta_score
                Delta_score = fitness;
                Delta_pos = Positions(i, :);
            end
        end

        % Linearly decreasing parameter a from 2 to 0
        a = 2 - l * (2 / Max_iter);

        % Update positions of all agents
        for i = 1:N_init
            for j = 1:Dim
                % Alpha
                r1 = rand(); r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - A1 * D_alpha;

                % Beta
                r1 = rand(); r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - A2 * D_beta;

                % Delta
                r1 = rand(); r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                X3 = Delta_pos(j) - A3 * D_delta;

                Positions(i, j) = (X1 + X2 + X3) / 3;
            end
        end

        Curve(l) = Alpha_score;
    end

    Best_F = Alpha_score;
    Best_X = Alpha_pos;
end


%% ==================== PSO: Particle Swarm Optimization ====================
function [Best_F, Best_X, curve] = run_PSO(N, Max_iter, LB, UB, Dim, fobj)
% Standard Particle Swarm Optimization
% Reference: Kennedy & Eberhart, 1995

    if isscalar(LB), LB = LB * ones(1, Dim); UB = UB * ones(1, Dim); end
    lb = LB; ub = UB;

    % PSO parameters
    w  = 0.729;   % inertia weight
    c1 = 1.494;   % cognitive parameter
    c2 = 1.494;   % social parameter

    % Initialize positions and velocities
    X = rand(N, Dim) .* (ub - lb) + lb;
    V = zeros(N, Dim);
    pbest_pos = X;
    pbest_fit = inf(N, 1);
    curve = zeros(Max_iter, 1);

    for i = 1:N
        fit = fobj(X(i, :));
        pbest_fit(i) = fit;
    end

    [gbest_fit, gbest_idx] = min(pbest_fit);
    gbest_pos = X(gbest_idx, :);
    curve(1) = gbest_fit;

    for t = 2:Max_iter
        r1 = rand(N, Dim);
        r2 = rand(N, Dim);
        V = w * V + c1 * r1 .* (pbest_pos - X) + c2 * r2 .* (gbest_pos - X);
        X = X + V;

        % Boundary handling
        X = max(min(X, ub), lb);

        for i = 1:N
            fit = fobj(X(i, :));
            if fit < pbest_fit(i)
                pbest_fit(i) = fit;
                pbest_pos(i, :) = X(i, :);
                if fit < gbest_fit
                    gbest_fit = fit;
                    gbest_pos = X(i, :);
                end
            end
        end
        curve(t) = gbest_fit;
    end

    Best_F = gbest_fit;
    Best_X = gbest_pos;
end


%% ==================== DP: Delta Plus ====================
function [Best_F, Best_X, curve] = run_DP(N, Max_iter, LB, UB, Dim, fobj)
% Delta Plus (standalone, official code embedded)
% Reference: Yuansheng Gao et al., "Freedom from Inspiration: Achieving
%            Efficient Metaheuristic Optimization with Delta", 2025.

    % ----- 1. Boundary handling -----
    if isscalar(LB)
        l = LB;
        u = UB;
    else
        l = LB(:)';
        u = UB(:)';
    end

    % ----- 2. Parameter setting -----
    n   = N;
    dim = Dim;
    T   = N * Max_iter;   % total number of evaluations
    F   = fobj;

    % ----- 3. Initialization -----
    if isscalar(u)
        x = rand(n, dim) .* (u - l) + l;
    else
        x = zeros(n, dim);
        for i = 1:dim
            ubi = u(i);
            lbi = l(i);
            x(:, i) = rand(n, 1) .* (ubi - lbi) + lbi;
        end
    end

    % ----- 4. Initial fitness evaluation -----
    f = zeros(n, 1);
    best_f = inf;
    t = 0;
    ConvergenceCurve = zeros(1, T);
    for i = 1:n
        t = t + 1;
        if t > T; break; end
        f(i) = F(x(i, :));
        if best_f > f(i)
            best_f = f(i);
            best_x = x(i, :);
        end
        ConvergenceCurve(t) = best_f;
    end

    % ----- 5. Initialize parameters -----
    v      = zeros(n, dim);
    c      = zeros(n, dim);
    delta  = zeros(n, dim);
    xi     = zeros(n-1, dim);
    mu     = zeros(n, dim);
    f_new  = zeros(n, 1);
    w1 = 1; w2 = 1;
    lbExtended = repmat(l, [n, dim]);
    ubExtended = repmat(u, [n, dim]);

    % ----- 6. Main loop -----
    while t < T
        eta = 1 - t / T;
        lamda = (eta + w1) * norm(mean(x) + w2 * eta * randn(1, dim) - x);

        for i = 1:n
            % Eq.(2)
            for j = setdiff(1:n, i)
                xi(j, :) = rand * (x(j, :) - x(i, :));
            end
            % Eq.(3)
            mu(i, :) = sum(rand(1, dim) .* xi, 1);
            v(i, :)  = mu(i, :) / (norm(mu(i, :)) + eps);
            % Eq.(4)
            if t / n == 1
                c(i, :) = v(i, :);
            else
                c(i, :) = delta(i, :) / (norm(delta(i, :)) + eps);
            end
            % Eq.(5)
            delta(i, :) = lamda * (rand * v(i, :) + rand * c(i, :));
        end

        % New candidate solutions
        x_new = x + delta;

        % Boundary handling
        lbViolated = x_new < lbExtended;
        ubViolated = x_new > ubExtended;
        x_new(lbViolated) = lbExtended(lbViolated);
        x_new(ubViolated) = ubExtended(ubViolated);

        % Eq.(6)
        for i = 1:n
            f_new(i) = F(x_new(i, :));
            t = t + 1;
            if t > T; break; end
            if f(i) > f_new(i)
                f(i) = f_new(i);
                x(i, :) = x_new(i, :);
            end
            if best_f > f(i)
                best_f = f(i);
                best_x = x(i, :);
            end
            ConvergenceCurve(t) = best_f;
        end
    end

    % ----- 7. Truncate convergence curve to actual length -----
    ConvergenceCurve = ConvergenceCurve(1:t);

    % ----- 8. Extract per-iteration best -----
    curve = zeros(1, Max_iter);
    for iter = 1:Max_iter
        idx = min(iter * N, length(ConvergenceCurve));
        curve(iter) = ConvergenceCurve(idx);
    end

    Best_F = best_f;
    Best_X = best_x;
end


%% ==================== AO: Aquila Optimizer ====================
function [Best_F, Best_X, curve] = run_AO(N, Max_iter, LB, UB, Dim, fobj)
% Aquila Optimizer (with evaluation-count correction; only new solutions
% are evaluated each generation).
% Reference: Abualigah et al., Computers & Industrial Engineering, 2021

    if isscalar(LB)
        LB = LB * ones(1, Dim);
        UB = UB * ones(1, Dim);
    end
    lb = LB; ub = UB;

    % Initialize population
    X = rand(N, Dim) .* (ub - lb) + lb;
    Ffun = zeros(1, N);
    Best_P = zeros(1, Dim);
    Best_FF = inf;

    % Initial evaluation
    for i = 1:N
        X(i, :) = max(min(X(i, :), ub), lb);
        Ffun(i) = fobj(X(i, :));
        if Ffun(i) < Best_FF
            Best_FF = Ffun(i);
            Best_P = X(i, :);
        end
    end

    curve = zeros(1, Max_iter);
    curve(1) = Best_FF;

    % Fixed parameters
    alpha = 0.1;
    delta = 0.1;

    for t = 2:Max_iter
        % ---- Global parameters ----
        G2 = 2 * rand() - 1;
        G1 = 2 * (1 - (t / Max_iter));
        to = 1:Dim;
        u  = 0.0265;
        r0 = 10;
        r  = r0 + u * to;
        omega  = 0.005;
        phi0   = 3 * pi / 2;
        phi    = -omega * to + phi0;
        x = r .* sin(phi);
        y = r .* cos(phi);
        QF = t ^ ((2 * rand() - 1) / (1 - Max_iter)^2);

        % ---- Update each individual (only new solutions evaluated) ----
        Xnew = zeros(N, Dim);
        Fnew = zeros(1, N);
        for i = 1:N
            if t <= (2/3) * Max_iter
                if rand < 0.5
                    % Extended exploration (X1)
                    Xnew(i, :) = Best_P * (1 - t / Max_iter) + ...
                                 (mean(X, 1) - Best_P) * rand();
                else
                    % Narrowed exploration (X2)
                    Xnew(i, :) = Best_P .* Levy(Dim) + ...
                                 X(floor(N * rand() + 1), :) + ...
                                 (y - x) * rand();
                end
            else
                if rand < 0.5
                    % Extended exploitation (X3)
                    Xnew(i, :) = (Best_P - mean(X, 1)) * alpha - rand + ...
                                 ((ub - lb) * rand + lb) * delta;
                else
                    % Narrowed exploitation (X4)
                    Xnew(i, :) = QF * Best_P - (G2 * X(i, :) * rand) - ...
                                 G1 .* Levy(Dim) + rand * G2;
                end
            end

            % Boundary handling
            Xnew(i, :) = max(min(Xnew(i, :), ub), lb);
            Fnew(i) = fobj(Xnew(i, :));

            % Greedy selection
            if Fnew(i) < Ffun(i)
                X(i, :) = Xnew(i, :);
                Ffun(i) = Fnew(i);
                if Fnew(i) < Best_FF
                    Best_FF = Fnew(i);
                    Best_P  = X(i, :);
                end
            end
        end

        curve(t) = Best_FF;
    end

    Best_F = Best_FF;
    Best_X = Best_P;

    % ---- Nested Levy flight ----
    function o = Levy(d)
        beta = 1.5;
        sigma = (gamma(1+beta) * sin(pi*beta/2) / ...
                (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);
        u = randn(1, d) * sigma;
        v = randn(1, d);
        step = u ./ abs(v).^(1/beta);
        o = step;
    end
end


%% ==================== MVO: Multi-Verse Optimizer ====================
function [Best_F, Best_X, Curve] = run_MVO(N, Max_iter, LB, UB, Dim, fobj)
% Multi-Verse Optimizer (official implementation, self-contained)
% Reference: Mirjalili et al., Neural Computing and Applications, 2016

    % ---------- Unify bounds as row vectors ----------
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)';
        ub = UB(:)';
    end

    % ---------- Initialization ----------
    if size(ub, 2) == 1
        Universes = rand(N, Dim) .* (ub - lb) + lb;
    else
        Universes = zeros(N, Dim);
        for i = 1:Dim
            ub_i = ub(i);
            lb_i = lb(i);
            Universes(:, i) = rand(N, 1) .* (ub_i - lb_i) + lb_i;
        end
    end

    Best_universe = zeros(1, Dim);
    Best_universe_Inflation_rate = inf;
    Curve = zeros(1, Max_iter);

    % Algorithm parameters
    WEP_Max = 1;
    WEP_Min = 0.2;

    % ---------- Main loop ----------
    for Time = 1:Max_iter
        % Update WEP and TDR
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_iter);
        TDR = 1 - ((Time)^(1/6) / (Max_iter)^(1/6));

        % Boundary check and fitness
        Inflation_rates = zeros(1, N);
        for i = 1:N
            Flag4ub = Universes(i, :) > ub;
            Flag4lb = Universes(i, :) < lb;
            Universes(i, :) = (Universes(i, :) .* (~(Flag4ub + Flag4lb))) ...
                              + ub .* Flag4ub + lb .* Flag4lb;
            Inflation_rates(i) = fobj(Universes(i, :));
            if Inflation_rates(i) < Best_universe_Inflation_rate
                Best_universe_Inflation_rate = Inflation_rates(i);
                Best_universe = Universes(i, :);
            end
        end

        % Sort
        [sorted_Inflation_rates, sorted_indexes] = sort(Inflation_rates);
        Sorted_universes = Universes(sorted_indexes, :);

        % Normalize inflation rates (for roulette wheel)
        if range(sorted_Inflation_rates) == 0
            normalized_sorted_Inflation_rates = ones(1, N) / N;
        else
            normalized_sorted_Inflation_rates = (sorted_Inflation_rates - min(sorted_Inflation_rates)) / ...
                                                (max(sorted_Inflation_rates) - min(sorted_Inflation_rates) + eps);
        end

        % Elitism
        Universes(1, :) = Sorted_universes(1, :);

        % Update other universes
        for i = 2:N
            Back_hole_index = i;
            for j = 1:Dim
                % White/black hole exchange (roulette wheel)
                r1 = rand();
                if r1 < normalized_sorted_Inflation_rates(i)
                    White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates);
                    if White_hole_index == -1
                        White_hole_index = 1;
                    end
                    Universes(Back_hole_index, j) = Sorted_universes(White_hole_index, j);
                end

                % Wormhole existence probability (WEP) and travelling distance (TDR)
                r2 = rand();
                if r2 < WEP
                    r3 = rand();
                    if size(lb, 2) == 1
                        if r3 < 0.5
                            Universes(i, j) = Best_universe(j) + TDR * ((ub - lb) * rand + lb);
                        else
                            Universes(i, j) = Best_universe(j) - TDR * ((ub - lb) * rand + lb);
                        end
                    else
                        if r3 < 0.5
                            Universes(i, j) = Best_universe(j) + TDR * ((ub(j) - lb(j)) * rand + lb(j));
                        else
                            Universes(i, j) = Best_universe(j) - TDR * ((ub(j) - lb(j)) * rand + lb(j));
                        end
                    end
                end
            end
        end

        Curve(Time) = Best_universe_Inflation_rate;
    end

    Best_F = Best_universe_Inflation_rate;
    Best_X = Best_universe;

    % ---------- Nested function: roulette wheel selection ----------
    function choice = RouletteWheelSelection(weights)
        accumulation = cumsum(weights);
        p = rand() * accumulation(end);
        chosen_index = -1;
        for index = 1:length(accumulation)
            if accumulation(index) > p
                chosen_index = index;
                break;
            end
        end
        choice = chosen_index;
    end
end


%% ==================== SCA: Sine Cosine Algorithm ====================
function [Best_F, Best_X, Curve] = run_SCA(N, Max_iter, LB, UB, Dim, fobj)
% Sine Cosine Algorithm (official implementation, self-contained)
% Reference: Mirjalili, Knowledge-Based Systems, 2016
% DOI: 10.1016/j.knosys.2015.12.022

    % ---------- Unify bounds as row vectors ----------
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)';
        ub = UB(:)';
    end

    % ---------- Initialization ----------
    if size(ub, 2) == 1
        X = rand(N, Dim) .* (ub - lb) + lb;
    else
        X = zeros(N, Dim);
        for i = 1:Dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(N, 1) .* (ub_i - lb_i) + lb_i;
        end
    end

    % ---------- Initialize best solution ----------
    Destination_position = zeros(1, Dim);
    Destination_fitness  = inf;
    Curve = zeros(1, Max_iter);

    for i = 1:N
        Flag4ub = X(i, :) > ub;
        Flag4lb = X(i, :) < lb;
        X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        fitness = fobj(X(i, :));
        if fitness < Destination_fitness
            Destination_fitness  = fitness;
            Destination_position = X(i, :);
        end
    end
    Curve(1) = Destination_fitness;

    % ---------- Main loop ----------
    t = 2;
    while t <= Max_iter
        % r1 linearly decreases from 2 to 0
        a  = 2;
        r1 = a - t * (a / Max_iter);

        for i = 1:N
            for j = 1:Dim
                r2 = 2 * pi * rand();
                r3 = 2 * rand();
                r4 = rand();

                if r4 < 0.5
                    % Eq. (3.1)
                    X(i, j) = X(i, j) + r1 * sin(r2) * abs(r3 * Destination_position(j) - X(i, j));
                else
                    % Eq. (3.2)
                    X(i, j) = X(i, j) + r1 * cos(r2) * abs(r3 * Destination_position(j) - X(i, j));
                end
            end

            % Boundary handling
            Flag4ub = X(i, :) > ub;
            Flag4lb = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;

            fitness = fobj(X(i, :));
            if fitness < Destination_fitness
                Destination_fitness  = fitness;
                Destination_position = X(i, :);
            end
        end

        Curve(t) = Destination_fitness;
        t = t + 1;
    end

    Best_F = Destination_fitness;
    Best_X = Destination_position;
end


%% ==================== INFO: Weighted Mean of Vectors ====================
function [Best_F, Best_X, Curve] = run_INFO(N_init, Max_iter, LB, UB, Dim, fobj)
% INFO: Weighted Mean of Vectors Optimizer
% Reference: Ahmadianfar et al., Expert Systems with Applications, 2022

    if isscalar(LB), LB = LB * ones(1, Dim); UB = UB * ones(1, Dim); end
    lb = LB; ub = UB;

    % ----- Initialization -----
    X    = rand(N_init, Dim) .* (ub - lb) + lb;
    Cost = zeros(N_init, 1);
    M    = zeros(N_init, 1);
    for i = 1:N_init
        Cost(i) = fobj(X(i, :));
        M(i)    = Cost(i);
    end

    % Sort and initialize best, worst, better
    [~, ind]     = sort(Cost);
    Best_X       = X(ind(1), :);
    Best_Cost    = Cost(ind(1));
    Worst_X      = X(ind(end), :);
    Worst_Cost   = Cost(ind(end));
    I            = randi([2, 5]);
    Better_X     = X(ind(I), :);
    Better_Cost  = Cost(ind(I));

    Curve = zeros(1, Max_iter);

    % ----- Main loop -----
    for it = 1:Max_iter
        alpha = 2 * exp(-4 * (it / Max_iter));   % Eq. (5.1) & (9.1)

        M_Best   = Best_Cost;
        M_Better = Better_Cost;
        M_Worst  = Worst_Cost;

        for i = 1:N_init
            del  = 2 * rand * alpha - alpha;   % Eq. (5)
            sigm = 2 * rand * alpha - alpha;   % Eq. (9)

            % Select three random solutions (different from i)
            A1 = randperm(N_init);
            A1(A1 == i) = [];
            a = A1(1); b = A1(2); c = A1(3);

            epsi = 1e-25 * rand;

            % Weights for random individuals (Eq. 4.1)
            omg = max([M(a), M(b), M(c)]);
            MM  = [M(a)-M(b), M(a)-M(c), M(b)-M(c)];
            W(1) = cos(MM(1) + pi) * exp(-MM(1)/omg);
            W(2) = cos(MM(2) + pi) * exp(-MM(2)/omg);
            W(3) = cos(MM(3) + pi) * exp(-MM(3)/omg);
            Wt = sum(W);
            WM1 = del * (W(1)*(X(a,:)-X(b,:)) + W(2)*(X(a,:)-X(c,:)) + W(3)*(X(b,:)-X(c,:))) / (Wt + 1) + epsi;

            % Weights for best/better/worst (Eq. 4.6)
            omg = max([M_Best, M_Better, M_Worst]);
            MM  = [M_Best-M_Better, M_Best-M_Better, M_Better-M_Worst];
            W(1) = cos(MM(1) + pi) * exp(-MM(1)/omg);
            W(2) = cos(MM(2) + pi) * exp(-MM(2)/omg);
            W(3) = cos(MM(3) + pi) * exp(-MM(3)/omg);
            Wt = sum(W);
            WM2 = del * (W(1)*(Best_X-Better_X) + W(2)*(Best_X-Worst_X) + W(3)*(Better_X-Worst_X)) / (Wt + 1) + epsi;

            % MeanRule (Eq. 4)
            r = unifrnd(0.1, 0.5);
            MeanRule = r * WM1 + (1 - r) * WM2;

            % Generate two candidate vectors (Eq. 8)
            if rand < 0.5
                z1 = X(i,:) + sigm * (rand * MeanRule) + randn * (Best_X - X(a,:)) / (M_Best - M(a) + 1);
                z2 = Best_X + sigm * (rand * MeanRule) + randn * (X(a,:) - X(b,:)) / (M(a) - M(b) + 1);
            else
                z1 = X(a,:) + sigm * (rand * MeanRule) + randn * (X(b,:) - X(c,:)) / (M(b) - M(c) + 1);
                z2 = Better_X + sigm * (rand * MeanRule) + randn * (X(a,:) - X(b,:)) / (M(a) - M(b) + 1);
            end

            % Vector combining stage (Eq. 10)
            u = zeros(1, Dim);
            for j = 1:Dim
                mu = 0.05 * randn;
                if rand < 0.5
                    if rand < 0.5
                        u(j) = z1(j) + mu * abs(z1(j) - z2(j));   % Eq. 10.1
                    else
                        u(j) = z2(j) + mu * abs(z1(j) - z2(j));   % Eq. 10.2
                    end
                else
                    u(j) = X(i, j);                               % Eq. 10.3
                end
            end

            % Local search stage (Eq. 11)
            if rand < 0.5
                L  = rand < 0.5;
                v1 = (1-L) * 2 * rand + L;
                v2 = rand * L + (1-L);
                Xavg = (X(a,:) + X(b,:) + X(c,:)) / 3;                  % Eq. 11.4
                phi  = rand;
                Xrnd = phi * Xavg + (1-phi) * (phi * Better_X + (1-phi) * Best_X);   % Eq. 11.3
                Randn = L * randn(1, Dim) + (1-L) * randn;
                if rand < 0.5
                    u = Best_X + Randn .* (MeanRule + randn * (Best_X - X(a,:)));     % Eq. 11.1
                else
                    u = Xrnd + Randn .* (MeanRule + randn * (v1 * Best_X - v2 * Xrnd));  % Eq. 11.2
                end
            end

            % Boundary handling
            New_X    = max(min(u, ub), lb);
            New_Cost = fobj(New_X);

            if New_Cost < Cost(i)
                X(i, :)  = New_X;
                Cost(i)  = New_Cost;
                M(i)     = Cost(i);
                if Cost(i) < Best_Cost
                    Best_X    = X(i, :);
                    Best_Cost = Cost(i);
                end
            end
        end

        % Update worst and better solutions for next generation
        [~, ind]    = sort(Cost);
        Worst_X     = X(ind(end), :);
        Worst_Cost  = Cost(ind(end));
        I           = randi([2, 5]);
        Better_X    = X(ind(I), :);
        Better_Cost = Cost(ind(I));

        Curve(it) = Best_Cost;
    end

    Best_F = Best_Cost;
    Best_X = Best_X;
end


%% ==================== DBO: Dung Beetle Optimizer ====================
function [Best_F, Best_X, Curve] = run_DBO(N, Max_iter, LB, UB, Dim, fobj)
% Dung Beetle Optimizer
% Reference: J. Xue & B. Shen, J. Supercomputing, 2022
% Note: Hard-coded segment boundaries (12, 19) inherit from the official
%       release; keep N >= 20 for correctness.

    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)'; ub = UB(:)';
    end

    pop = N;
    M   = Max_iter;
    dim = Dim;

    P_percent = 0.2;
    pNum = round(pop * P_percent);

    x   = zeros(pop, dim);
    fit = zeros(pop, 1);
    for i = 1:pop
        x(i, :) = lb + (ub - lb) .* rand(1, dim);
        fit(i)  = fobj(x(i, :));
    end

    pFit = fit;
    pX   = x;
    XX   = pX;
    [fMin, bestI] = min(fit);
    bestX = x(bestI, :);

    Convergence_curve = zeros(1, M);

    for t = 1:M
        [~, B] = max(fit);
        worse  = x(B, :);
        r2     = rand(1);

        for i = 1:pNum
            if r2 < 0.9
                r1 = rand(1);
                a  = rand(1,1);
                if a > 0.1, a = 1; else, a = -1; end
                x(i, :) = pX(i, :) + 0.3 * abs(pX(i, :) - worse) + a * 0.1 * (XX(i, :));
            else
                aaa = randperm(180, 1);
                if (aaa == 0 || aaa == 90 || aaa == 180)
                    x(i, :) = pX(i, :);
                end
                theta = aaa * pi / 180;
                x(i, :) = pX(i, :) + tan(theta) .* abs(pX(i, :) - XX(i, :));
            end
            x(i, :) = Bounds(x(i, :), lb, ub);
            fit(i)  = fobj(x(i, :));
        end

        [~, bestII] = min(fit);
        bestXX = x(bestII, :);

        R = 1 - t / M;
        Xnew1  = bestXX .* (1 - R);
        Xnew2  = bestXX .* (1 + R);
        Xnew1  = Bounds(Xnew1, lb, ub);
        Xnew2  = Bounds(Xnew2, lb, ub);

        Xnew11 = bestX .* (1 - R);
        Xnew22 = bestX .* (1 + R);
        Xnew11 = Bounds(Xnew11, lb, ub);
        Xnew22 = Bounds(Xnew22, lb, ub);

        for i = (pNum + 1):12
            x(i, :) = bestXX + (rand(1, dim) .* (pX(i, :) - Xnew1)) + ...
                               (rand(1, dim) .* (pX(i, :) - Xnew2));
            x(i, :) = Bounds(x(i, :), Xnew1, Xnew2);
            fit(i)  = fobj(x(i, :));
        end

        for i = 13:19
            x(i, :) = pX(i, :) + randn(1) .* (pX(i, :) - Xnew11) + ...
                               rand(1, dim) .* (pX(i, :) - Xnew22);
            x(i, :) = Bounds(x(i, :), lb, ub);
            fit(i)  = fobj(x(i, :));
        end

        for j = 20:pop
            x(j, :) = bestX + randn(1, dim) .* (abs(pX(j, :) - bestXX) + abs(pX(j, :) - bestX)) / 2;
            x(j, :) = Bounds(x(j, :), lb, ub);
            fit(j)  = fobj(x(j, :));
        end

        XX = pX;
        for i = 1:pop
            if fit(i) < pFit(i)
                pFit(i) = fit(i);
                pX(i, :) = x(i, :);
            end
            if pFit(i) < fMin
                fMin   = pFit(i);
                bestX  = pX(i, :);
            end
        end

        Convergence_curve(t) = fMin;
    end

    Best_F = fMin;
    Best_X = bestX;
    Curve  = Convergence_curve(:)';

    function s = Bounds(s, Lb, Ub)
        temp = s;
        I = temp < Lb; temp(I) = Lb(I);
        J = temp > Ub; temp(J) = Ub(J);
        s = temp;
    end
end


%% ==================== NRBO: Newton-Raphson-Based Optimizer ====================
function [Best_F, Best_X, Curve] = run_NRBO(N, Max_iter, LB, UB, Dim, fobj)
% NRBO - Newton-Raphson-Based Optimizer
% Reference: R. Sowmya, M. Premkumar, P. Jangir,
%            Engineering Applications of Artificial Intelligence,
%            Vol. 128, 107532, 2024. DOI: 10.1016/j.engappai.2023.107532

    % ----- 1. Unify bounds as row vectors -----
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)';
        ub = UB(:)';
    end

    % ----- 2. Call NRBO core -----
    [Best_Score, Best_Pos, CG_curve] = NRBO_core(N, Max_iter, lb, ub, Dim, fobj);

    % ----- 3. Outputs -----
    Best_F = Best_Score;
    Best_X = Best_Pos;
    Curve  = CG_curve(:)';

    % ========== Nested: NRBO core ==========
    function [Best_Score, Best_Pos, CG_curve] = NRBO_core(N, MaxIt, LB, UB, dim, fobj)

        % Deciding factor for trap avoidance operator
        DF = 0.6;

        % Population initialization
        Position = initialization(N, dim, UB, LB);
        Fitness  = zeros(N, 1);
        for i = 1:N
            Fitness(i) = fobj(Position(i,:));
        end

        % Best / worst in initial population
        [~, Ind]     = sort(Fitness);
        Best_Score   = Fitness(Ind(1));
        Best_Pos     = Position(Ind(1),:);
        Worst_Cost   = Fitness(Ind(end));
        Worst_Pos    = Position(Ind(end),:);

        CG_curve = zeros(1, MaxIt);

        % Main optimization loop
        for it = 1:MaxIt
            delta = (1 - ((2 * it) / MaxIt)) .^ 5;

            for i = 1:N
                % Randomly select two different indices
                P1 = randperm(N, 2);
                a1 = P1(1); a2 = P1(2);

                rho = rand * (Best_Pos - Position(i,:)) + ...
                      rand * (Position(a1,:) - Position(a2,:));

                % Newton-Raphson Search Rule
                Flag = 1;
                NRSR = SearchRule(Best_Pos, Worst_Pos, Position(i,:), rho, Flag);
                X1   = Position(i,:) - NRSR + rho;
                X2   = Best_Pos      - NRSR + rho;

                Xupdate = zeros(1, dim);
                for j = 1:dim
                    X3 = Position(i,j) - delta * (X2(j) - X1(j));
                    a1 = rand; a2 = rand;
                    Xupdate(j) = a1 * (a1 * X1(j) + (1 - a2) * X2(j)) + (1 - a2) * X3;
                end

                % Trap avoidance operator
                if rand < DF
                    theta1 = -1 + 2 * rand(); theta2 = -0.5 + rand();
                    beta = rand < 0.5;
                    u1 = beta * 3 * rand + (1 - beta); u2 = beta * rand + (1 - beta);
                    if u1 < 0.5
                        X_TAO = Xupdate + theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + ...
                                theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));
                    else
                        X_TAO = Best_Pos + theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + ...
                                theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));
                    end
                    Xnew = X_TAO;
                else
                    Xnew = Xupdate;
                end

                % Boundary enforcement
                Xnew = min(max(Xnew, LB), UB);

                Xnew_Cost = fobj(Xnew);

                if Xnew_Cost < Fitness(i)
                    Position(i,:) = Xnew;
                    Fitness(i)    = Xnew_Cost;
                    if Fitness(i) < Best_Score
                        Best_Pos   = Position(i,:);
                        Best_Score = Fitness(i);
                    end
                end

                if Fitness(i) > Worst_Cost
                    Worst_Pos  = Position(i,:);
                    Worst_Cost = Fitness(i);
                end
            end

            CG_curve(it) = Best_Score;
        end
    end

    % ========== Nested: population initialization ==========
    function X = initialization(nP, dim, ub, lb)
        Boundary_no = size(ub, 2);
        if Boundary_no == 1
            X = rand(nP, dim) .* (ub - lb) + lb;
        else
            X = zeros(nP, dim);
            for i = 1:dim
                X(:, i) = rand(nP, 1) .* (ub(i) - lb(i)) + lb(i);
            end
        end
    end

    % ========== Nested: Newton-Raphson search rule ==========
    function NRSR = SearchRule(Best_Pos, Worst_Pos, Position, rho, Flag)
        dim  = size(Position, 2);
        DelX = rand(1, dim) .* abs(Best_Pos - Position);
        NRSR = randn * ((Best_Pos - Worst_Pos) .* DelX) ./ (2 * (Best_Pos + Worst_Pos - 2 * Position));
        if Flag == 1
            Xa = Position - NRSR + rho;
        else
            Xa = Best_Pos - NRSR + rho;
        end
        r1 = rand; r2 = rand;
        yp = r1 * (mean(Xa + Position) + r1 * DelX);
        yq = r2 * (mean(Xa + Position) - r2 * DelX);
        NRSR = randn * ((yp - yq) .* DelX) ./ (2 * (yp + yq - 2 * Position));
    end
end