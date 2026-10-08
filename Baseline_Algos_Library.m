function [best_f, best_x, curve] = Baseline_Algos_Library(algo_name, N_init, Max_iter, LB, UB, Dim, fobj)
      switch upper(algo_name)
        case 'PSO'
        [best_f, best_x, curve] = run_PSO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'GWO'
        [best_f, best_x, curve] = run_GWO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'SCA'
        [best_f, best_x, curve] = run_SCA(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'MVO'
        [best_f, best_x, curve] = run_MVO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'AO'
        [best_f, best_x, curve] = run_AO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'INFO'
        [best_f, best_x, curve] = run_INFO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'DBO'
        [best_f, best_x, curve] = run_DBO(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'PSA'
        [best_f, best_x, curve] = run_PSA(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'DP'
        [best_f, best_x, curve] = run_DP(N_init, Max_iter, LB, UB, Dim, fobj);
        case 'NRBO'
        [best_f, best_x, curve] = run_NRBO(N_init, Max_iter, LB, UB, Dim, fobj);
        otherwise
        error('未知算法名称！请选择 PSO, GWO, SCA, MVO, AO, INFO, DBO, TSO, DP, GSO');
      end
end
function [Best_F, Best_X, Curve] = run_GWO(N_init, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB), LB = LB * ones(1, Dim); UB = UB * ones(1, Dim); end
    lb = LB; ub = UB;
    Alpha_pos = zeros(1, Dim);
    Alpha_score = inf;
    Beta_pos = zeros(1, Dim);
    Beta_score = inf;
    Delta_pos = zeros(1, Dim);
    Delta_score = inf;
    Positions = rand(N_init, Dim) .* (ub - lb) + lb;
    Curve = zeros(1, Max_iter);
    for l = 1:Max_iter
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
        a = 2 - l * (2 / Max_iter);
        for i = 1:N_init
            for j = 1:Dim
                r1 = rand(); r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - A1 * D_alpha;
                r1 = rand(); r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - A2 * D_beta;
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
function [Best_F, Best_X, curve] = run_PSO(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB), LB = LB * ones(1, Dim); UB = UB * ones(1, Dim); end
    lb = LB; ub = UB;
    w = 0.729;
    c1 = 1.494;
    c2 = 1.494;
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
function [Best_F, Best_X, curve] = run_DP(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        l = LB;
        u = UB;
    else
        l = LB(:)';
        u = UB(:)';
    end
    n = N;
    dim = Dim;
    T = N * Max_iter;
    F = fobj;
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
    v = zeros(n, dim);
    c = zeros(n, dim);
    delta = zeros(n, dim);
    xi = zeros(n-1, dim);
    mu = zeros(n, dim);
    f_new = zeros(n, 1);
    w1 = 1;
    w2 = 1;
    lbExtended = repmat(l, [n, dim]);
    ubExtended = repmat(u, [n, dim]);
    while t < T
        eta = 1 - t / T;
        lamda = (eta + w1) * norm(mean(x) + w2 * eta * randn(1, dim) - x);
        for i = 1:n
            for j = setdiff(1:n, i)
                xi(j, :) = rand * (x(j, :) - x(i, :));
            end
            mu(i, :) = sum(rand(1, dim) .* xi, 1);
            v(i, :) = mu(i, :) / (norm(mu(i, :)) + eps);
            if t / n == 1
                c(i, :) = v(i, :);
            else
                c(i, :) = delta(i, :) / (norm(delta(i, :)) + eps);
            end
            delta(i, :) = lamda * (rand * v(i, :) + rand * c(i, :));
        end
        x_new = x + delta;
        lbViolated = x_new < lbExtended;
        ubViolated = x_new > ubExtended;
        x_new(lbViolated) = lbExtended(lbViolated);
        x_new(ubViolated) = ubExtended(ubViolated);
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
    ConvergenceCurve = ConvergenceCurve(1:t);
    curve = zeros(1, Max_iter);
    for iter = 1:Max_iter
        idx = min(iter * N, length(ConvergenceCurve));
        curve(iter) = ConvergenceCurve(idx);
    end
    Best_F = best_f;
    Best_X = best_x;
end
function [Best_F, Best_X, curve] = run_AO(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        LB = LB * ones(1, Dim);
        UB = UB * ones(1, Dim);
    end
    lb = LB; ub = UB;
    X = rand(N, Dim) .* (ub - lb) + lb;
    Ffun = zeros(1, N);
    Best_P = zeros(1, Dim);
    Best_FF = inf;
    for i = 1 : N
        X(i, :) = max(min(X(i, :), ub), lb);
        Ffun(i) = fobj(X(i, :));
        if Ffun(i) < Best_FF
            Best_FF = Ffun(i);
            Best_P = X(i, :);
        end
    end
    curve = zeros(1, Max_iter);
    curve(1) = Best_FF;
    alpha = 0.1;
    delta = 0.1;
    for t = 2 : Max_iter
        G2 = 2 * rand() - 1;
        G1 = 2 * (1 - (t / Max_iter));
        to = 1 : Dim;
        u = 0.0265;
        r0 = 10;
        r = r0 + u * to;
        omega = 0.005;
        phi0 = 3 * pi / 2;
        phi = -omega * to + phi0;
        x = r .* sin(phi);
        y = r .* cos(phi);
        QF = t ^ ((2 * rand() - 1) / (1 - Max_iter)^2);
        Xnew = zeros(N, Dim);
        Fnew = zeros(1, N);
        for i = 1 : N
            if t <= (2/3) * Max_iter
                if rand < 0.5
                    Xnew(i, :) = Best_P * (1 - t / Max_iter) + ...
                                 (mean(X, 1) - Best_P) * rand();
                else
                    Xnew(i, :) = Best_P .* Levy(Dim) + ...
                                 X(floor(N * rand() + 1), :) + ...
                                 (y - x) * rand();
                end
            else
                if rand < 0.5
                    Xnew(i, :) = (Best_P - mean(X, 1)) * alpha - rand + ...
                                 ((ub - lb) * rand + lb) * delta;
                else
                    Xnew(i, :) = QF * Best_P - (G2 * X(i, :) * rand) - ...
                                 G1 .* Levy(Dim) + rand * G2;
                end
            end
            Xnew(i, :) = max(min(Xnew(i, :), ub), lb);
            Fnew(i) = fobj(Xnew(i, :));
            if Fnew(i) < Ffun(i)
                X(i, :) = Xnew(i, :);
                Ffun(i) = Fnew(i);
                if Fnew(i) < Best_FF
                    Best_FF = Fnew(i);
                    Best_P = X(i, :);
                end
            end
        end
        curve(t) = Best_FF;
    end
    Best_F = Best_FF;
    Best_X = Best_P;
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
function [Best_F, Best_X, Curve] = run_MVO(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)';
        ub = UB(:)';
    end
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
    WEP_Max = 1;
    WEP_Min = 0.2;
    for Time = 1:Max_iter
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_iter);
        TDR = 1 - ((Time)^(1/6) / (Max_iter)^(1/6));
        Inflation_rates = zeros(1, N);
        for i = 1:N
            Flag4ub = Universes(i, :) > ub;
            Flag4lb = Universes(i, :) < lb;
            Universes(i, :) = (Universes(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            Inflation_rates(i) = fobj(Universes(i, :));
            if Inflation_rates(i) < Best_universe_Inflation_rate
                Best_universe_Inflation_rate = Inflation_rates(i);
                Best_universe = Universes(i, :);
            end
        end
        [sorted_Inflation_rates, sorted_indexes] = sort(Inflation_rates);
        Sorted_universes = Universes(sorted_indexes, :);
        if range(sorted_Inflation_rates) == 0
            normalized_sorted_Inflation_rates = ones(1, N) / N;
        else
            normalized_sorted_Inflation_rates = (sorted_Inflation_rates - min(sorted_Inflation_rates)) / ...
                                                 (max(sorted_Inflation_rates) - min(sorted_Inflation_rates) + eps);
        end
        Universes(1, :) = Sorted_universes(1, :);
        for i = 2:N
            Back_hole_index = i;
            for j = 1:Dim
                r1 = rand();
                if r1 < normalized_sorted_Inflation_rates(i)
                    White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates);
                    if White_hole_index == -1
                        White_hole_index = 1;
                    end
                    Universes(Back_hole_index, j) = Sorted_universes(White_hole_index, j);
                end
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
function [Best_F, Best_X, Curve] = run_SCA(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)';
        ub = UB(:)';
    end
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
    Destination_position = zeros(1, Dim);
    Destination_fitness = inf;
    Curve = zeros(1, Max_iter);
    for i = 1:N
        Flag4ub = X(i, :) > ub;
        Flag4lb = X(i, :) < lb;
        X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        fitness = fobj(X(i, :));
        if fitness < Destination_fitness
            Destination_fitness = fitness;
            Destination_position = X(i, :);
        end
    end
    Curve(1) = Destination_fitness;
    t = 2;
    while t <= Max_iter
        a = 2;
        r1 = a - t * (a / Max_iter);
        for i = 1:N
            for j = 1:Dim
                r2 = 2 * pi * rand();
                r3 = 2 * rand();
                r4 = rand();
                if r4 < 0.5
                    X(i, j) = X(i, j) + r1 * sin(r2) * abs(r3 * Destination_position(j) - X(i, j));
                else
                    X(i, j) = X(i, j) + r1 * cos(r2) * abs(r3 * Destination_position(j) - X(i, j));
                end
            end
            Flag4ub = X(i, :) > ub;
            Flag4lb = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            fitness = fobj(X(i, :));
            if fitness < Destination_fitness
                Destination_fitness = fitness;
                Destination_position = X(i, :);
            end
        end
        Curve(t) = Destination_fitness;
        t = t + 1;
    end
    Best_F = Destination_fitness;
    Best_X = Destination_position;
end
function [Best_F, Best_X, Curve] = run_INFO(N_init, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB), LB = LB * ones(1, Dim); UB = UB * ones(1, Dim); end
    lb = LB; ub = UB;
    X = rand(N_init, Dim) .* (ub - lb) + lb;
    Cost = zeros(N_init, 1);
    M = zeros(N_init, 1);
    for i = 1:N_init
        Cost(i) = fobj(X(i, :));
        M(i) = Cost(i);
    end
    [~, ind] = sort(Cost);
    Best_X = X(ind(1), :);
    Best_Cost = Cost(ind(1));
    Worst_X = X(ind(end), :);
    Worst_Cost = Cost(ind(end));
    I = randi([2, 5]);
    Better_X = X(ind(I), :);
    Better_Cost = Cost(ind(I));
    Curve = zeros(1, Max_iter);
    for it = 1:Max_iter
        alpha = 2 * exp(-4 * (it / Max_iter));
        M_Best = Best_Cost;
        M_Better = Better_Cost;
        M_Worst = Worst_Cost;
        for i = 1:N_init
            del = 2 * rand * alpha - alpha;
            sigm = 2 * rand * alpha - alpha;
            A1 = randperm(N_init);
            A1(A1 == i) = [];
            a = A1(1); b = A1(2); c = A1(3);
            epsi = 1e-25 * rand;
            omg = max([M(a), M(b), M(c)]);
            MM = [M(a)-M(b), M(a)-M(c), M(b)-M(c)];
            W(1) = cos(MM(1) + pi) * exp(-MM(1)/omg);
            W(2) = cos(MM(2) + pi) * exp(-MM(2)/omg);
            W(3) = cos(MM(3) + pi) * exp(-MM(3)/omg);
            Wt = sum(W);
            WM1 = del * (W(1)*(X(a,:)-X(b,:)) + W(2)*(X(a,:)-X(c,:)) + W(3)*(X(b,:)-X(c,:))) / (Wt + 1) + epsi;
            omg = max([M_Best, M_Better, M_Worst]);
            MM = [M_Best-M_Better, M_Best-M_Better, M_Better-M_Worst];
            W(1) = cos(MM(1) + pi) * exp(-MM(1)/omg);
            W(2) = cos(MM(2) + pi) * exp(-MM(2)/omg);
            W(3) = cos(MM(3) + pi) * exp(-MM(3)/omg);
            Wt = sum(W);
            WM2 = del * (W(1)*(Best_X-Better_X) + W(2)*(Best_X-Worst_X) + W(3)*(Better_X-Worst_X)) / (Wt + 1) + epsi;
            r = unifrnd(0.1, 0.5);
            MeanRule = r * WM1 + (1 - r) * WM2;
            if rand < 0.5
                z1 = X(i,:) + sigm * (rand * MeanRule) + randn * (Best_X - X(a,:)) / (M_Best - M(a) + 1);
                z2 = Best_X + sigm * (rand * MeanRule) + randn * (X(a,:) - X(b,:)) / (M(a) - M(b) + 1);
            else
                z1 = X(a,:) + sigm * (rand * MeanRule) + randn * (X(b,:) - X(c,:)) / (M(b) - M(c) + 1);
                z2 = Better_X + sigm * (rand * MeanRule) + randn * (X(a,:) - X(b,:)) / (M(a) - M(b) + 1);
            end
            u = zeros(1, Dim);
            for j = 1:Dim
                mu = 0.05 * randn;
                if rand < 0.5
                    if rand < 0.5
                        u(j) = z1(j) + mu * abs(z1(j) - z2(j));
                    else
                        u(j) = z2(j) + mu * abs(z1(j) - z2(j));
                    end
                else
                    u(j) = X(i, j);
                end
            end
            if rand < 0.5
                L = rand < 0.5;
                v1 = (1-L) * 2 * rand + L;
                v2 = rand * L + (1-L);
                Xavg = (X(a,:) + X(b,:) + X(c,:)) / 3;
                phi = rand;
                Xrnd = phi * Xavg + (1-phi) * (phi * Better_X + (1-phi) * Best_X);
                Randn = L * randn(1, Dim) + (1-L) * randn;
                if rand < 0.5
                    u = Best_X + Randn .* (MeanRule + randn * (Best_X - X(a,:)));
                else
                    u = Xrnd + Randn .* (MeanRule + randn * (v1 * Best_X - v2 * Xrnd));
                end
            end
            New_X = max(min(u, ub), lb);
            New_Cost = fobj(New_X);
            if New_Cost < Cost(i)
                X(i, :) = New_X;
                Cost(i) = New_Cost;
                M(i) = Cost(i);
                if Cost(i) < Best_Cost
                    Best_X = X(i, :);
                    Best_Cost = Cost(i);
                end
            end
        end
        [~, ind] = sort(Cost);
        Worst_X = X(ind(end), :);
        Worst_Cost = Cost(ind(end));
        I = randi([2, 5]);
        Better_X = X(ind(I), :);
        Better_Cost = Cost(ind(I));
        Curve(it) = Best_Cost;
    end
    Best_F = Best_Cost;
    Best_X = Best_X;
end
function [Best_F, Best_X, Curve] = run_DBO(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)'; ub = UB(:)';
    end
    pop = N;
    M = Max_iter;
    dim = Dim;
    P_percent = 0.2;
    pNum = round(pop * P_percent);
    x = zeros(pop, dim);
    fit = zeros(pop, 1);
    for i = 1 : pop
        x(i, :) = lb + (ub - lb) .* rand(1, dim);
        fit(i) = fobj(x(i, :));
    end
    pFit = fit;
    pX = x;
    XX = pX;
    [fMin, bestI] = min(fit);
    bestX = x(bestI, :);
    Convergence_curve = zeros(1, M);
    for t = 1 : M
        [~, B] = max(fit);
        worse = x(B, :);
        r2 = rand(1);
        for i = 1 : pNum
            if r2 < 0.9
                r1 = rand(1);
                a = rand(1,1);
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
            fit(i) = fobj(x(i, :));
        end
        [fMMin, bestII] = min(fit);
        bestXX = x(bestII, :);
        R = 1 - t / M;
        Xnew1 = bestXX .* (1 - R);
        Xnew2 = bestXX .* (1 + R);
        Xnew1 = Bounds(Xnew1, lb, ub);
        Xnew2 = Bounds(Xnew2, lb, ub);
        Xnew11 = bestX .* (1 - R);
        Xnew22 = bestX .* (1 + R);
        Xnew11 = Bounds(Xnew11, lb, ub);
        Xnew22 = Bounds(Xnew22, lb, ub);
        for i = (pNum + 1) : 12
            x(i, :) = bestXX + (rand(1, dim) .* (pX(i, :) - Xnew1)) + ...
                            (rand(1, dim) .* (pX(i, :) - Xnew2));
            x(i, :) = Bounds(x(i, :), Xnew1, Xnew2);
            fit(i) = fobj(x(i, :));
        end
        for i = 13 : 19
            x(i, :) = pX(i, :) + randn(1) .* (pX(i, :) - Xnew11) + ...
                               rand(1, dim) .* (pX(i, :) - Xnew22);
            x(i, :) = Bounds(x(i, :), lb, ub);
            fit(i) = fobj(x(i, :));
        end
        for j = 20 : pop
            x(j, :) = bestX + randn(1, dim) .* (abs(pX(j, :) - bestXX) + abs(pX(j, :) - bestX)) / 2;
            x(j, :) = Bounds(x(j, :), lb, ub);
            fit(j) = fobj(x(j, :));
        end
        XX = pX;
        for i = 1 : pop
            if fit(i) < pFit(i)
                pFit(i) = fit(i);
                pX(i, :) = x(i, :);
            end
            if pFit(i) < fMin
                fMin = pFit(i);
                bestX = pX(i, :);
            end
        end
        Convergence_curve(t) = fMin;
    end
    Best_F = fMin;
    Best_X = bestX;
    Curve = Convergence_curve(:)';
    function s = Bounds(s, Lb, Ub)
        temp = s;
        I = temp < Lb; temp(I) = Lb(I);
        J = temp > Ub; temp(J) = Ub(J);
        s = temp;
    end
end
function [Best_F, Best_X, Curve] = run_ETO(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)'; ub = UB(:)';
    end
    Destination_position = zeros(1, Dim);
    Destination_fitness = inf;
    Destination_position_second = zeros(1, Dim);
    Curve = zeros(1, Max_iter);
    Position_sort = zeros(N, Dim);
    b = 1.55;
    CE = floor(1 + (Max_iter / b));
    T = floor(1.2 + Max_iter / 2.25);
    CEi = 0;
    CEi_temp = 0;
    UB_2 = ub;
    LB_2 = lb;
    X = rand(N, Dim) .* (ub - lb) + lb;
    Objective_values = zeros(1, N);
    for i = 1 : N
        Objective_values(i) = fobj(X(i, :));
        if Objective_values(i) < Destination_fitness
            Destination_fitness = Objective_values(i);
            Destination_position = X(i, :);
        end
    end
    Curve(1) = Destination_fitness;
    t = 2;
    while t <= Max_iter
        for i = 1 : N
            for j = 1 : Dim
                d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * (1 - t / Max_iter));
                d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * (1 - t / Max_iter));
                CM = (sqrt(t / Max_iter)^tan(d1 / (d2))) * rand() * 0.01;
                if t == CEi
                    UB_2 = Destination_position(j) + (1 - t / Max_iter) * abs(rand() * Destination_position(j) - Destination_position_second(j)) * rand();
                    LB_2 = Destination_position(j) - (1 - t / Max_iter) * abs(rand() * Destination_position(j) - Destination_position_second(j)) * rand();
                    if UB_2 > ub(j), UB_2 = ub(j); end
                    if LB_2 < lb(j), LB_2 = lb(j); end
                    X = rand(N, Dim) .* (UB_2 - LB_2) + LB_2;
                    CEi_temp = CEi;
                    CEi = 0;
                end
                if t <= T
                    q1 = rand(); q3 = rand(); q4 = rand(); q5 = rand();
                    if CM > 1
                        d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * q1);
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * q1);
                        alpha_1 = q5 * 3 * (t / Max_iter - 0.85) * exp(abs(d1 / d2) - 1);
                        if q1 <= 0.5
                            X(i, j) = Destination_position(j) + q5 * alpha_1 * abs(Destination_position(j) - X(i, j));
                        else
                            X(i, j) = Destination_position(j) - q5 * alpha_1 * abs(Destination_position(j) - X(i, j));
                        end
                    else
                        d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * q3);
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * q3);
                        alpha_3 = rand() * 3 * (t / Max_iter - 0.85) * exp(abs(d1 / d2) - 1.3);
                        if q3 <= 0.5
                            X(i, j) = Destination_position(j) + q4 * alpha_3 * abs(q5 * Destination_position(j) - X(i, j));
                        else
                            X(i, j) = Destination_position(j) - q4 * alpha_3 * abs(q5 * Destination_position(j) - X(i, j));
                        end
                    end
                else
                    q2 = rand(); q6 = rand();
                    alpha_2 = q6 * exp(tanh(1.5 * (-t / Max_iter - 0.75) - q6));
                    if CM < 1
                        d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * q2);
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_iter * q2);
                        X(i, j) = X(i, j) + exp(tan(abs(d1 / d2)) * abs(q6 * alpha_2 * Destination_position(j) - X(i, j)));
                    else
                        if q2 <= 0.5
                            X(i, j) = X(i, j) + 3 * (abs(q6 * alpha_2 * Destination_position(j) - X(i, j)));
                        else
                            X(i, j) = X(i, j) - 3 * (abs(q6 * alpha_2 * Destination_position(j) - X(i, j)));
                        end
                    end
                end
            end
            CEi = CEi_temp;
        end
        for i = 1 : N
            Flag4ub = X(i, :) > UB_2;
            Flag4lb = X(i, :) < LB_2;
            X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + (UB_2 + LB_2) / 2 .* Flag4ub + LB_2 .* Flag4lb;
            Objective_values(i) = fobj(X(i, :));
            if Objective_values(i) < Destination_fitness
                Destination_fitness = Objective_values(i);
                Destination_position = X(i, :);
            end
        end
        if t == CE
            CEi = CE + 1;
            CE = CE + floor(2 - t * 2 / (Max_iter - CE * 4.6) / 1);
            [~, sort_idx] = sort(Objective_values);
            Position_sort = X(sort_idx, :);
            Destination_position_second = Position_sort(2, :);
        end
        Curve(t) = Destination_fitness;
        t = t + 1;
    end
    Best_F = Destination_fitness;
    Best_X = Destination_position;
end
function [Best_F, Best_X, Curve] = run_NRBO(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)';
        ub = UB(:)';
    end
    [Best_Score, Best_Pos, CG_curve] = NRBO_core(N, Max_iter, lb, ub, Dim, fobj);
    Best_F = Best_Score;
    Best_X = Best_Pos;
    Curve = CG_curve(:)';  % 转为行向量
    function [Best_Score, Best_Pos, CG_curve] = NRBO_core(N, MaxIt, LB, UB, dim, fobj)
        DF = 0.6;
        Position = initialization(N, dim, UB, LB);
        Fitness = zeros(N, 1);
        for i = 1:N
            Fitness(i) = fobj(Position(i,:));
        end
        [~, Ind] = sort(Fitness);
        Best_Score = Fitness(Ind(1));
        Best_Pos = Position(Ind(1),:);
        Worst_Cost = Fitness(Ind(end));
        Worst_Pos = Position(Ind(end),:);
        CG_curve = zeros(1, MaxIt);
        for it = 1:MaxIt
            delta = (1 - ((2 * it) / MaxIt)) .^ 5;
            for i = 1:N
                P1 = randperm(N, 2);
                a1 = P1(1); a2 = P1(2);
                rho = rand * (Best_Pos - Position(i,:)) + rand * (Position(a1,:) - Position(a2,:));
                Flag = 1;
                NRSR = SearchRule(Best_Pos, Worst_Pos, Position(i,:), rho, Flag);
                X1 = Position(i,:) - NRSR + rho;
                X2 = Best_Pos - NRSR + rho;
                Xupdate = zeros(1, dim);
                for j = 1:dim
                    X3 = Position(i,j) - delta * (X2(j) - X1(j));
                    a1 = rand; a2 = rand;
                    Xupdate(j) = a1 * (a1 * X1(j) + (1 - a2) * X2(j)) + (1 - a2) * X3;
                end
                if rand < DF
                    theta1 = -1 + 2 * rand(); theta2 = -0.5 + rand();
                    beta = rand < 0.5;
                    u1 = beta * 3 * rand + (1 - beta); u2 = beta * rand + (1 - beta);
                    if u1 < 0.5
                        X_TAO = Xupdate + theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));
                    else
                        X_TAO = Best_Pos + theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));
                    end
                    Xnew = X_TAO;
                else
                    Xnew = Xupdate;
                end
                Xnew = min(max(Xnew, LB), UB);
                Xnew_Cost = fobj(Xnew);
                if Xnew_Cost < Fitness(i)
                    Position(i,:) = Xnew;
                    Fitness(i) = Xnew_Cost;
                    if Fitness(i) < Best_Score
                        Best_Pos = Position(i,:);
                        Best_Score = Fitness(i);
                    end
                end
                if Fitness(i) > Worst_Cost
                    Worst_Pos = Position(i,:);
                    Worst_Cost = Fitness(i);
                end
            end
            CG_curve(it) = Best_Score;
        end
    end
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
    function NRSR = SearchRule(Best_Pos, Worst_Pos, Position, rho, Flag)
        dim = size(Position, 2);
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
function [Best_F, Best_X, Curve] = run_PSA(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        lb = LB * ones(1, Dim);
        ub = UB * ones(1, Dim);
    else
        lb = LB(:)';
        ub = UB(:)';
    end
    nvars = Dim;
    T = Max_iter;
    x = lb + (ub - lb) .* rand(N, nvars);
    f = zeros(N, 1);
    for i = 1:N
        f(i) = fobj(x(i,:));
    end
    TargetF = min(f);
    TargetX = x(find(f == TargetF, 1), :);
    newTargetX = TargetX;
    newTargetF = TargetF;
    ConvergenceCurve = zeros(1, T);
    t = 1;
    Kp = 1; Ki = 0.5; Kd = 1.2;
    LogT = log(T);
    while t <= T
        if t == 1
            [TargetF, index] = min(f);
            TargetX = x(index,:);
            newTargetX = TargetX;
            newTargetF = TargetF;
        else
            for i = 1:N
                f(i) = fobj(x(i,:));
            end
            if min(f) < TargetF
                [newTargetF, index] = min(f);
                newTargetX = x(index,:);
            end
        end
        ConvergenceCurve(t) = newTargetF;
        if t == 1
            Ek = TargetX - x;
            Ek_1 = Ek;
            Ek_2 = Ek;
        else
            Ek_2 = Ek_1;
            Ek_1 = Ek + newTargetX - TargetX;
            Ek = newTargetX - x;
            TargetX = newTargetX;
            TargetF = newTargetF;
        end
        a = (log(T - t + 2) / LogT)^2;
        out0 = (cos(1 - t/T) + a * rand(N, nvars) .* PSA_levyFlight(N, nvars)) .* Ek;
        pid = rand(N,1) * Kp .* (Ek - Ek_1) + rand(N,1) * Ki .* Ek + rand(N,1) * Kd .* (Ek - 2*Ek_1 + Ek_2);
        r = rand(N,1) * cos(t/T);
        x = x + r .* pid + (1 - r) .* out0;
        lbExtended = repmat(lb, [N, 1]);
        ubExtended = repmat(ub, [N, 1]);
        lbViolated = x < lbExtended;
        ubViolated = x > ubExtended;
        x(lbViolated) = lbExtended(lbViolated);
        x(ubViolated) = ubExtended(ubViolated);
        t = t + 1;
    end
    Best_F = newTargetF;
    Best_X = newTargetX;
    Curve = ConvergenceCurve;
    function o = PSA_levyFlight(n, d)
        beta = 1.5;
        sigma = (gamma(1+beta) * sin(pi*beta/2) / (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);
        u = randn(n, d) * sigma;
        v = randn(n, d);
        step = u ./ abs(v).^(1/beta);
        o = step;
    end
end
