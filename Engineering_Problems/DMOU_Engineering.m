function [Best_F, Best_X, Best_Viol, Curve] = DMOU_Engineering(N, Max_iter, LB, UB, Dim, probfun, tol)
if isscalar(LB), LB = LB*ones(1,Dim); UB = UB*ones(1,Dim); end
lb = LB; ub = UB;
X = rand(N, Dim) .* (ub - lb) + lb;
F_orig = zeros(N,1); Viol = zeros(N,1);
for i = 1:N
    [F_orig(i), Viol(i)] = prob_eval(probfun, X(i,:));
end
Best_F = inf; Best_Viol = inf; Best_X = zeros(1,Dim);
for i = 1:N
    if is_better(F_orig(i), Viol(i), Best_F, Best_Viol, tol)
        Best_F = F_orig(i); Best_Viol = Viol(i); Best_X = X(i,:);
    end
end
Theta = 0.5 + 0.5*rand(N,1);
Gamma = 0.2 + 0.8*rand(N,1);
range_len = ub - lb;
base_step = range_len * 0.01;
Sigma = repmat(base_step, N, 1);
Sigma_max = range_len * 0.005;
tau = 0.1;
micro_scale = 0.05;
amp = exp(tau);
shrink = exp(-tau);
Curve = zeros(1, Max_iter);
for t = 1:Max_iter
    decay_ratio = 1 - t/Max_iter;
    dt = decay_ratio^2 + 0.001;
    Phase_Prob = decay_ratio^1.5;
    pop_std = std(X,0,1);
    X_Next = zeros(N, Dim);
    F_Next = zeros(N,1);
    Viol_Next = zeros(N,1);
    Sigma_Next = Sigma;
    for i = 1:N
        V_c = Best_X - X(i,:);
        if rand() < Phase_Prob
            V_e = randn(1,Dim) .* pop_std;
            norm_Ve = norm(V_e) + 1e-20;
            W_isotropic = randn(1,Dim) .* Sigma(i,:);
            W_directional = randn() * (V_e / norm_Ve);
            Diffusion = (W_isotropic + 0.5*W_directional) * sqrt(dt);
            X_Candidate = X(i,:) + Theta(i)*V_c*dt + Gamma(i)*V_e*dt + Diffusion;
        else
            Shrink_Scale = rand(1,Dim) .* (2*decay_ratio);
            Local_Micro_Diff = micro_scale * decay_ratio^2 * randn(1,Dim) .* Sigma(i,:);
            X_Candidate = Best_X - Shrink_Scale.*V_c + Local_Micro_Diff;
        end
        low = X_Candidate < lb;
        high = X_Candidate > ub;
        if any(low)
            X_Candidate(low) = (X(i,low) + lb(low)) / 2;
        end
        if any(high)
            X_Candidate(high) = (X(i,high) + ub(high)) / 2;
        end
        X_Candidate = max(min(X_Candidate, ub), lb);
        [f_cand, v_cand] = prob_eval(probfun, X_Candidate);
        if is_better(f_cand, v_cand, F_orig(i), Viol(i), tol)
            X_Next(i,:) = X_Candidate;
            F_Next(i) = f_cand;
            Viol_Next(i) = v_cand;
            Theta(i) = min(2.0, Theta(i) * amp);
            Sigma_Next(i,:) = min(Sigma_max, Sigma(i,:) * amp);
        else
            X_Next(i,:) = X(i,:);
            F_Next(i) = F_orig(i);
            Viol_Next(i) = Viol(i);
            Theta(i) = max(0.1, Theta(i) * 0.8);
            Gamma(i) = 0.5 + 1.0*rand();
            Sigma_Next(i,:) = max(1e-12, Sigma(i,:) * shrink);
        end
    end
    X = X_Next;
    F_orig = F_Next;
    Viol = Viol_Next;
    Sigma = Sigma_Next;
    for i = 1:N
        if is_better(F_orig(i), Viol(i), Best_F, Best_Viol, tol)
            Best_F = F_orig(i); Best_Viol = Viol(i); Best_X = X(i,:);
        end
    end
    Curve(t) = Best_F;
end
end

function better = is_better(f1, v1, f2, v2, tol)
if v1 <= tol && v2 <= tol
    better = f1 < f2;
elseif v1 <= tol && v2 > tol
    better = true;
elseif v1 > tol && v2 > tol
    better = v1 < v2;
else
    better = false;
end
end

function [f, viol] = prob_eval(probfun, x)
[f, g, h] = probfun(x);
viol = sum(max(0, g(:))) + sum(abs(h(:)));
end
