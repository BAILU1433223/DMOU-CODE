function [Best_F, Best_X, Hist] = DMOU_NoMicro(N, Max_iter, LB, UB, Dim, fobj)
    if isscalar(LB)
        LB = LB * ones(1, Dim);
        UB = UB * ones(1, Dim);
    end
    X = rand(N, Dim) .* (UB - LB) + LB;
    Fits = arrayfun(@(i) fobj(X(i, :)), 1:N)';
    [Best_F, b_idx] = min(Fits);
    Best_X = X(b_idx, :);
    Hist = zeros(1, Max_iter);
    Theta = 0.5 + 0.5 * rand(N, 1);
    Gamma = 0.2 + 0.8 * rand(N, 1);
    range_len = UB - LB;
    base_step = range_len * 0.01;
    Sigma = repmat(base_step, N, 1);
    tau = 0.1;
    Sigma_max = range_len * 0.005;
    for t = 1:Max_iter
        [current_min, current_b_idx] = min(Fits);
        if current_min < Best_F
            Best_F = current_min;
            Best_X = X(current_b_idx, :);
        end
        X_Next = zeros(N, Dim);
        Fits_Next = zeros(N, 1);
        Sigma_Next = Sigma;
        decay_ratio = 1 - t / Max_iter;
        dt = decay_ratio^2 + 0.001;
        amp_scale = exp(tau);
        shrink_scale = exp(-tau);
        for i = 1:N
            V_c = Best_X - X(i, :);
            pop_std = std(X, 0, 1);
            V_e = randn(1, Dim) .* pop_std;
            norm_Ve = norm(V_e) + 1e-20;
            W_isotropic = randn(1, Dim) .* Sigma(i, :);
            W_directional = randn() * (V_e / norm_Ve);
            Diffusion = (W_isotropic + 0.5 * W_directional) * sqrt(dt);
            X_Candidate = X(i, :) + Theta(i) * V_c * dt + Gamma(i) * V_e * dt + Diffusion;
            low_idx = X_Candidate < LB;
            high_idx = X_Candidate > UB;
            if any(low_idx)
                X_Candidate(low_idx) = (X(i, low_idx) + LB(low_idx)) / 2;
            end
            if any(high_idx)
                X_Candidate(high_idx) = (X(i, high_idx) + UB(high_idx)) / 2;
            end
            X_Candidate = max(min(X_Candidate, UB), LB);
            F_Candidate = fobj(X_Candidate);
            if F_Candidate < Fits(i)
                X_Next(i, :) = X_Candidate;
                Fits_Next(i) = F_Candidate;
                Theta(i) = min(2.0, Theta(i) * amp_scale);
                Sigma_Next(i, :) = min(Sigma_max, Sigma(i, :) * amp_scale);
            else
                X_Next(i, :) = X(i, :);
                Fits_Next(i) = Fits(i);
                Theta(i) = max(0.1, Theta(i) * 0.8);
                Gamma(i) = 0.5 + rand();
                Sigma_Next(i, :) = max(1e-12, Sigma(i, :) * shrink_scale);
            end
        end
        X = X_Next;
        Fits = Fits_Next;
        Sigma = Sigma_Next;
        Hist(t) = Best_F;
    end
end
