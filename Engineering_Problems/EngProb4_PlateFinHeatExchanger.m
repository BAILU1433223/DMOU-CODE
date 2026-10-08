function [f, g, h] = EngProb4_PlateFinHeatExchanger(x)
if nargin == 0
    f = struct('n', 7, 'ng', 0, 'nh', 1, 'lb', [0.1,0.1,0.002,100,1e-4,0.001,1], 'ub', [1,1,0.010,1000,2e-4,0.010,1200]);
    g = []; h = [];
    return
end
ps = size(x,1);
        Lh = x(:,1);
        Lc = x(:,2);
        H  = x(:,3);
        n  = x(:,4);
        t  = x(:,5);
        l  = x(:,6);
        Nh = round(x(:,7));
        Nh = max(1, min(1200, Nh));
        Nc = Nh + 1;
        m_h    = 0.8962;    T_h_in = 513;     P_h_in = 100000;
        Cp_h   = 1017.7;    rho_h = 0.8196;   mu_h   = 2.410e-5;
        Pr_h   = 0.6878;
        R_cte_h = P_h_in / (rho_h * T_h_in);
        m_c    = 0.8296;    T_c_in = 277;     P_c_in = 100000;
        Cp_c   = 1011.8;    rho_c = 0.9385;   mu_c   = 2.182e-5;
        Pr_c   = 0.6954;
        R_cte_c = P_c_in / (rho_c * T_c_in);
        Q_c    = 160000;
        s = 1 ./ n - t;
        D_H = 4 .* s .* (H - t) .* l ./ (2 .* (s.*l + l.*(H - t) + t.*(H - t)) + t.*s);
        A_ff_h = (H - t) .* (1 - n.*t) .* Lc .* Nh;
        A_ff_c = (H - t) .* (1 - n.*t) .* Lh .* Nc;
        A_h = Lh .* Lc .* Nh .* (1 + 2.*n.*(H - t));
        A_c = Lh .* Lc .* Nc .* (1 + 2.*n.*(H - t));
        Re_h = m_h .* D_H ./ (A_ff_h .* mu_h);
        Re_c = m_c .* D_H ./ (A_ff_c .* mu_c);
        j_h = zeros(ps,1); j_c = zeros(ps,1);
        idx_h = Re_h > 1500;  idx_c = Re_c > 1500;
        j_h(idx_h) = 0.21 .* Re_h(idx_h).^(-0.4) .* (l(idx_h)./D_H(idx_h)).^(-0.24) .* (s(idx_h)./H(idx_h) - t(idx_h)).^(-0.02);
        j_h(~idx_h) = 0.53 .* Re_h(~idx_h).^(-0.5) .* (l(~idx_h)./D_H(~idx_h)).^(-0.15) .* (s(~idx_h)./H(~idx_h) - t(~idx_h)).^(-0.14);
        j_c(idx_c) = 0.21 .* Re_c(idx_c).^(-0.4) .* (l(idx_c)./D_H(idx_c)).^(-0.24) .* (s(idx_c)./H(idx_c) - t(idx_c)).^(-0.02);
        j_c(~idx_c) = 0.53 .* Re_c(~idx_c).^(-0.5) .* (l(~idx_c)./D_H(~idx_c)).^(-0.15) .* (s(~idx_c)./H(~idx_c) - t(~idx_c)).^(-0.14);
        h_h = j_h .* Cp_h .* Pr_h.^(-2/3) .* m_h ./ A_ff_h;
        h_c = j_c .* Cp_c .* Pr_c.^(-2/3) .* m_c ./ A_ff_c;
        C_h = m_h * Cp_h;  C_c = m_c * Cp_c;
        C_min = min(C_h, C_c);  C_max = max(C_h, C_c);
        C_r = C_min / C_max;
        UA = 1 ./ (1./(h_h.*A_h) + 1./(h_c.*A_c));
        NTU = UA ./ C_min;
        epsilon = 1 - exp( (1./C_r) .* NTU.^0.22 .* ( exp(-C_r .* NTU.^0.78) - 1 ) );
        Q = epsilon .* C_min .* (T_h_in - T_c_in);
        f_h = zeros(ps,1); f_c = zeros(ps,1);
        f_h(idx_h) = 1.12 .* Re_h(idx_h).^(-0.36) .* (l(idx_h)./D_H(idx_h)).^(-0.65) .* (s(idx_h)./H(idx_h) - t(idx_h)).^(-0.17);
        f_h(~idx_h) = 8.12 .* Re_h(~idx_h).^(-0.74) .* (l(~idx_h)./D_H(~idx_h)).^(-0.41) .* (s(~idx_h)./H(~idx_h) - t(~idx_h)).^(-0.02);
        f_c(idx_c) = 1.12 .* Re_c(idx_c).^(-0.36) .* (l(idx_c)./D_H(idx_c)).^(-0.65) .* (s(idx_c)./H(idx_c) - t(idx_c)).^(-0.17);
        f_c(~idx_c) = 8.12 .* Re_c(~idx_c).^(-0.74) .* (l(~idx_c)./D_H(~idx_c)).^(-0.41) .* (s(~idx_c)./H(~idx_c) - t(~idx_c)).^(-0.02);
        Delta_P_h = 2 .* f_h .* Lh .* (m_h ./ A_ff_h).^2 ./ (rho_h .* D_H);
        Delta_P_c = 2 .* f_c .* Lc .* (m_c ./ A_ff_c).^2 ./ (rho_c .* D_H);
        N_s = (1 - epsilon) .* ((T_c_in - T_h_in).^2 ./ (T_c_in .* T_h_in)) + ...
              (R_cte_h ./ Cp_h) .* (Delta_P_h ./ P_h_in) + ...
              (R_cte_c ./ Cp_c) .* (Delta_P_c ./ P_c_in);
        f = N_s;
        g = zeros(ps, 0);
        h = Q - Q_c;
end
