function [f, g, h] = EngProb6_HanoiNetwork(x)
if nargin == 0
    f = struct('n', 34, 'ng', 31, 'nh', 0, 'lb', ones(1,34), 'ub', 6*ones(1,34));
    g = []; h = [];
    return
end
ps = size(x,1);
f = zeros(ps,1); g = zeros(ps,31); h = zeros(ps,0);
for i = 1:ps
    [f(i,1), g(i,:)] = Hanoi_Cost(x(i,:));
end
end

function [f, g] = Hanoi_Cost(x)
persistent D
if isempty(D)
    D = Hanoi_Data();
end
x = x(:).';
idx = round(x);
idx = max(1, min(6, idx));
DiamIn = D.diam_in(idx);
f = sum(D.cost_per_m(idx) .* D.pipe(:,3).');
H = hanoi_heads(D, DiamIn);
g = D.min_head - H(2:end);
end
function H = hanoi_heads(D, DiamIn)
C   = D.C;
L   = double(D.pipe(:,3));
u   = double(D.pipe(:,1));
v   = double(D.pipe(:,2));
np  = D.npipe;
njn = D.nnode - 1;
res = D.res_node;
d   = D.demand_m3h(2:end).' / 3600;  % m3/s (junctions only)
nn  = 1.852;
Dm  = DiamIn(:) * 0.0254;
r   = 10.67 .* L ./ (C^nn .* Dm.^4.871);
A12 = zeros(njn, np);
A10 = zeros(1, np);
for k = 1:np
    a = u(k); b = v(k);
    if a == res && b > res
        A12(b-1, k) = 1;
        A10(k) = -1;
    elseif a > res && b == res
        A12(a-1, k) = -1;
        A10(k) = 1;
    else
        A12(a-1, k) = -1;
        A12(b-1, k) = 1;
    end
end
H0 = D.res_head;
Q  = 1e-3 * ones(np, 1);
tol = 1e-8;
maxit = 500;
for it = 1:maxit
    p = r .* max(abs(Q), 1e-12).^(nn - 1);
    invp = 1 ./ p;
    LHS = A12 * (invp .* A12');
    LHS = LHS + 1e-9 * eye(njn);
    RHS = -d - A12 * (invp .* (A10' * H0));
    H = LHS \ RHS;
    Qnew = -invp .* (A12' * H + A10' * H0);
    if max(abs(Qnew - Q)) < 1e-8 || max(abs(A12 * Qnew - d)) < tol
        Q = Qnew;
        break;
    end
    Q = Qnew;
end
H = [H0, H(:)'];               % heads at nodes 1..32
end

function D = Hanoi_Data()
D.npipe = 34;
D.nnode = 32;
D.res_node = 1;
D.res_head = 100;
D.min_head = 30;
D.C = 130;
D.pipe = [
    1   2   100;
    2   3  1350;
    3   4   900;
    4   5  1150;
    5   6  1450;
    6   7   450;
    7   8   850;
    8   9   850;
    9  10   800;
   10  11   950;
   11  12  1200;
   12  13  3500;
   10  14   800;
   14  15   500;
   15  16   550;
   17  16  2730;
   18  17  1750;
   19  18   800;
    3  19   400;
    3  20  2200;
   20  21  1500;
   21  22   500;
   20  23  2650;
   23  24  1230;
   24  25  1300;
   25  26   850;
   26  27   300;
   16  27   750;
   23  28  1500;
   28  29  2000;
   30  29  1600;
   31  30   150;
   32  31   860;
   25  32   950];
D.demand_m3h = [-19940, 890, 850, 130, 725, 1005, 1350, 550, 525, 525, ...
                500, 560, 940, 615, 280, 310, 865, 1345, 60, 1275, ...
                930, 485, 1045, 820, 170, 900, 370, 290, 360, 360, ...
                105, 805];
D.diam_in    = [12, 16, 20, 24, 30, 40];
D.cost_per_m = [45.726, 70.4, 98.378, 129.333, 180.748, 278.28];
D.x = [3891.67, 3896.83, 3896.83, 3071.43, 2706.35, 2309.52, 2055.56, ...
       1611.11, 1214.29, 833.33, -39.68, -1420.63, -4293.65, 1865.08, ...
       2595.24, 3468.25, 3833.33, 3896.83, 3896.83, 5928.57, 7214.29, ...
       8325.40, 5912.70, 5611.11, 5198.41, 4769.84, 4150.79, 7690.48, ...
       7357.14, 7208.33, 6753.97, 6166.67];
D.y = [9616.67, 9031.75, 7904.76, 7873.02, 6761.90, 5714.29, 5142.86, ...
       4111.11, 3206.35, 2269.84, 2238.10, 2238.10, 2269.84, 2190.48, ...
       2063.49, 1841.27, 4174.60, 5746.03, 7206.35, 7920.63, 8000.00, ...
       7904.76, 5539.68, 4047.62, 2253.97, 2031.75, 1857.14, 5412.70, ...
       3761.90, 716.67, 1095.24, 1857.14];
D.best_known = 6081086.97;
end
