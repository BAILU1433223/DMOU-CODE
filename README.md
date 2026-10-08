# DMOU: A Dual-Mode Ornstein-Uhlenbeck Optimizer

Reference MATLAB implementation of the DMOU metaheuristic proposed in the paper
*"DMOU: A Dual-Mode Ornstein-Uhlenbeck Optimizer Based on Stochastic Differential Equations"*,
together with the nine baseline algorithms and the CEC2017 / CEC2022 benchmark suites used in
the comparative experiments.

## Contents

| Path | Description |
| --- | --- |
| `DMOU.m` | The proposed algorithm. `[Best_F, Best_X, Hist] = DMOU(N, Max_iter, LB, UB, Dim, fobj)` |
| `Baseline_Algos_Library.m` | Dispatcher for the nine baseline algorithms: PSO, GWO, SCA, MVO, AO, INFO, DBO, DP, NRBO. `[best_f, best_x, curve] = Baseline_Algos_Library(algo_name, N, Max_iter, LB, UB, Dim, fobj)` |
| `CEC_Single_Func_Curve.m` | Demo driver: runs DMOU and the baselines on one CEC function and plots the convergence curves (nothing is written to disk). |
| `CEC2017/` | Official CEC2017 test suite: C++ source, prebuilt Windows mex file, and `input_data`. |
| `CEC2022/` | Official CEC2022 test suite: C++ source, prebuilt Windows mex file, and `input_data`. |

## Requirements

MATLAB (developed and tested with R2025b on Windows x64). No toolbox is required.

Prebuilt Windows mex files (`cec17_func.mexw64`, `cec22_test_func.mexw64`) are included. To
rebuild them on another platform, run from inside the corresponding suite folder:

```matlab
mex cec17_func.cpp -DWINDOWS        % CEC2017
mex cec22_test_func.cpp -DWINDOWS   % CEC2022
```

## Quick start

Option 1 - run the ready-made demo (DMOU plus the nine baselines on one CEC function):

```matlab
CEC_Single_Func_Curve   % edit the "User Parameters" block to select suite / function / Dim / N / iterations
```

Option 2 - call DMOU directly on any objective that accepts a `1 x Dim` row vector:

```matlab
addpath(pwd);
fobj = @(x) sum(x.^2);
rng(1, 'twister');
[Best_F, Best_X, Hist] = DMOU(50, 3000, -100, 100, 20, fobj);
```

The CEC mex functions read their shift and rotation data from the `input_data` folder of the
corresponding suite, so that suite folder must be the current directory when they are called.
`CEC_Single_Func_Curve.m` takes care of this automatically.

## Interface of the main algorithm

```matlab
[Best_F, Best_X, Hist] = DMOU(N, Max_iter, LB, UB, Dim, fobj)
```

| Argument | Meaning |
| --- | --- |
| `N` | population size |
| `Max_iter` | number of iterations |
| `LB`, `UB` | lower / upper bounds (scalar or `1 x Dim` vector) |
| `Dim` | problem dimension |
| `fobj` | objective function handle, evaluated on a `1 x Dim` row vector |
| `Best_F` | best fitness value found |
| `Best_X` | best solution (`1 x Dim` row vector) |
| `Hist` | convergence history, `1 x Max_iter`, best fitness per iteration |

## Experimental protocol reported in the paper

- Function-evaluation budget: `10000 * Dim` per run.
- Population size: `N = 50` at `Dim = 10` and `30`; `N = 100` at `Dim = 50` and `100`.
- 51 independent runs per benchmark function; the mean of the final fitness values is the
  performance indicator used for ranking.
- All random draws come from MATLAB's global stream, so results are reproducible for a fixed
  seed (see the demo script).

## Third-party material

`CEC2017/` and `CEC2022/` contain the official CEC benchmark suites distributed by the
competition organizers (CEC2017 suite by Noor Awad, Nanyang Technological University; CEC2022
suite), including their `input_data` files. They are redistributed here only so that the
experiments in the paper can be reproduced; please refer to the original distribution for their
terms of use. The C++ code is unchanged - only two comment lines were edited for text-encoding
consistency.

## Citation

If you use this code, please cite the paper:

> DMOU: A Dual-Mode Ornstein-Uhlenbeck Optimizer Based on Stochastic Differential Equations.

DOI to be added after publication.
