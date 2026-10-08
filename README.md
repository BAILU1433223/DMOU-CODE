# DMOU: Dual-Mode Ornstein-Uhlenbeck Optimizer

Reference MATLAB implementation of the DMOU metaheuristic and of the benchmark
material used in the paper *"Dual-Mode Ornstein-Uhlenbeck (DMOU) Optimizer Based
on Stochastic Differential Equations"*.

## Contents

| Path | Description |
| --- | --- |
| `DMOU.m` | The proposed algorithm, `[Best_F, Best_X, Hist] = DMOU(N, Max_iter, LB, UB, Dim, fobj)`. |
| `Baseline_Algos_Library.m` | The nine baseline algorithms: PSO, GWO, SCA, MVO, AO, INFO, DBO, DP, NRBO. `[best_f, best_x, curve] = Baseline_Algos_Library(algo_name, N, Max_iter, LB, UB, Dim, fobj)`. |
| `CEC_Single_Func_Curve.m` | Demo driver: runs DMOU and the baselines on one CEC function and plots the convergence curves. |
| `CEC2017/` | Official CEC2017 test suite (C++ source, prebuilt Windows mex file, `input_data`). |
| `Ablation/` | The five ablation variants of the algorithm (see below). |
| `Engineering_Problems/` | The six constrained engineering design problems and the constrained version of DMOU. |

## Requirements

MATLAB (developed and tested with R2025b on Windows x64). No toolbox is required.

The prebuilt Windows mex file `cec17_func.mexw64` is included. To rebuild it on
another platform, run inside `CEC2017/`:

```matlab
mex cec17_func.cpp -DWINDOWS
```

The CEC mex function reads its shift and rotation data from `CEC2017/input_data`,
so that folder must be the current directory when it is called;
`CEC_Single_Func_Curve.m` takes care of this automatically.

## Main algorithm

```matlab
[Best_F, Best_X, Hist] = DMOU(N, Max_iter, LB, UB, Dim, fobj)
```

| Argument | Meaning |
| --- | --- |
| `N` | population size |
| `Max_iter` | maximum number of iterations |
| `LB`, `UB` | lower / upper bounds (scalar or `1 x Dim` vector) |
| `Dim` | problem dimension |
| `fobj` | objective function handle, evaluated on a `1 x Dim` row vector |
| `Best_F` | best fitness value found |
| `Best_X` | best solution (`1 x Dim` row vector) |
| `Hist` | convergence history, `1 x Max_iter`, best fitness per iteration |

## Ablation variants

`Ablation/` contains the variants compared in the ablation study of the paper.
Each file is a drop-in replacement for `DMOU.m` with the same input and output
arguments:

| File | Variant |
| --- | --- |
| `DMOU_Full.m` | complete algorithm |
| `DMOU_NoMacro.m` | macro state removed (`p_macro = 0`, micro search only) |
| `DMOU_NoMicro.m` | micro state removed (`p_macro = 1`, macro search only) |
| `DMOU_NoAdapt.m` | adaptive parameters frozen at fixed initial values |
| `DMOU_NoDrift.m` | drift term removed from the macro state |

## Engineering problems

`Engineering_Problems/` holds one self-contained file per problem. Each file
returns the objective value, the inequality constraints and the equality
constraints, and calling it without arguments returns a small structure with the
dimension, the constraint counts and the bounds.

```matlab
info = EngProb1_ProcessSynthesis();          % n, ng, nh, lb, ub
[f, g, h] = EngProb1_ProcessSynthesis(x);    % x is 1 x n or ps x n
```

| File | Problem | `n` | `ng` | `nh` |
| --- | --- | --- | --- | --- |
| `EngProb1_ProcessSynthesis.m` | process synthesis | 7 | 9 | 0 |
| `EngProb2_PlanetaryGearTrain.m` | planetary gear train design | 9 | 10 | 1 |
| `EngProb3_StepConePulley.m` | step-cone pulley | 5 | 8 | 3 |
| `EngProb4_PlateFinHeatExchanger.m` | plate-fin heat exchanger | 7 | 0 | 1 |
| `EngProb5_Truss200.m` | 200-bar planar truss | 29 | 200 | 0 |
| `EngProb6_HanoiNetwork.m` | Hanoi water distribution network | 34 | 31 | 0 |

`DMOU_Engineering.m` is the constrained version of the algorithm used for these
problems. It keeps the best feasible solution with a tolerance-based comparison
and returns the constraint violation of the reported solution:

```matlab
info = EngProb1_ProcessSynthesis();
[Best_F, Best_X, Best_Viol, Curve] = DMOU_Engineering(50, 3000, info.lb, info.ub, ...
    info.n, @EngProb1_ProcessSynthesis, 1e-6);
```

Feasibility convention: `g <= 0` and `h == 0`.

## Experimental protocol reported in the paper

- Function-evaluation budget: `10000 * Dim` per run.
- Population size: `N = 50` at `Dim = 10` and `30`; `N = 100` at `Dim = 50` and `100`.
- 51 independent runs per benchmark function; the mean of the final fitness
  values is the performance indicator used for ranking.
- All random draws come from the global stream, so results are reproducible for a
  fixed seed (`rng(1, 'twister')`).

## Third-party material

`CEC2017/` contains the official CEC2017 benchmark suite distributed by the
competition organizers, including its `input_data` files. It is redistributed here
only so that the experiments of the paper can be reproduced; please refer to the
original distribution for its terms of use.

## Citation

If you use this code, please cite the paper:

> Dual-Mode Ornstein-Uhlenbeck (DMOU) Optimizer Based on Stochastic Differential Equations.

DOI to be added after publication.
