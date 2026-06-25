<p align="center">
  <img src="docs/logo.svg" alt="TeNeT" width="420">
</p>

# TeNeT - Tensor Network Thermodynamics

TeNeT is a tensor-network renormalization code for the statistical mechanics of
lattice models. It computes the grand potential `ln Z` per site of 1D/2D lattice models
via several renormalization-group (RG) schemes and takes finite-difference
derivatives of it to obtain thermodynamic observables (coverage/density,
entropy, susceptibility, heat capacity).

The main application is **adsorption of molecules on surfaces** - Langmuir and
binary lattice gases, hard-core models, and several molecule-specific models
(pentacene, 1,4-CHD on Si, a TPB + Cu system), alongside standard benchmarks
such as the 2D Ising model.

## Publications

Calculations for the following papers were performed with this code:

- Akimenko, S. S. (2023). Tensor network construction for lattice gas models: Hard-core and triangular lattice models. *Physical Review E*, 107(5), 054116. [https://doi.org/10.1103/PhysRevE.107.054116](https://doi.org/10.1103/PhysRevE.107.054116)
- Gorbunov, V. A., Uliankina, A. I., Akimenko, S. S., & Myshlyavtsev, A. V. (2023). Tensor renormalization group study of orientational ordering in simple models of adsorption monolayers. *Physical Review E*, 108(1), 014133. [https://doi.org/10.1103/PhysRevE.108.014133](https://doi.org/10.1103/PhysRevE.108.014133)
- Sergienko, A. V., Akimenko, S. S., Karpov, A. A., & Myshlyavtsev, A. V. (2024). Influence of the simplest type of multiparticle interactions on the example of a lattice model of an adsorption layer. *Computer research and modeling*, 16(2), 445-458. [https://doi.org/10.20537/2076-7633-2024-16-2-445-458](https://doi.org/10.20537/2076-7633-2024-16-2-445-458)
- Akimenko, S. S., Gorbunov, V. A., Myshlyavtsev, A. V., Myshlyavtseva, M. D., & Podgornyi, S. O. (2024). Shape-interaction dualism: unraveling complex phase behavior in triangular particle monolayers. *Journal of Physics: Condensed Matter*, 36(23), 235402. [https://doi.org/10.1088/1361-648X/ad2f56](https://doi.org/10.1088/1361-648X/ad2f56)
- Gorbunov, V. A., Uliankina, A. I., Akimenko, S. S., & Myshlyavtsev, A. V. (2024). Equilibrium structure of 1, 4-cyclohexadiene monolayer on Si (001)-(2× 1). *Physical Review B*, 110(4), 045416. [https://doi.org/10.1103/PhysRevB.110.045416](https://doi.org/10.1103/PhysRevB.110.045416)
- Uliankina, A. I., Gorbunov, V. A., Akimenko, S. S., & Myshlyavtsev, A. V. (2024). Initial growth of the pentacene monolayer on a Si (001)-2× 1 substrate: A lattice model view. *The Journal of Physical Chemistry C*, 128(41), 17658-17667. [https://doi.org/10.1021/acs.jpcc.4c04305](https://doi.org/10.1021/acs.jpcc.4c04305)
- Akimenko, S. S., & Myshlyavtsev, A. V. (2024). Tensor networks for hierarchical lattices. *Europhysics Letters*, 148(6), 61001. [https://doi.org/10.1209/0295-5075/ad994b](https://doi.org/10.1209/0295-5075/ad994b)
- Karpova, A. V., Akimenko, S. S., Uliankina, A. I., & Myshlyavtsev, A. V. (2025). Extending Tensor Network Methods Beyond Pairwise Interactions in Adsorption Systems. *The Journal of Physical Chemistry A*, 129(14), 3345-3352. [https://doi.org/10.1021/acs.jpca.4c08371](https://doi.org/10.1021/acs.jpca.4c08371)
- Karpova, A. V., Uliankina, A. I., Gorbunov, V. A., Akimenko, S. S., & Myshlyavtsev, A. V. (2026). Long-Range Interactions in 1D Adsorption Models: Tensor Network Approach. *Journal of Statistical Physics*, 193(2), 9. [https://doi.org/10.1007/s10955-025-03566-y](https://doi.org/10.1007/s10955-025-03566-y)

## Repository layout

```
Scripts/
  MainScripts.py     orchestration: CalcConfig, simulate(), thermodynamics()
  BuildTensors.py    build_matrix(): per-model Boltzmann weight matrices
  TensorNetworks.py  build_tensor() + the RG steps (trg/btrg/hotrg/tm/htn)
*.py                 entry scripts, one per model/experiment (run from the root)
tests/               pytest suite (fast golden tier + slow physics etalons)
Additional/          standalone reference code (exact Ising solution, etc.)
```

The root-level scripts (e.g. `mono.py`, `binary.py`, `ising.py`,
`14CHD_Si_simple_tm.py`, `Pentacene_model_3_trg.py`) are self-contained
experiments. Many of them embed a reference (`etalon`) result and assert against
it, so they double as regression tests.

## Requirements

- Python 3.11
- `numpy`, `scipy`
- `pytest` (to run the test suite)
- `matplotlib` (only for `Additional/TRG_code_for_registration.py`)

```bash
pip install numpy scipy pytest matplotlib
```

## Quick start

Run any entry script from the repository root:

```bash
python mono.py        # Langmuir gas, square lattice, TRG; self-checks its etalon
python ising.py       # 2D Ising, HOTRG
```

Or drive the engine directly:

```python
import Scripts.MainScripts as ms

calc = ms.CalcConfig()
calc.method   = "trg"        # trg | btrg | hotrg | tm | htn
calc.model    = "langmuir"
calc.lattice  = "square"
calc.metParam = 16           # bond dimension chi (truncation)
calc.constant = 1.0          # gas constant R; use 1.0 for dimensionless models

# ln Z per site (grand potential), m_par is model-specific (m_par[0] is mu)
lnZ = ms.simulate(calc, T=120.0, m_par=[4.0, 4.0])

# thermodynamic observables (finite-difference derivatives of ln Z)
obs = ms.thermodynamics(
    calc, T=120.0, m_par=[4.0, 4.0],
    coverage=True, susceptibility=True, entropy=True, heat_capacity=True,
)
print(obs["coverage"], obs["heat_capacity"], obs["grand_potential"])
```

### `thermodynamics(...)`

```python
thermodynamics(calc, T=1.0, m_par=None, *,
               coverage=False, susceptibility=False,
               entropy=False, heat_capacity=False,
               mu_index=0, dmu=1e-3, dT=1e-3) -> dict
```

Computes only the requested observables (and only the `simulate()` evaluations
they need), sharing the central point `Omega(mu0, T0)` between the two second
derivatives. Returns a dict with the requested keys among `coverage`,
`susceptibility`, `entropy`, `heat_capacity`, plus `grand_potential` whenever a
second derivative is requested.

`mu_index` selects which chemical potential in `m_par` is differentiated. It may
be a single index or a **list of indices that are shifted together** — this
gives the *total* coverage/susceptibility with respect to several chemical
potentials, which the multi-component adsorption models need, e.g.
`mu_index=[0, 1]`.

## Models, methods and lattices

`CalcConfig` selects everything. Key fields:

| field | meaning |
|-------|---------|
| `method` | RG scheme: `trg`, `btrg`, `hotrg`, `tm` (transfer matrix), `htn` (hierarchical lattice) |
| `model` | which lattice model (see below) |
| `lattice` | `square`, `triangular`, `hexagonal`, `complex`; `FSHL`/`diamond` for `htn` |
| `gen_tensor` | tensor-network construction variant (`default`, `svd`, `to_square`, `IRF`, `six_leg_tensor`, ...) |
| `metParam` | method parameter — bond dimension `chi` for `trg/btrg/hotrg/tm`; block size for `htn` |
| `metModification` | per-method tweak (e.g. `"hex"`, the BTRG `k`, the `tm` `[steps, chi]`) |
| `constant` | gas constant `R` (default `0.008314`); set to `1.0` for dimensionless models |
| `coord` | lattice coordination number |
| `join_tensors` | merge several sites into one tensor before the RG, `[horizontal, vertical]` |
| `iterations`, `methodTolerance` | RG iteration budget and convergence threshold |

Implemented models (`build_matrix` in `BuildTensors.py`):

- **Lattice gases:** `langmuir`, `langmuir_m`, `binary`, `hard-hexagon`, `dimers`, `dimers_test`
- **Spin / benchmark:** `ising`, `TLAT`, `qstate`
- **Exclusion series:** `1NN`, `2NN`, `3NN`, `4NN`, `5NN`
- **Adsorption (molecule-specific):** `CHD_simple`, `CHD_complex`,
  `Pentacene_model_1_simple`, `Pentacene_model_1_complex`,
  `Pentacene_model_2`, `Pentacene_model_3`, `six_leg_test`
- **Long-range interactions:** `1D_long-range`, `2D_long-range`, `2D_long-range_V`

`m_par` is model-specific; `m_par[0]` is the chemical potential `mu`. See the
corresponding branch in `build_matrix` (and the matching entry script) for the
exact layout of each model's parameters.

## Tests

The suite has two tiers:

```bash
pytest                 # fast tier (~2 s) — run on every change
pytest --runslow       # full physics validation (~4 h)
```

- **Fast tier** (`tests/test_build_matrix.py`, `tests/test_fast_engine.py`):
  golden fingerprints of `build_matrix` for every implemented model, plus small
  `simulate()` smoke tests covering each RG method and lattice geometry at tiny
  `chi`. Pins the construction (`BuildTensors.py`) and the engine
  (`TensorNetworks.py`) in milliseconds.
- **Slow tier** (`tests/test_etalon_scripts.py`): runs the 16 self-checking
  entry scripts at full scientific parameters and asserts each reproduces its
  embedded etalon. Gated behind `--runslow`.

Regenerate the golden fixtures from the current (known-good) code:

```bash
python -m tests.generate_golden
```

## Adding a model

1. Add a branch to `build_matrix` in `Scripts/BuildTensors.py` returning the
   model's Boltzmann weight matrices, and register the name in `models_dict`.
2. Add a fast `build_matrix` golden case in `tests/build_cases.py` and
   regenerate the golden (`python -m tests.generate_golden`).
3. Add an entry script (optionally with an `etalon` self-check) and list it in
   `tests/test_etalon_scripts.py` if you want it in the slow tier.

## Notes

- `Additional/` holds standalone reference code that is **not** part of the main
  pipeline: `exact_solution_for_ising_model.py` (the exact Onsager solution,
  used to validate the Ising results) and `TRG_code_for_registration.py` (a
  self-contained, interactive TRG demo with plotting).
