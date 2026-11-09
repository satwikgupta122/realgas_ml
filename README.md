# PVT behaviour of real gases

A Chemical Engineering Thermodynamics project. Python package `realgas-ml` 0.3.0 ([GitHub Repository](https://github.com/satwikgupta122/realgas_ml)).

Predicts properties of 15 pure gases using models trained on **Peng–Robinson (PR) calculations implemented in this project**. Each gas has 65,536 training states: 983,040 in total. No thermodynamic property library is called during training or prediction.

## Browser interface (recommended)

Extract the ZIP, open a terminal in the folder containing `app.py`, then run:

```bash
python -m pip install ".[ui]"
python -m streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically. Keep the terminal open; press Ctrl+C to stop. After the initial installation, Windows users can double-click `run_app.bat` in this folder.

Choose a gas and enter temperature in K or °C, pressure in bar/kPa/MPa/Pa, and amount in mol. Results update automatically. You can also calculate pressure from temperature and molar volume, compare the model with direct PR, explore pressure curves and download CSV results. Temperature bounds follow the selected gas; invalid volumes show a helpful message.

The interface runs locally on your computer and uses the included trained models; no retraining or thermodynamic property service is needed. `app.py` contains the interface code. Streamlit is an optional UI dependency; the command-line package still works without it.

## Install and run

Use Python 3.11 or newer. Open a terminal in this folder:

```bash
python -m pip install .
python example.py
realgas-ml --gas CarbonDioxide --T 400 --P 2000000 --n 2
```

```python
from realgas_ml import GasModel

model = GasModel('CarbonDioxide')
result = model.predict(T=400, P=2000000, n=2)
print(result['fugacity_Pa'])
print(result['phi'])
print(result['H_J'], result['S_J_K'])
```

Inputs are temperature in kelvin, pressure in pascals and amount in moles. Outputs include Z, molar volume, total volume, density, molar enthalpy and entropy, total H and S, fugacity coefficient `phi` and fugacity `fugacity_Pa`. `model.pressure(T, v)` inverts predicted molar volume to pressure. Fugacity is an intensive property: it is not multiplied by the number of moles.

To calculate directly with PR without loading an ML model:

```python
from realgas_ml.peng_robinson import properties
print(properties(400, 2000000, 'CarbonDioxide'))
```

## Train and export data

```bash
python train.py
python train.py --gas CarbonDioxide
python export_data.py --gas CarbonDioxide --output co2.csv
python -m pip install pytest
python -m pytest -q
```

Training regenerates the data and overwrites the selected models. Each gas uses a 256 by 256 grid in temperature and log-pressure. A fixed seed gives 512 validation points, 2,048 interior test points and 256 boundary test points per gas. Validation learning curves cover 1,024, 4,096, 16,384 and 65,536 training points. The final model always uses 65,536; test data do not select the model.

`train.py` contains the complete data-generation and training loop. NumPy handles arrays and polynomial roots; SciPy supplies local RBF interpolation and pressure inversion; scikit-learn supplies scaling and the regression interface; joblib saves models. These numerical dependencies remain necessary. CoolProp and Cantera are not dependencies.

## Equations

For molar volume v, the original PR76 equation is

P = RT/(v − b) − a(T)/(v² + 2bv − b²).

a(T) = 0.45724 R²Tc²/Pc × [1 + kappa(1 − sqrt(T/Tc))]²;
b = 0.07780 RTc/Pc;
kappa = 0.37464 + 1.54226 omega − 0.26992 omega².

The code solves the cubic for Z with `numpy.roots` and selects the largest real root with Z > B. Define A = aP/(RT)², B = bP/(RT), and L = ln[(Z + (1 + sqrt(2))B)/(Z + (1 − sqrt(2))B)]. Then:

- ln(phi) = Z − 1 − ln(Z − B) − A L/(2 sqrt(2) B).
- Fugacity f = phi × P.
- h departure = RT(Z − 1) + [T da/dT − a] L/(2 sqrt(2) b).
- s departure = R ln(Z − B) + [da/dT] L/(2 sqrt(2) b).

Ideal-gas h and s come from analytical integration of NASA heat-capacity polynomials. Their reference is h_ideal = s_ideal = 0 at 300 K and 1 bar; PR departures are added at the actual state. Real-gas h and s therefore need not be zero at that reference state. These values are not formation enthalpies or absolute third-law entropies.

The model learns `[ln(Z), h, s, ln(phi)]` with local quintic RBF regression using 48 neighbors and degree 2. Exponentiation keeps predicted Z and phi positive. Independently fitted outputs do not guarantee exact thermodynamic derivative identities; use the direct EOS for those calculations.

## Gases and limits

Nitrogen, Oxygen, CarbonDioxide, Hydrogen, Helium, Argon, Neon, Methane, Ethane, Propane, n-Butane, CarbonMonoxide, Ammonia, SulfurDioxide and Water. Exact temperature ranges are in `realgas_ml/data/fluid_parameters.json`; all start above the critical temperature and remain within the heat-capacity correlation range. Pressure range: 1–50 bar. Inputs outside the trained range are rejected. This package does not solve mixtures, liquid properties or phase equilibrium.

Training data are **EOS-generated, not experimental measurements**. Reported errors measure agreement with PR, not accuracy against nature. PR can have appreciable physical error, especially for light gases and polar fluids. A small ML error does not remove that limitation.

## Sources

- Peng and Robinson (1976), *A New Two-Constant Equation of State*: https://doi.org/10.1021/i160057a011
- NASA TM-4513, thermodynamic polynomial data: https://ntrs.nasa.gov/citations/19940013151
- NASA7 equations: https://www.cantera.org/stable/reference/thermo/species-thermo.html
- Static coefficient tables: Cantera 3.2.0 `nasa_gas.yaml` and `critical-properties.yaml`. Selected missing critical constants are transcribed from the fluid documentation URLs recorded per gas in `fluid_parameters.json`. These are static input numbers, not property-library calculations.
- Fugacity equation: https://ejmaginn.github.io/CBE20260-notes/class34/fugacity-EOS.html

Every gas records its coefficient provenance and critical-constant source. `THIRD_PARTY_NOTICES.txt` retains the coefficient distribution license.

## Files

- `realgas_ml/peng_robinson.py`: direct EOS, h, s and fugacity equations.
- `train.py`: generate labels, fit and evaluate models.
- `realgas_ml/model.py`: prediction interface.
- `realgas_ml/data/*_dataset.npz`: training and evaluation arrays.
- `realgas_ml/data/*_metrics.json`: errors and validation learning curves.
- `export_data.py`: export physical values to CSV or compressed CSV.
- `tests/test_properties.py`: EOS identities, fugacity checks and model accuracy.

## Validation results

All 47 tests pass, including interface tests for all gases, unit conversions, amount scaling and invalid volumes. Across all 15 gases and both interior and boundary test sets, the largest absolute errors against direct PR are:

| Quantity | Maximum error |
|---|---:|
| v_max_APE_percent | 1.0365e-05 |
| fugacity_max_APE_percent | 1.49849e-06 |
| h_J_mol_max_AE | 0.000409671 |
| s_J_mol_K_max_AE | 8.24171e-07 |

These are interpolation errors against the implemented EOS, not experimental errors.
