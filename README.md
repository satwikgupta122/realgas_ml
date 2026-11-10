# `realgas-ml` (v0.3.0) — Predicting Pressure–Volume–Temperature (PVT) and Thermodynamic Properties of Real Gases Using the Peng–Robinson Equation of State (EOS) and Machine Learning

**Authors:** Satwik Gupta (`2025CH11012`) & Udhav Jaipuria (`2025CH11729`)  
**Course Project:** Chemical Engineering Thermodynamics · Department of Chemical Engineering  
**Live Web App:** [**`https://satwikgupta122.github.io/realgas_ml/`**](https://satwikgupta122.github.io/realgas_ml/)  
**Project Report:** [**`report/realgas_ml_report.pdf`**](report/realgas_ml_report.pdf) ([LaTeX Source](report/realgas_ml_report.tex))

`realgas-ml` is a Python package and web application built for our Chemical Engineering Thermodynamics course project. It predicts the Pressure–Volume–Temperature ($PVT$), molar enthalpy ($h$), molar entropy ($s$), and fugacity ($\phi, f$) properties of **15 pure real gases** in the supercritical region. Each gas Machine Learning (ML) model is trained on **$65{,}536$ state points** ($983{,}040$ points in total) calculated directly from the [Peng–Robinson (1976) cubic Equation of State (PR76 EOS)](https://doi.org/10.1021/i160057a011) combined with [National Aeronautics and Space Administration (NASA) 7-coefficient ideal-gas heat-capacity polynomials](https://ntrs.nasa.gov/citations/19940013151). No external property library (like CoolProp or Cantera) is called during training or prediction.

## Repository Structure

```text
realgas_ml/
├── realgas_ml/                      # Main Python package
│   ├── __init__.py                  # Package exports (GasModel, gases)
│   ├── peng_robinson.py             # Peng-Robinson EOS, NASA7 integrals & departures
│   ├── regressor.py                 # Local quintic Radial Basis Function (RBF) model (k=48, degree=2)
│   ├── model.py                     # Forward (T,P) prediction & inverse P(T,v) solver
│   ├── cli.py                       # Command-Line Interface (CLI) tool (realgas-ml)
│   └── data/                        # Fluid constants, trained models, datasets & metrics
│       ├── fluid_parameters.json    # Critical constants, NASA7 coefficients & sources
│       ├── metadata.json            # Package metadata & training settings
│       ├── <Fluid>.joblib           # Saved LocalRBFRegressor models (15 gases)
│       ├── <Fluid>_dataset.npz      # Train/validation/test/boundary splits (15 gases)
│       └── <Fluid>_metrics.json     # Learning curves & test set errors (15 gases)
├── report/                          # Project report & figures
│   ├── realgas_ml_report.pdf        # 4-page project report (PDF)
│   ├── realgas_ml_report.tex        # LaTeX source file
│   ├── fig_learning_curves.pdf      # Validation learning curves (PDF)
│   ├── fig_learning_curves.png      # Validation learning curves (PNG)
│   ├── fig_test_boundary_errors.pdf # Interior vs. boundary test errors (PDF)
│   └── fig_test_boundary_errors.png # Interior vs. boundary test errors (PNG)
├── docs/                            # Interactive web app for GitHub Pages
│   ├── index.html                   # Browser-based Real Gas Lab calculator
│   ├── realgas_ml_report.pdf        # 4-page project report served on GitHub Pages
│   └── realgas_ml_report.tex        # LaTeX source served on GitHub Pages
├── tests/                           # Automated pytest suite (47 tests)
│   ├── test_properties.py           # Thermodynamic identities & model accuracy checks
│   └── test_app.py                  # Streamlit UI, unit conversion & molar scaling tests
├── .streamlit/config.toml           # Streamlit theme settings
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Dependencies for Streamlit Cloud deployment
├── app.py                           # Interactive Streamlit web app
├── train.py                         # Dataset generation & model training script
├── export_data.py                   # Script to export NumPy Zip (NPZ) datasets to Comma-Separated Values (CSV)
├── example.py                       # Simple Python Application Programming Interface (API) example script
├── run_app.bat                      # Windows launcher for the Streamlit app
├── pyproject.toml                   # Package build configuration
└── THIRD_PARTY_NOTICES.txt          # BSD 3-Clause license notice for NASA7 tables
```

## How to Install and Run

### 1. Browser Interface (Live Web App or Local Streamlit App)

You can use the live browser app directly at [**`https://satwikgupta122.github.io/realgas_ml/`**](https://satwikgupta122.github.io/realgas_ml/), or clone the repository and run the Streamlit User Interface (UI) locally:

```bash
git clone https://github.com/satwikgupta122/realgas_ml.git
cd realgas_ml
python -m pip install ".[ui]"
python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically (Windows users can double-click `run_app.bat` after installation). Choose a gas and enter temperature (`K` or `°C`), pressure (`bar`, `kPa`, `MPa`, or `Pa`), and number of moles (`mol`), or switch to **Temperature + molar volume** mode to calculate pressure from $T$ and $v$. You can compare the Machine Learning (ML) model predictions with direct Peng–Robinson Equation of State (EOS) calculations, plot pressure curves, and download Comma-Separated Values (CSV) results.

### 2. Python Application Programming Interface (API) and Command-Line Interface (CLI)

Use Python 3.11 or newer:

```bash
python -m pip install .
python example.py
realgas-ml --gas CarbonDioxide --T 400 --P 2000000 --n 2
```

In Python:

```python
from realgas_ml import GasModel
from realgas_ml.peng_robinson import properties

# Predict properties with the trained Machine Learning (ML) model
model = GasModel("CarbonDioxide")
result = model.predict(T=400, P=2000000, n=2)
print(result["fugacity_Pa"], result["phi"], result["H_J"], result["S_J_K"])
print("Recovered P [Pa]:", model.pressure(T=400, v=result["v_m3_mol"]))

# Calculate directly with Peng-Robinson EOS (without loading an ML model)
print(properties(400, 2000000, "CarbonDioxide"))
```

Inputs are in International System of Units (SI): temperature in kelvin (`K`), pressure in pascals (`Pa`), and amount in moles (`mol`). Note that fugacity (`fugacity_Pa`) is an intensive property, so it is not multiplied by the number of moles `n`.

### 3. Training Models and Running Tests

```bash
python train.py                                      # Train models for all 15 gases
python train.py --gas CarbonDioxide                  # Train a single gas
python export_data.py --gas CarbonDioxide --output co2.csv
python -m pip install ".[test]"
python -m pytest -q                                  # Run all 47 automated tests
```

## Selected Gases and Operating Ranges

All 15 pure gases operate over $P \in [1.0, 50.0]\;\text{bar}$ ($[10^5, 5\times 10^6]\;\text{Pa}$) and supercritical temperatures $T \ge \max(300.0\;\text{K}, 1.10\,T_c)$ as listed in [`realgas_ml/data/fluid_parameters.json`](realgas_ml/data/fluid_parameters.json):

| Fluid | Formula | Molar Mass $M$ ($\text{g/mol}$) | Critical Temp. $T_c$ ($\text{K}$) | Critical Pressure $P_c$ ($\text{bar}$) | Acentric Factor $\omega$ | Operating Range $T_{\min}$–$T_{\max}$ ($\text{K}$) | Reduced Temp. $T_r = T/T_c$ Range |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Helium** | $\text{He}$ | 4.0026 | 5.1953 | 2.2832 | $-0.38354$ | 300.00–600.00 | 57.74–115.49 |
| **Hydrogen** | $\text{H}_2$ | 2.0159 | 32.9800 | 12.9300 | $-0.21700$ | 300.00–600.00 | 9.10–18.19 |
| **Neon** | $\text{Ne}$ | 20.1790 | 44.4000 | 26.6163 | $-0.03549$ | 300.00–600.00 | 6.76–13.51 |
| **Nitrogen** | $\text{N}_2$ | 28.0135 | 126.2000 | 33.9000 | $0.03800$ | 300.00–600.00 | 2.38–4.75 |
| **CarbonMonoxide** | $\text{CO}$ | 28.0101 | 133.4000 | 35.0000 | $0.06600$ | 300.00–499.00 | 2.25–3.74 |
| **Argon** | $\text{Ar}$ | 39.9480 | 150.6870 | 48.6300 | $-0.00219$ | 300.00–600.00 | 1.99–3.98 |
| **Oxygen** | $\text{O}_2$ | 31.9988 | 154.5800 | 50.4300 | $0.02200$ | 300.00–600.00 | 1.94–3.88 |
| **Methane** | $\text{CH}_4$ | 16.0428 | 190.7000 | 46.3000 | $0.01100$ | 300.00–600.00 | 1.57–3.15 |
| **CarbonDioxide** | $\text{CO}_2$ | 44.0098 | 304.2000 | 73.9000 | $0.22800$ | 334.62–600.00 | 1.10–1.97 |
| **Ethane** | $\text{C}_2\text{H}_6$ | 30.0690 | 305.4000 | 48.8000 | $0.09900$ | 335.94–600.00 | 1.10–1.96 |
| **Propane** | $\text{C}_3\text{H}_8$ | 44.0956 | 369.7000 | 42.5000 | $0.15300$ | 406.88–628.81 | 1.10–1.70 |
| **Ammonia** | $\text{NH}_3$ | 17.0305 | 405.5600 | 113.6339 | $0.25569$ | 446.12–689.45 | 1.10–1.70 |
| **n-Butane** | $n\text{-C}_4\text{H}_{10}$ | 58.1222 | 425.6000 | 37.6000 | $0.19900$ | 468.16–574.00 | 1.10–1.35 |
| **SulfurDioxide** | $\text{SO}_2$ | 64.0638 | 430.6400 | 78.8658 | $0.25613$ | 473.70–524.00 | 1.10–1.22 |
| **Water** | $\text{H}_2\text{O}$ | 18.0153 | 647.0960 | 220.6400 | $0.34429$ | 711.81–1100.06 | 1.10–1.70 |

## Equations Used

For molar volume $v$, the Peng–Robinson (1976) Equation of State (EOS) in [`realgas_ml/peng_robinson.py`](realgas_ml/peng_robinson.py) is:

$$P(T,v) = \frac{RT}{v - b} - \frac{a(T)}{v^2 + 2bv - b^2}, \qquad a(T) = 0.45724\,\frac{R^2T_c^2}{P_c}\left[1 + \kappa\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2, \qquad b = 0.07780\,\frac{RT_c}{P_c}$$

where $\kappa = 0.37464 + 1.54226\,\omega - 0.26992\,\omega^2$. Defining $A = aP/(RT)^2$, $B = bP/(RT)$, and $L = \ln[(Z + (1 + \sqrt{2})B)/(Z + (1 - \sqrt{2})B)]$:

- $\ln\phi = Z - 1 - \ln(Z - B) - \frac{A}{2\sqrt{2}B}L$ and fugacity $f = \phi P$
- $h_{\text{dep}} = RT(Z - 1) + \frac{T(da/dT) - a}{2\sqrt{2}b}L$
- $s_{\text{dep}} = R\ln(Z - B) + \frac{da/dT}{2\sqrt{2}b}L$

Ideal-gas enthalpy and entropy changes ($h_{\text{ig}}, s_{\text{ig}}$) are integrated analytically from the National Aeronautics and Space Administration (NASA) 7-coefficient polynomials starting from $h_{\text{ig}} = s_{\text{ig}} = 0$ at $300\;\text{K}$ and $1\;\text{bar}$, and the real-gas departures are added at the actual $(T, P)$ state. Each gas model in [`realgas_ml/regressor.py`](realgas_ml/regressor.py) uses local quintic Radial Basis Function (RBF) interpolation ($k=48$ nearest neighbors, degree-2 polynomial) over normalized $(T, \ln P) \in [0, 1]^2$ to predict $[\ln Z, h, s, \ln\phi]$.

## Validation Results and Error Metrics

All 47 automated tests pass (`pytest -q`). We evaluate accuracy using:
- **Mean Absolute Percentage Error (MAPE)** and **Maximum Absolute Percentage Error (Max APE)** for positive quantities (molar volume $v$ and fugacity $f$).
- **Mean Absolute Error (MAE)** and **Maximum Absolute Error (Max AE)** for caloric quantities (molar enthalpy $h$ and molar entropy $s$).

Across all 15 gases on both the Interior Test Split ($N=2{,}048$) and Boundary Test Split ($N=256$), the largest errors compared to direct Peng–Robinson Equation of State (EOS) calculations are:

| Error Metric (Full Form & Metric Key) | Largest Test Error | Automated Test Threshold |
| :--- | ---: | ---: |
| **Molar Volume Maximum Absolute Percentage Error** (`v_max_APE_percent`) | `1.0365e-05 %` | `< 0.01 %` |
| **Fugacity Maximum Absolute Percentage Error** (`fugacity_max_APE_percent`) | `1.49849e-06 %` | `< 0.01 %` |
| **Molar Enthalpy Maximum Absolute Error** (`h_J_mol_max_AE`) | `0.000409671 J/mol` | `< 0.5 J/mol` |
| **Molar Entropy Maximum Absolute Error** (`s_J_mol_K_max_AE`) | `8.24171e-07 J/(mol·K)` | `< 0.005 J/(mol·K)` |

![Validation Learning Curves](report/fig_learning_curves.png)

![Interior and Boundary Test Errors](report/fig_test_boundary_errors.png)

## Sources

- Peng, D.-Y., & Robinson, D. B. (1976), *A New Two-Constant Equation of State*: https://doi.org/10.1021/i160057a011
- McBride, B. J., Gordon, S., & Reno, M. A. (1993), *National Aeronautics and Space Administration (NASA) Technical Memorandum 4513*: https://ntrs.nasa.gov/citations/19940013151
- Cantera 3.2.0 NASA 7-coefficient polynomial reference: https://www.cantera.org/stable/reference/thermo/species-thermo.html
- Lemmon, E. W., et al. (2022), *National Institute of Standards and Technology (NIST) Chemistry WebBook, SRD 69*: https://doi.org/10.18434/T4D303
- Maginn, E. J., *CBE 20260 Fugacity from Equations of State*: https://ejmaginn.github.io/CBE20260-notes/class34/fugacity-EOS.html
- License notices for the static NASA7 coefficient tables are in [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt).
