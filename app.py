"""Local browser interface. Run: python -m streamlit run app.py"""
import csv
import io

import numpy as np
import streamlit as st

from realgas_ml import GasModel, gases
from realgas_ml.peng_robinson import properties

st.set_page_config(page_title='Real gas lab', page_icon='🧪', layout='wide')
st.markdown('''<style>
.block-container {max-width:1200px; padding-top:2rem;}
[data-testid="stMetric"] {background:#f0f5f5; border-radius:12px; padding:16px;}
</style>''', unsafe_allow_html=True)

# Cache the trained model so changing an input does not reload it from disk.
@st.cache_resource
def load_model(gas):
    return GasModel(gas)


def csv_text(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# Display units are separate from the SI units used by the package.
FIELDS = [
    ('Z', 'Compressibility factor, Z', '—', 1),
    ('v_m3_mol', 'Molar volume', 'L/mol', 1000),
    ('V_m3', 'Total volume', 'L', 1000),
    ('rho_kg_m3', 'Density', 'kg/m³', 1),
    ('h_J_mol', 'Molar enthalpy', 'kJ/mol', 0.001),
    ('s_J_mol_K', 'Molar entropy', 'J/(mol·K)', 1),
    ('H_J', 'Total enthalpy', 'kJ', 0.001),
    ('S_J_K', 'Total entropy', 'J/K', 1),
    ('phi', 'Fugacity coefficient, φ', '—', 1),
    ('fugacity_Pa', 'Fugacity', 'bar', 0.00001),
]

st.caption('CHEMICAL ENGINEERING THERMODYNAMICS')
st.title('Real gas lab')
st.write('Explore PVT behaviour, energy and fugacity for 15 pure gases.')

with st.sidebar:
    st.header('Set your state')
    gas = st.selectbox('Gas', gases(), index=list(gases()).index('CarbonDioxide'), key='gas')
    model = load_model(gas)
    tmin, tmax = model.info['T_range_K']
    pmin, pmax = model.info['P_range_Pa']
    mode = st.radio('Calculate from', ['Temperature + pressure', 'Temperature + molar volume'], key='mode')
    temp_unit = st.selectbox('Temperature unit', ['K', '°C'], key='temp_unit')
    offset = 273.15 if temp_unit == '°C' else 0.0
    t_default = min(max(400.0, tmin), tmax)
    temp = st.number_input(f'Temperature ({temp_unit})', min_value=tmin-offset,
                           max_value=tmax-offset, value=t_default-offset,
                           format='%.4f', key=f'temp_{gas}_{temp_unit}')
    T = min(max(temp+offset, tmin), tmax)
    st.caption(f'Valid range: {tmin-offset:.4f}–{tmax-offset:.4f} {temp_unit}')
    n = st.number_input('Amount (mol)', min_value=0.000001, value=1.0,
                        format='%.6f', key='amount')
    if mode == 'Temperature + pressure':
        p_unit = st.selectbox('Pressure unit', ['bar', 'kPa', 'MPa', 'Pa'], key='p_unit')
        factor = {'bar': 1e5, 'kPa': 1e3, 'MPa': 1e6, 'Pa': 1}[p_unit]
        pressure = st.number_input(f'Pressure ({p_unit})', min_value=pmin/factor,
                                  max_value=pmax/factor, value=2e6/factor,
                                  format='%.6f', key=f'pressure_{p_unit}')
        P = min(max(pressure*factor, pmin), pmax)
    else:
        vmin = model.predict(T, pmax)['v_m3_mol']
        vmax = model.predict(T, pmin)['v_m3_mol']
        volume = st.number_input('Molar volume (L/mol)', min_value=0.000001,
                                 value=model.predict(T, 2e6)['v_m3_mol']*1000,
                                 format='%.8f', key=f'volume_{gas}')
        st.caption(f'At this temperature: {vmin*1000:.6g}–{vmax*1000:.6g} L/mol')
        try:
            P = model.pressure(T, volume/1000)
        except ValueError:
            st.error('This volume is outside the trained pressure range. Enter a volume within the range above.')
            st.stop()
    st.caption('Results update when you change an input. Press Enter after typing.')

try:
    result = model.predict(T, P, n)
    direct = {key: float(value) for key, value in properties(T, P, gas).items()}
except ValueError as error:
    st.error(str(error))
    st.stop()
direct.update(V_m3=n*direct['v_m3_mol'], H_J=n*direct['h_J_mol'], S_J_K=n*direct['s_J_mol_K'])

st.subheader(f'{gas} · {T:.2f} K · {P/1e5:.4f} bar')
st.caption(f'{n:g} mol · ML prediction trained on Peng–Robinson data')
columns = st.columns(4)
for column, (label, value) in zip(columns, [
    ('Compressibility, Z', f"{result['Z']:.6f}"),
    ('Molar volume · L/mol', f"{result['v_m3_mol']*1000:.6g}"),
    ('Fugacity · bar', f"{result['fugacity_Pa']/1e5:.6g}"),
    ('Density · kg/m³', f"{result['rho_kg_m3']:.6g}"),
]):
    column.metric(label, value)

results_tab, plot_tab, notes_tab = st.tabs(['All properties', 'Pressure trends', 'How to read this'])
with results_tab:
    compare = st.checkbox('Compare with direct Peng–Robinson', value=True)
    rows = []
    for key, label, unit, scale in FIELDS:
        row = {'Property': label, 'Unit': unit, 'ML prediction': result[key]*scale}
        if compare:
            row['Direct PR'] = direct[key]*scale
            row['Absolute difference'] = abs(result[key]-direct[key])*scale
        rows.append(row)
    st.dataframe(rows, hide_index=True, use_container_width=True,
                 column_config={name: st.column_config.NumberColumn(format='%.8g')
                                for name in ['ML prediction', 'Direct PR', 'Absolute difference']})
    # Download includes the state and amount, not just unlabeled property numbers.
    export = [dict(gas=gas, T_K=T, P_Pa=P, n_mol=n, **row) for row in rows]
    st.download_button('Download results as CSV', csv_text(export),
                       file_name=f'{gas}_properties.csv', mime='text/csv')

with plot_tab:
    choices = {label: (key, unit, scale) for key, label, unit, scale in FIELDS}
    label = st.selectbox('Property to plot', list(choices), key='plot_property')
    key, unit, scale = choices[label]
    pressures = np.geomspace(pmin, pmax, 80)
    ml_values = [model.predict(T, float(p), n)[key]*scale for p in pressures]
    pr = properties(T, pressures, gas)
    pr.update(V_m3=n*pr['v_m3_mol'], H_J=n*pr['h_J_mol'], S_J_K=n*pr['s_J_mol_K'])
    chart = {'Pressure (bar)': pressures/1e5, 'ML prediction': ml_values,
             'Direct PR': pr[key]*scale}
    st.caption(f'{label} ({unit}) at {T:.2f} K and {n:g} mol')
    st.line_chart(chart, x='Pressure (bar)', y=['ML prediction', 'Direct PR'],
                  x_label='Pressure (bar)', y_label=f'{label} ({unit})',
                  color=['#087f8c', '#e6a23c'])
    st.caption('The curves may overlap because the model closely reproduces PR.')
    chart_rows = [dict(gas=gas, T_K=T, n_mol=n, property=label, unit=unit,
                       **dict(zip(chart, values))) for values in zip(*chart.values())]
    st.download_button('Download curve as CSV', csv_text(chart_rows),
                       file_name=f'{gas}_pressure_curve.csv', mime='text/csv')

with notes_tab:
    st.markdown('''
**Z** measures departure from ideal-gas volume: Z = 1 is ideal at that state.

**Fugacity** is f = φP. Its coefficient φ measures departure from ideal-gas chemical potential.
Fugacity has pressure units and does not increase when the amount of gas increases.

**h and s** are molar properties; **H = nh** and **S = ns** are totals.
The ideal-gas reference is zero at 300 K and 1 bar, with PR departures added.
Negative values are possible with this reference.

**Scope:** pure gases above their critical temperature, within the displayed training ranges.
The model uses 65,536 training states per gas. It does not calculate mixtures or phase equilibrium.

**Accuracy:** the comparison measures interpolation error against Peng–Robinson,
not experimental accuracy. PR itself is approximate, particularly for light and polar gases.
''')
