"""Check EOS consistency, thermodynamic identities and trained predictions."""
import numpy as np
import pytest
from realgas_ml import GasModel, gases
from realgas_ml.peng_robinson import R, parameters, properties, pressure_tv, ideal_integrals


@pytest.mark.parametrize('gas', gases())
def test_thermodynamics(gas):
    info = parameters(gas)
    T = sum(info['T_range_K'])/2
    P = 2e6
    state = properties(T, P, gas)
    assert pressure_tv(T, state['v_m3_mol'], info) == pytest.approx(P, rel=1e-10)
    # Residual Gibbs energy must equal RT ln(phi).
    gres = state['h_departure_J_mol']-T*state['s_departure_J_mol_K']
    assert gres == pytest.approx(R*T*np.log(state['phi']), abs=1e-8)
    assert state['fugacity_Pa'] == pytest.approx(P*state['phi'])
    dt, dp = 0.01, 100.0
    warm, cool = properties(T+dt, P, gas), properties(T-dt, P, gas)
    high, low = properties(T, P+dp, gas), properties(T, P-dp, gas)
    dv_dt = (warm['v_m3_mol']-cool['v_m3_mol'])/(2*dt)
    dh_dp = (high['h_J_mol']-low['h_J_mol'])/(2*dp)
    ds_dp = (high['s_J_mol_K']-low['s_J_mol_K'])/(2*dp)
    assert dh_dp == pytest.approx(state['v_m3_mol']-T*dv_dt, rel=2e-5, abs=1e-10)
    assert ds_dp == pytest.approx(-dv_dt, rel=2e-5)
    dlogf_dp = (np.log(high['fugacity_Pa'])-np.log(low['fugacity_Pa']))/(2*dp)
    assert dlogf_dp == pytest.approx(state['v_m3_mol']/(R*T), rel=2e-5)
    dh_dt = (warm['h_J_mol']-cool['h_J_mol'])/(2*dt)
    ds_dt = (warm['s_J_mol_K']-cool['s_J_mol_K'])/(2*dt)
    assert dh_dt == pytest.approx(T*ds_dt, rel=2e-5)
    assert ideal_integrals(300, info) == pytest.approx((0, 0))


@pytest.mark.parametrize('gas', gases())
def test_model(gas):
    model = GasModel(gas)
    T = sum(model.info['T_range_K'])/2
    direct = properties(T, 2e6, gas)
    pred = model.predict(T, 2e6, n=2)
    for key in ['Z', 'v_m3_mol', 'phi', 'fugacity_Pa']:
        assert pred[key] == pytest.approx(direct[key], rel=1e-4)
    assert pred['H_J'] == 2*pred['h_J_mol']
    assert pred['S_J_K'] == 2*pred['s_J_mol_K']
    assert model.pressure(T, pred['v_m3_mol']) == pytest.approx(2e6, rel=1e-8)
    with pytest.raises(ValueError):
        model.predict(T, 0)
    with pytest.raises(ValueError):
        model.predict(T, 2e6, n=-1)


@pytest.mark.parametrize('gas', gases())
def test_held_out_accuracy(gas):
    import json
    from pathlib import Path
    data = Path(__file__).parents[1]/'realgas_ml'/'data'
    report = json.loads((data/f'{gas}_metrics.json').read_text())
    for split in ['test', 'boundary']:
        assert report[split]['v_max_APE_percent'] < 0.01
        assert report[split]['fugacity_max_APE_percent'] < 0.01
        assert report[split]['h_J_mol_max_AE'] < 0.5
        assert report[split]['s_J_mol_K_max_AE'] < 0.005
