"""Exercise unit conversion, gas selection and the inverse calculation."""
from pathlib import Path
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1]/'app.py')


def test_units_and_inverse():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    initial = [metric.value for metric in app.metric]
    app.selectbox(key='temp_unit').set_value('°C').run()
    app.selectbox(key='p_unit').set_value('MPa').run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == initial
    app.radio(key='mode').set_value('Temperature + molar volume').run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == initial
    app.number_input(key='volume_CarbonDioxide').set_value(0.000001).run()
    assert app.error
    assert len(app.metric) == 0  # No stale results for an invalid state.


def test_gases_and_amount():
    app = AppTest.from_file(APP, default_timeout=30).run()
    for gas in app.selectbox(key='gas').options:
        app.selectbox(key='gas').set_value(gas).run()
        assert not app.exception, gas
        assert not app.error, gas
    app.selectbox(key='gas').set_value('CarbonDioxide').run()
    before = app.dataframe[0].value
    app.number_input(key='amount').set_value(2.0).run()
    after = app.dataframe[0].value
    assert not app.exception
    for name in ['Total volume', 'Total enthalpy', 'Total entropy']:
        a = before.loc[before.Property == name, 'ML prediction'].iloc[0]
        b = after.loc[after.Property == name, 'ML prediction'].iloc[0]
        assert b == pytest.approx(2*a)
    for name in ['Molar volume', 'Fugacity']:
        a = before.loc[before.Property == name, 'ML prediction'].iloc[0]
        b = after.loc[after.Property == name, 'ML prediction'].iloc[0]
        assert b == pytest.approx(a)
