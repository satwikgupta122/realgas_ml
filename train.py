"""Generate Peng–Robinson data, train four-output models, and report errors."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from realgas_ml.peng_robinson import properties
from realgas_ml.regressor import LocalRBFRegressor

DATA = Path(__file__).parent / 'realgas_ml' / 'data'
SEED = 20261016


def grid(side):
    """Uniform grid in normalized temperature and log-pressure."""
    axis = np.linspace(0, 1, side)
    return np.array([(t, p) for t in axis for p in axis])


def labels(X, info):
    tmin, tmax = info['T_range_K']
    pmin, pmax = info['P_range_Pa']
    T = tmin + X[:, 0]*(tmax-tmin)
    P = pmin*(pmax/pmin)**X[:, 1]
    result = properties(T, P, info)
    y = np.column_stack([np.log(result['Z']), result['h_J_mol'],
                         result['s_J_mol_K'], np.log(result['phi'])])
    if not np.all(np.isfinite(y)):
        raise ValueError('Non-finite training data')
    return y


def errors(pred, truth):
    """Relative errors for volume/fugacity; absolute errors for h/s."""
    result = {}
    for name, column in [('v', 0), ('fugacity', 3)]:
        error = 100*np.abs(np.exp(pred[:, column]-truth[:, column])-1)
        result[name+'_MAPE_percent'] = float(error.mean())
        result[name+'_max_APE_percent'] = float(error.max())
    for name, column in [('h_J_mol', 1), ('s_J_mol_K', 2)]:
        error = np.abs(pred[:, column]-truth[:, column])
        result[name+'_MAE'] = float(error.mean())
        result[name+'_max_AE'] = float(error.max())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gas', help='Train just one gas')
    args = parser.parse_args()
    metadata = json.loads((DATA/'fluid_parameters.json').read_text())
    if args.gas and args.gas not in metadata['fluids']:
        parser.error('Unknown gas')
    metadata.update(training_points_per_gas=65536, seed=SEED,
                    targets=['ln_Z', 'h_J_mol', 's_J_mol_K', 'ln_phi'])
    for index, (gas, info) in enumerate(metadata['fluids'].items()):
        if args.gas and gas != args.gas:
            continue
        rng = np.random.default_rng(SEED+index)
        boundary = []
        for side in [0, 1]:
            boundary.extend([(side, p) for p in rng.random(64)])
            boundary.extend([(t, side) for t in rng.random(64)])
        X = dict(train=grid(256), validation=rng.random((512, 2)),
                 test=rng.random((2048, 2)), boundary=np.array(boundary))
        y = {split: labels(points, info) for split, points in X.items()}
        curve = []
        for side in [32, 64, 128, 256]:
            points = X['train'] if side == 256 else grid(side)
            values = y['train'] if side == 256 else labels(points, info)
            model = LocalRBFRegressor().fit(points, values)
            curve.append(dict(points=len(points), **errors(model.predict(X['validation']), y['validation'])))
        # Fixed final size: test data never select the model or its settings.
        report = {split: errors(model.predict(X[split]), y[split])
                  for split in ['validation', 'test', 'boundary']}
        report['learning_curve'] = curve
        (DATA/f'{gas}_metrics.json').write_text(json.dumps(report, indent=2))
        joblib.dump(model, DATA/f'{gas}.joblib', compress=3)
        arrays = {}
        for split in X:
            arrays[split+'_X'] = X[split]
            arrays[split+'_y'] = y[split]
        np.savez_compressed(DATA/f'{gas}_dataset.npz', **arrays)
        print(gas, json.dumps(report['test']), flush=True)
    (DATA/'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
