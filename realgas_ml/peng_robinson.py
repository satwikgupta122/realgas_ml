"""Pure-fluid Peng-Robinson EOS and caloric departure functions in SI units.

No property-library calls. Inputs are documented constants and NASA7 Cp
coefficients bundled as JSON. Only the supported supercritical-temperature
gas domain is exposed; this is not a liquid/vapor equilibrium solver.
"""
import json
from importlib.resources import files
import numpy as np

R = 8.31446261815324
T_REF = 300.0
P_REF = 100000.0
SQRT2 = np.sqrt(2.0)


def parameters(gas):
    data = json.loads(files('realgas_ml').joinpath('data/fluid_parameters.json').read_text())
    if gas not in data['fluids']:
        raise ValueError(f'Unknown gas {gas!r}')
    return data['fluids'][gas]


def attraction(T, info):
    """Return a(T), da/dT and b for the original PR76 alpha function."""
    tc, pc, w = info['Tc_K'], info['Pc_Pa'], info['acentric_factor']
    kappa = 0.37464 + 1.54226*w - 0.26992*w*w
    root = np.sqrt(np.asarray(T)/tc)
    g = 1 + kappa*(1-root)
    ac = 0.45724*R**2*tc**2/pc
    return ac*g*g, -ac*kappa*g/(tc*root), 0.07780*R*tc/pc


def cubic_z(A, B):
    """Solve the PR cubic and choose the largest real root (gas branch)."""
    A, B = np.broadcast_arrays(A, B)
    result = []
    for a, b in zip(A.flat, B.flat):
        roots = np.roots([1, b-1, a-3*b*b-2*b, -a*b+b*b+b**3])
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > b]
        if not real_roots:
            raise ValueError('No admissible gas root')
        result.append(max(real_roots))
    return np.array(result).reshape(A.shape)


def pressure_tv(T, v, info):
    """Explicit PR pressure P(T,v), v in m³/mol."""
    T, v = np.broadcast_arrays(np.asarray(T, float), np.asarray(v, float))
    a, _, b = attraction(T, info)
    if np.any(~np.isfinite(T)) or np.any(~np.isfinite(v)) or np.any(T<=0) or np.any(v<=b):
        raise ValueError('Finite T > 0 and v > b are required')
    return R*T/(v-b)-a/(v*v+2*b*v-b*b)


def _primitive(T, coefficients):
    """Primitives of Cp and Cp/T using the five NASA7 Cp coefficients."""
    a0,a1,a2,a3,a4 = coefficients[:5]
    h = R*(a0*T+a1*T**2/2+a2*T**3/3+a3*T**4/4+a4*T**5/5)
    s = R*(a0*np.log(T)+a1*T+a2*T**2/2+a3*T**3/3+a4*T**4/4)
    return h, s


def ideal_integrals(T, info):
    """Integrate Cp and Cp/T from 300 K, continuously across NASA intervals.

    The NASA formation/reference constants a5/a6 are intentionally unused.
    Interval offsets enforce continuous h/s even if source fits have tiny jumps.
    """
    T = np.asarray(T, float)
    ranges = info['nasa_temperature_ranges_K']
    if np.any(T < ranges[0]) or np.any(T > ranges[-1]):
        raise ValueError('Temperature outside the NASA Cp correlation range')
    h, s = np.zeros_like(T), np.zeros_like(T)
    for low, high, coef in zip(ranges[:-1], ranges[1:], info['nasa7_coefficients']):
        upper = np.clip(T, low, high)
        ref = np.clip(T_REF, low, high)
        hu,su = _primitive(upper, coef)
        hr,sr = _primitive(ref, coef)
        h += hu-hr
        s += su-sr
    return h, s


def properties(T, P, gas):
    """Vectorized direct PR properties; T [K], P [Pa], molar outputs in SI."""
    info = parameters(gas) if isinstance(gas, str) else gas
    T,P = np.broadcast_arrays(np.asarray(T,float),np.asarray(P,float))
    tlo,thi = info['T_range_K']; plo,phi = info['P_range_Pa']
    if (np.any(~np.isfinite(T)) or np.any(~np.isfinite(P))
        or np.any(T<tlo) or np.any(T>thi) or np.any(P<plo) or np.any(P>phi)):
        raise ValueError('Outside supported temperature/pressure domain')
    a, da, b = attraction(T, info)
    A = a*P/(R*R*T*T)
    B = b*P/(R*T)
    z = cubic_z(A,B)
    # log1p avoids subtracting nearly equal logarithms at low density.
    L = np.log1p(2*SQRT2*B/(z+(1-SQRT2)*B))
    hres = R*T*(z-1)+(T*da-a)/(2*SQRT2*b)*L
    sres = R*np.log(z-B)+da/(2*SQRT2*b)*L
    hig,sig = ideal_integrals(T,info)
    h = hig+hres
    s = sig-R*np.log(P/P_REF)+sres
    ln_phi = z-1-np.log(z-B)-A/(2*SQRT2*B)*L
    phi = np.exp(ln_phi)
    v = z*R*T/P
    return dict(Z=z,v_m3_mol=v,rho_kg_m3=info['molar_mass_kg_mol']/v,
                h_J_mol=h,s_J_mol_K=s,h_departure_J_mol=hres,
                s_departure_J_mol_K=sres, phi=phi, fugacity_Pa=phi*P)
