"""
"""
import numpy as np
from ..negative_binomial_sats import MBK10Sats, AssembiasMBK10Sats
from ... import model_defaults


def enforce_mc_occupation_behavior(model):
    mh = 1e15
    correct_mu = model.mean_occupation(prim_haloprop=mh)

    nh = int(1e5)
    mharr = np.zeros(nh) + mh
    nsat_arr = model.mc_occupation(prim_haloprop=mharr, seed=0)

    mean_nsat_sample = np.mean(nsat_arr)
    assert np.allclose(mean_nsat_sample, correct_mu, rtol=0.1)

    p = model.non_poissonian_p(prim_haloprop=mh)
    n = correct_mu * p / (1 - p)
    correct_variance = n * (1 - p) / p / p
    correct_std = np.sqrt(correct_variance)
    std_nsat_sample = np.std(nsat_arr)
    assert np.allclose(correct_std, std_nsat_sample, rtol=0.1)


def enforce_mc_occupation_behavior_assembias(model):
    mh = 1e15
    rng = np.random.RandomState(0)

    ntests = 10
    percentile_tests = rng.uniform(0, 1, ntests)
    for percentile in percentile_tests:
        correct_mu = model.mean_occupation(
            prim_haloprop=mh, sec_haloprop_percentile=percentile
        )

        nh = int(1e5)
        mharr = np.zeros(nh) + mh
        percentile_arr = np.zeros(nh) + percentile
        nsat_arr = model.mc_occupation(
            prim_haloprop=mharr, sec_haloprop_percentile=percentile_arr, seed=0
        )

        mean_nsat_sample = np.mean(nsat_arr)
        assert np.allclose(mean_nsat_sample, correct_mu, rtol=0.1)

        p = model.non_poissonian_p(prim_haloprop=mh)
        n = correct_mu * p / (1 - p)
        correct_variance = n * (1 - p) / p / p
        correct_std = np.sqrt(correct_variance)
        std_nsat_sample = np.std(nsat_arr)
        assert np.allclose(correct_std, std_nsat_sample, rtol=0.1)


def test_default_mbk10_model():

    default_model = MBK10Sats()
    # First test the model with all default settings
    enforce_mc_occupation_behavior(default_model)


def test_default_assembias_mbk10_model():

    model = AssembiasMBK10Sats()
    # First test the model with all default settings
    enforce_mc_occupation_behavior_assembias(model)


def test_mc_occupation_stats_alternate_mbk10_models():
    model = MBK10Sats()
    up0_arr = np.linspace(-100, 100, 50)
    for up0 in up0_arr:
        model.param_dict["nsat_up0"] = up0
        enforce_mc_occupation_behavior(model)


def test_expected_behavior_of_poisson_deviations_of_mbk10():
    model = MBK10Sats()
    up0_arr = np.linspace(-100, 100, 50)
    m = 1e14

    mu_collector = []
    x_collector = []
    for up0 in up0_arr:
        model.param_dict["nsat_up0"] = up0
        var = model.std_occupation(prim_haloprop=m) ** 2
        mu = model.mean_occupation(prim_haloprop=m)
        x = var / mu
        mu_collector.append(mu[0])
        x_collector.append(x[0])
    mu_collector = np.array(mu_collector)
    x_collector = np.array(x_collector)
    assert np.allclose(mu_collector, mu_collector[0])
    assert np.all(np.diff(x_collector) <= 0)
    assert np.allclose(x_collector[-1], 1.0, atol=0.01)
    assert np.all(np.isfinite(x_collector))


def test_expected_behavior_of_poisson_deviations_of_assembias_mbk10():
    model = AssembiasMBK10Sats()
    up0_arr = np.linspace(-100, 100, 50)
    m = 1e14
    perc = 0.5

    mu_collector = []
    x_collector = []
    for up0 in up0_arr:
        model.param_dict["nsat_up0"] = up0
        var = model.std_occupation(prim_haloprop=m, sec_haloprop_percentile=perc) ** 2
        mu = model.mean_occupation(prim_haloprop=m, sec_haloprop_percentile=perc)
        x = var / mu
        mu_collector.append(mu[0])
        x_collector.append(x[0])
    mu_collector = np.array(mu_collector)
    x_collector = np.array(x_collector)
    assert np.allclose(mu_collector, mu_collector[0])
    assert np.all(np.diff(x_collector) <= 0)
    assert np.allclose(x_collector[-1], 1.0, atol=0.01)
    assert np.all(np.isfinite(x_collector))


def test_correct_instantiation():
    model = MBK10Sats()
    assert model.threshold == model_defaults.default_luminosity_threshold

    model = MBK10Sats(threshold=-19)
    assert model.threshold == -19
