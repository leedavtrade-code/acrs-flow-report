"""python -m unittest ml/test_acrs_ml.py"""
import math
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import acrs_ml as m  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def trade(into, out, r, setup='C', hold=5, **f):
    t = {'into': into, 'out': out, 'realized_R': r, 'setup': setup, 'hold': hold, 'exit_cause': 'MA50_CLOSE'}
    t.update({k: f.get(k, 1.0) for k in m.FEATURES})
    return t


class Stats(unittest.TestCase):
    def test_psr_at_its_own_benchmark_is_half(self):
        self.assertAlmostEqual(m.psr(0.1, 0.1, 500, -0.5, 6.0), 0.5)

    def test_expected_max_sr(self):
        self.assertEqual(m.expected_max_sr(1, 0.01), 0.0)
        a, b = m.expected_max_sr(10, 0.01), m.expected_max_sr(100, 0.01)
        self.assertTrue(0 < a < b)
        # Bailey & Lopez de Prado: E[max] of N=1000 standard normals ~ 3.26
        self.assertAlmostEqual(m.expected_max_sr(1000, 1.0), 3.26, delta=0.02)

    def test_dsr_falls_as_trials_rise(self):
        rng = np.random.default_rng(0)
        eq = np.cumprod(1 + rng.normal(0.0006, 0.01, 1000)) * 1e4
        curve = [(str(i), v) for i, v in enumerate(eq)]
        d = m.deflated_sharpe(curve, 20)
        vals = [r['dsr'] for r in d['table']]
        self.assertEqual(vals, sorted(vals, reverse=True))
        self.assertAlmostEqual(d['table'][0]['dsr'], d['psr_vs_zero'], places=3)

    def test_rankdata_averages_ties(self):
        self.assertEqual(list(m.rankdata(np.array([3., 1., 3., 2.]))), [2.5, 0., 2.5, 1.])


class Leakage(unittest.TestCase):
    def test_meta_label_trains_only_on_trades_closed_before_the_test_year(self):
        trades = [trade(f'2020-{1 + i % 12:02d}-01', f'2020-{1 + i % 12:02d}-20', (-1) ** i) for i in range(120)]
        # opened in 2020 but closed inside 2021: must not be in the 2021 training set
        trades += [trade('2020-12-01', '2021-03-01', 9.0) for _ in range(5)]
        trades += [trade('2021-02-01', '2021-02-20', 1.0) for _ in range(20)]
        res = m.meta_label(trades)
        self.assertEqual([f['year'] for f in res['folds']], [2021])
        self.assertEqual(res['folds'][0]['n_train'], 120)

    def test_live_pool_is_censored_to_window(self):
        rng = np.random.default_rng(0)
        bt = [trade('2025-01-01', '2025-02-01', -0.5, hold=10) for _ in range(40)]
        bt += [trade('2025-01-01', '2025-09-01', 40.0, hold=150) for _ in range(10)]
        live = [trade('2026-07-02', '2026-07-20', -0.5, hold=10) for _ in range(10)]
        res = m.live_check(bt, live, '2026-07-01', 60, rng, n_boot=2000, n_perm=50)
        self.assertEqual(res['n_pool'], 40)
        self.assertEqual(res['p_total_R_this_bad'], 1.0)           # identical to the censored pool
        self.assertLess(res['p_total_R_this_bad_uncensored'], 0.5)  # long winners make it look broken


class EndToEnd(unittest.TestCase):
    def test_build_on_published_simulation(self):
        rep = m.build(ROOT / 'ACRS_Simulation.html', ROOT / 'ml' / 'experiments.json')
        self.assertEqual(rep['post_mortem']['n'], len(m.load_embedded(ROOT / 'ACRS_Simulation.html')['trades']))
        self.assertTrue(0 <= rep['deflated_sharpe']['dsr'] <= 1)
        self.assertTrue(rep['findings'])
        self.assertIn('<title>ACRS ML Review</title>', m.render(rep))
        self.assertFalse(math.isnan(rep['meta_label']['models']['ridge']['mean_ic']))


if __name__ == '__main__':
    unittest.main()
