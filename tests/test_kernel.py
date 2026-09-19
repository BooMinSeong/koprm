import numpy as np

from koprm.label.kernel import RefDist, kernel


def test_example_2_5():
    """Plan §2.5: candidates 3 (p'=0.2), 6 (p'=0.06), 8 (p'=0.4) on an 8-step wrong solution."""
    # Build a fake ref dist whose tail_prob returns given values by monkeypatching.
    class Ref(RefDist):
        def tail_prob(self, d):
            pi = np.ones(len(d))
            pi[2] = 0.2 * 0.05
            pi[5] = 0.06 * 0.05
            pi[7] = 0.4 * 0.05
            return pi

    ref = Ref(b_min=-1e9, drops=np.array([0.0]))
    out = kernel(np.zeros(8), y=0, ref=ref)
    assert out.candidates == [2, 5, 7]
    np.testing.assert_allclose(out.r_lo[:2], [0.995, 0.995], atol=1e-3)
    np.testing.assert_allclose(out.r_lo[2:5], [0.195] * 3, atol=1e-3)
    np.testing.assert_allclose(out.r_hi[5:7], [0.012] * 2, atol=1e-3)
    assert out.labels == [1, 1, None, None, None, 0, 0, 0]


def test_no_candidates():
    ref = RefDist(b_min=0.0, drops=np.linspace(0, 5, 100))
    out = kernel(np.full(5, 3.0), y=0, ref=ref)
    assert out.labels == [None, None, None, None, 0]
    out1 = kernel(np.full(5, 3.0), y=1, ref=ref)
    assert out1.labels == [1] * 5


def test_fit_refdist():
    rng = np.random.default_rng(0)
    zs = [rng.normal(5, 1, size=8) for _ in range(50)]
    ref = RefDist.fit(zs)
    assert ref.drops.ndim == 1 and len(ref.drops) == 400
    z = np.array([5, 5, 5, -10, -10, -10])
    out = kernel(z, y=0, ref=ref)
    assert out.labels[:2] == [1, 1] and out.labels[-1] == 0 and out.labels[3] == 0
