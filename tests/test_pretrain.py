import numpy as np
import torch

from riskembed.encoder.pretrain import info_nce, sample_crop, vicreg_terms


def test_sample_crop_bounds():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 13, 400):
        for _ in range(200):
            a, b = sample_crop(n, rng, 3, 13)
            assert 0 <= a < b <= n and b - a <= min(13, n) and b - a >= min(3, n)


def test_info_nce_prefers_aligned_views():
    torch.manual_seed(0)
    z = torch.randn(64, 16)
    aligned = info_nce(z, z + 0.01 * torch.randn_like(z), 0.1)
    shuffled = info_nce(z, z[torch.randperm(64)], 0.1)
    assert aligned < 0.1 < shuffled


def test_vicreg_terms():
    torch.manual_seed(0)
    independent = torch.randn(4096, 8)  # std = 1, компоненты независимы
    var_l, cov_l = vicreg_terms(independent)
    assert var_l.item() < 0.01 and cov_l.item() < 0.01
    correlated = torch.randn(4096, 1).repeat(1, 8) + 0.1 * torch.randn(4096, 8)  # одна компонента 8 раз
    assert vicreg_terms(correlated)[1].item() > 5.0
    collapsed = 0.01 * torch.randn(4096, 8)  # дисперсия схлопнулась
    assert vicreg_terms(collapsed)[0].item() > 0.95
