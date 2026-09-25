import torch

from revlm import selftest
from revlm.bench import find_max_batch


def test_same_init_all_modes():
    assert selftest.check_same_init() > 0


def test_reconstruction_is_exact():
    for mode, err in selftest.check_reconstruction().items():
        assert err < 1e-9, (mode, err)


def test_memory_saving_gradients_match_autograd():
    for mode, (dl, dg) in selftest.check_gradients().items():
        assert dl < 1e-10 and dg < 1e-8, (mode, dl, dg)


def test_chunked_loss_matches():
    for mode, (dl, dg) in selftest.check_chunked_loss().items():
        assert dl < 1e-10 and dg < 1e-8, (mode, dl, dg)


def test_saved_bytes_flat_in_depth():
    depths, rows = selftest.check_saved_bytes(depths=(2, 8))
    assert rows["midpoint"][0] == rows["midpoint"][1]           # constant in depth
    assert rows["baseline"][1] > 3 * rows["baseline"][0]        # baseline grows


def test_bf16_autocast_path_runs():
    """custom backward must respect autocast (CPU bf16 stands in for GPU fp16/bf16)."""
    from revlm.model import GPT
    for mode in ("midpoint", "leapfrog", "hamiltonian"):
        torch.manual_seed(0)
        cfg = selftest._tiny(mode)
        net = GPT(cfg)
        x, y = selftest._batch(cfg)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = net(x, y)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_max_batch_search_logic():
    for true_max in (8, 24, 100, 517, 4096):
        fake = lambda B, m=true_max: dict(ok=B <= m, peak_mib=B)
        best, _ = find_max_batch(fake, start=8, limit=4096, verbose=False)
        assert best <= true_max and true_max - best < 8 or best == 4096, (true_max, best)
    assert find_max_batch(lambda B: dict(ok=False), start=8, verbose=False)[0] is None
