import torch

from pl_flow.alignment.objective import maximum_path


def test_monotonic_kernel_does_not_mutate_attention():
    scores = torch.rand(1, 3, 8, requires_grad=True)
    before = scores.detach().clone()
    path = maximum_path(scores, torch.ones_like(scores))
    torch.testing.assert_close(scores, before, atol=0, rtol=0)
    assert int(path.sum()) == 8
    assert torch.all(path.sum(-1) >= 1)
    (scores - path).abs().mean().backward()
    assert torch.isfinite(scores.grad).all()
