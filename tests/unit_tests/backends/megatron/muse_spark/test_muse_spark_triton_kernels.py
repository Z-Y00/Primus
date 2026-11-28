###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: the fused gated QKV + Q / K norm + RoPE path of MuseSparkSelfAttention against Megatron's
split / norm / RoPE, and the Triton logit soft-cap (alone and inside the deferred cross entropy) against the
eager ops."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from megatron.core.models.common.embeddings.rope_utils import (  # noqa: E402
    _apply_rotary_pos_emb_bshd,
)

from primus.backends.megatron.core.models.muse_spark import (  # noqa: E402
    muse_spark_model as msm,
)
from primus.backends.megatron.core.transformer.muse_spark.attention import (  # noqa: E402
    MuseSparkQKNorm,
    MuseSparkSelfAttention,
)

CAP, MULT, EPS = 20.0, 0.17677669529663687, 1e-5
DEV = torch.device("cuda")


def _snr(got, want):
    want = want.detach().float()
    noise = float((got.detach().float() - want).norm())
    return math.inf if noise == 0 else -20 * math.log10(noise / float(want.norm()))


# ---------------------------------------------------------------- fused Q / K norm + RoPE
class _Linear(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        return x @ self.w.t(), None


def _attention(G, H, D, hidden, no_rope_freq=None):
    a = object.__new__(MuseSparkSelfAttention)
    torch.nn.Module.__init__(a)
    a.config = SimpleNamespace(
        test_mode=False, num_query_groups=G, no_rope_freq=no_rope_freq, rotary_interleaved=False,
        apply_rope_fusion=False, multi_latent_attention=False,
    )  # fmt: skip
    a.attention_type, a.world_size, a.layer_number = "self", 1, 1
    a.num_attention_heads_per_partition, a.num_query_groups_per_partition = G * H, G
    a.hidden_size_per_attention_head = D
    a.linear_qkv = _Linear((torch.randn(G * (2 * H + 2) * D, hidden, device=DEV) / hidden**0.5).bfloat16())
    a.q_layernorm = MuseSparkQKNorm(None, D, eps=EPS)
    a.k_layernorm = MuseSparkQKNorm(None, D, eps=EPS)
    a.qk_norm_rope_fused = True
    a._qk_norm_rope = None
    return a


def _freqs(S, D):
    inv = 1.0 / (500000.0 ** (torch.arange(0, D, 2, device=DEV, dtype=torch.float32) / D))
    f = torch.outer(torch.arange(S, device=DEV, dtype=torch.float32), inv)
    return torch.cat([f, f], dim=-1)[:, None, None, :]


def _rope_rounded_once(t, freqs):
    """Rotate-half RoPE from fp32 cos / sin, rounded once per output (the fused op's contract)."""
    cos, sin = freqs.cos(), freqs.sin()
    lo, hi = t.float().chunk(2, dim=-1)
    rot = torch.cat((-hi, lo), dim=-1)
    return (t.float() * cos + rot * sin).to(t.dtype)


@pytest.mark.parametrize("rope", [True, False])
def test_fused_qkv_matches_megatron_split_norm_rope(rope):
    torch.manual_seed(0)
    S, B, G, H, D, hidden = 256, 2, 4, 8, 64, 512
    a = _attention(G, H, D, hidden)
    freqs = _freqs(S, D)
    h0 = torch.randn(S, B, hidden, device=DEV).bfloat16()
    outs, grads = [], []
    # Megatron's split / norm / unfused RoPE, the same with the RoPE rounded once, then the fused path
    for mode in ("megatron", "rounded_once", "fused"):
        a.linear_qkv.w.grad = None
        h = h0.clone().requires_grad_()
        a._qk_norm_rope = (freqs if rope else None, rope) if mode == "fused" else None
        q, k, v, gate = a.get_query_key_value_tensors(h, output_gate=True)
        a._qk_norm_rope = None
        if rope and mode != "fused":
            rot = _apply_rotary_pos_emb_bshd if mode == "megatron" else _rope_rounded_once
            q, k = rot(q, freqs), rot(k, freqs)
        torch.manual_seed(1)
        loss = sum((t.float() * torch.randn_like(t.float())).sum() for t in (q, k, v, gate))
        loss.backward()
        outs.append((q, k, v, gate))
        grads.append((h.grad, a.linear_qkv.w.grad.clone()))
    (q0, k0, v0, g0), (qr, kr, _, _), (q1, k1, v1, g1) = outs
    assert q1.shape == q0.shape and k1.shape == k0.shape and g1.shape == g0.shape
    assert torch.equal(v1, v0) and torch.equal(g1, g0)
    for got, once, megatron in ((q1, qr, q0), (k1, kr, k0)):
        assert float((got != once).float().mean()) <= 1e-4
        assert _snr(got, megatron) >= (45 if rope else 60)
    for i in (0, 1):
        assert _snr(grads[2][i], grads[0][i]) >= 45


def test_fused_qkv_context_selection():
    G, H, D = 2, 4, 64
    freqs = _freqs(16, D)
    a = _attention(G, H, D, 64, no_rope_freq=[0])
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": freqs}) == (freqs, True)
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": (freqs, freqs)}) == (freqs, True)
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": (freqs, freqs.clone())}) is None
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": freqs, "packed_seq_params": object()}) is None
    assert a._qk_norm_rope_context((None,), {"rotary_pos_emb": freqs}) is None  # positional call
    a.config.no_rope_freq = [1]
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": None}) == (None, False)
    a.qk_norm_rope_fused = False
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": freqs}) is None


# ---------------------------------------------------------------- Triton soft-cap
def _logits(rows, vocab, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(rows, vocab, device=DEV) * 300).bfloat16()


def _softcap_grad_formula(x, g):
    """The eager soft-cap's gradient, each autograd step rounded to bf16 (aten.tanh_backward rounds 1 - t^2)."""
    r = lambda t: t.bfloat16().float()  # noqa: E731
    t3 = r(torch.tanh(r(r(x.float() * MULT) / CAP)))
    g2 = r(r(g.float() * CAP) * r(1 - t3 * t3))
    return r(r(g2 / CAP) * MULT).bfloat16()


@pytest.mark.parametrize("vocab", [4096, 5000])
def test_fused_softcap_is_bitwise_the_eager_ops(vocab):
    from primus.backends.megatron.core.models.muse_spark.softcap_kernels import fused_softcap

    x = _logits(64, vocab)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    za, zb = fused_softcap(a, MULT, CAP), msm.softcap_logits(b, MULT, CAP)
    assert torch.equal(za, zb)
    g = torch.randn_like(za.float()).bfloat16()
    za.backward(g)
    assert torch.equal(a.grad, _softcap_grad_formula(x, g))
    if x.numel() % 4096 == 0:
        # bf16 aten.tanh_backward is wrong past the last multiple of 4096 elements on gfx1250
        # (ROCm torch 2.14 nightly): compare with eager autograd only where it is right
        zb.backward(g)
        assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("vocab", [5000, 50257])
def test_softcap_cross_entropy_is_bitwise_softcap_then_deferred_ce(vocab):
    from primus.backends.megatron.core.models.muse_spark.softcap_kernels import (
        fused_softcap,
        softcap_cross_entropy,
    )
    from primus.backends.megatron.patches.te_patches.cross_entropy_deferred_patches import (
        DeferredCrossEntropy,
    )

    S, B = 48, 2
    x = _logits(S * B, vocab).view(S, B, vocab)
    target = torch.randint(0, vocab, (S, B), device=DEV)
    target[::7] = -100
    upstream = torch.rand(S, B, device=DEV)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    la = softcap_cross_entropy(a * 1, target, MULT, CAP)  # the backward writes dx over its input
    lb = DeferredCrossEntropy.apply(fused_softcap(b, MULT, CAP), target)
    assert torch.equal(la, lb)
    (la * upstream).sum().backward()
    (lb * upstream).sum().backward()
    assert torch.equal(a.grad, b.grad)


def _model(**cfg):
    m = object.__new__(msm.MuseSparkModel)
    torch.nn.Module.__init__(m)
    m.config = SimpleNamespace(
        use_mup=False, mup_output_mult=1.0, output_logit_multiplier=MULT, final_logit_softcapping=CAP, **cfg
    )
    m.pg_collection = SimpleNamespace(tp=None)
    m.triton_softcap = True
    m._muse_defer_softcap = False
    return m


def test_model_defers_the_softcap_into_the_cross_entropy_once():
    from primus.backends.megatron.core.models.muse_spark.softcap_kernels import (
        fused_softcap,
        softcap_cross_entropy,
    )

    m = _model()
    x = _logits(32 * 2, 3000).view(32, 2, 3000)
    labels = torch.randint(0, 3000, (2, 32), device=DEV)
    # not deferred: the Triton soft-cap, bitwise the eager one
    assert torch.equal(m._scale_logits(x), msm.softcap_logits(x, MULT, CAP))
    m._muse_defer_softcap = True
    tagged = m._scale_logits(x.clone())
    assert not m._muse_defer_softcap and tagged.__dict__["_muse_pending_softcap"] == (MULT, CAP)
    assert torch.equal(m._scale_logits(x), fused_softcap(x, MULT, CAP))  # later calls are not deferred
    loss = m.compute_language_model_loss(labels, tagged)
    assert "_muse_pending_softcap" not in tagged.__dict__
    want = softcap_cross_entropy(x.clone(), labels.t().contiguous(), MULT, CAP).t().contiguous()
    assert loss.shape == (2, 32) and torch.equal(loss, want)


def test_model_cross_entropy_falls_back_for_ineligible_logits(monkeypatch):
    from megatron.core.models.gpt.gpt_model import GPTModel

    seen = []
    monkeypatch.setattr(
        GPTModel, "compute_language_model_loss", lambda self, labels, z: seen.append(z) or z.sum()
    )
    m = _model()
    # not contiguous: the deferred cross entropy does not take it
    x = _logits(16 * 2, 1000).view(16, 2, 1000)[:, :, :999]
    x.__dict__["_muse_pending_softcap"] = (MULT, CAP)
    m.compute_language_model_loss(torch.zeros(2, 16, dtype=torch.long, device=DEV), x)
    assert len(seen) == 1 and torch.equal(seen[0], msm.softcap_logits(x, MULT, CAP))
