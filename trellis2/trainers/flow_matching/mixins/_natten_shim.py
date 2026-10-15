# Pure-torch, memory-efficient replacement for the `natten` package so the real
# NAF model runs on AMD ROCm. NAF legacy_attention layout is heads_first:
# (B, heads, H, W, head_dim). Attention is computed in row-chunks + bf16 to fit VRAM.
import os
import sys
import types
import importlib.machinery
import torch
import torch.nn as nn

_LAYOUT = os.environ.get("TRELLIS2_NATTEN_LAYOUT", "heads_first").strip().lower()
_DEBUG = os.environ.get("TRELLIS2_NATTEN_DEBUG", "0") == "1"
_CHUNK = int(os.environ.get("TRELLIS2_NAF_CHUNK", "64"))
_first = {"qk": True, "av": True}


def _pair(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _offsets(H, W, kH, kW, dH, dW, device):
    ar = torch.arange
    out = []
    for a in range(kH):
        oh = (a - kH // 2) * dH
        hs = ar(H, device=device) + oh
        vh = (hs >= 0) & (hs < H)
        hsc = hs.clamp(0, H - 1)
        for b in range(kW):
            ow = (b - kW // 2) * dW
            ws = ar(W, device=device) + ow
            vw = (ws >= 0) & (ws < W)
            wsc = ws.clamp(0, W - 1)
            out.append((hsc, wsc, vh, vw, a, b))
    return out


def na2d_qk(query, key, kernel_size, dilation=1, rpb=None, is_causal=False, **kw):
    # query/key: (B, heads, H, W, D) -> logits (B, heads, H, W, K) in FP32
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
    # [natten-shim] FORCE FP32
    #
    # gfx1030 / RX 6800 has no native BF16 arithmetic.
    # Keep neighborhood attention entirely in FP32.
    qf = query.float()
    kf = key.float()
    B, hd, H, W, D = qf.shape
    K = kH * kW
    cd = torch.float32
    logits = torch.empty(B, hd, H, W, K, device=qf.device, dtype=cd)
    offs = _offsets(H, W, kH, kW, dH, dW, qf.device)
    for h0 in range(0, H, _CHUNK):
        h1 = min(h0 + _CHUNK, H)
        qc = qf[:, :, h0:h1].to(cd)
        for o, (hsc, wsc, vh, vw, a, b) in enumerate(offs):
            kg = kf[:, :, hsc[h0:h1]][:, :, :, wsc].to(cd)
            dot = (qc * kg).sum(-1)
            if rpb is not None:
                dot = dot + rpb[:, (a - kH // 2) + (kH - 1), (b - kW // 2) + (kW - 1)].view(1, hd, 1, 1)
            dot = dot.masked_fill(~vh[h0:h1][:, None] | ~vw[None, :], float("-inf"))
            logits[:, :, h0:h1, :, o] = dot
    if _DEBUG and _first["qk"]:
        print("[natten-shim] na2d_qk q=%s K=%d dilation=(%d,%d) chunk=%d (FP32)" % (tuple(query.shape), K, dH, dW, _CHUNK))
        _first["qk"] = False
    return logits


def na2d_av(attn, value, kernel_size, dilation=1, rpb=None, **kw):
    # attn: (B, heads, H, W, K) [softmaxed]  value: (B, heads, H, W, D) -> (B, heads, H, W, D)
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
    af = attn.float()
    vf = value.float()
    B, hd, H, W, D = vf.shape
    out = torch.zeros(B, hd, H, W, D, device=vf.device, dtype=vf.dtype)
    offs = _offsets(H, W, kH, kW, dH, dW, vf.device)
    for h0 in range(0, H, _CHUNK):
        h1 = min(h0 + _CHUNK, H)
        ac = af[:, :, h0:h1]
        acc = torch.zeros(B, hd, h1 - h0, W, D, device=vf.device, dtype=vf.dtype)
        for o, (hsc, wsc, vh, vw, a, b) in enumerate(offs):
            valid = (vh[h0:h1][:, None] & vw[None, :]).to(vf.dtype).view(1, 1, h1 - h0, W, 1)
            vg = vf[:, :, hsc[h0:h1]][:, :, :, wsc] * valid
            w = ac[:, :, :, :, o:o + 1].to(vf.dtype)
            acc = acc + w * vg
        out[:, :, h0:h1] = acc
    if _DEBUG and _first["av"]:
        print("[natten-shim] na2d_av v=%s chunk=%d" % (tuple(value.shape), _CHUNK))
        _first["av"] = False
    return out


class _ShimNA2D(nn.Module):
    def __init__(self, dim=None, num_heads=None, kernel_size=7, dilation=1, *a, **k):
        super().__init__()
        self.dim = k.get("dim", dim)
        self.num_heads = k.get("num_heads", num_heads)
        if self.dim is None or self.num_heads is None:
            raise TypeError("[natten-shim] _ShimNA2D needs dim & num_heads (got %s,%s)" % (self.dim, self.num_heads))
        self.head_dim = self.dim // self.num_heads
        self.kernel_size = k.get("kernel_size", kernel_size)
        self.dilation = k.get("dilation", dilation)
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == self.dim:
            bchw = True
            B, C, H, W = x.shape
            flat = x.permute(0, 2, 3, 1)
        elif x.dim() == 4 and x.shape[-1] == self.dim:
            bchw = False
            B, H, W, C = x.shape
            flat = x
        else:
            raise ValueError("[natten-shim] _ShimNA2D bad shape %s dim=%s" % (tuple(x.shape), self.dim))
        # Keep qkv projection out of any inherited BF16 path.
        flat = flat.float()

        q, k, v = self.qkv(flat).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous() for t in (q, k, v)]
        attn = na2d_qk(q, k, self.kernel_size, self.dilation).softmax(dim=-1)
        out = na2d_av(attn, v, self.kernel_size, self.dilation).permute(0, 2, 3, 1, 4).reshape(B, H, W, self.dim)
        out = self.proj(out)
        return out.permute(0, 3, 1, 2) if bchw else out


def install():
    if sys.modules.get("natten") is not None and getattr(sys.modules["natten"], "_pure_torch_shim", False):
        return
    mod = types.ModuleType("natten")
    mod._pure_torch_shim = True
    mod.__spec__ = importlib.machinery.ModuleSpec("natten", loader=None)
    mod.__path__ = []
    mod.__version__ = "0.14.0"
    mod.__file__ = __file__
    func = types.ModuleType("natten.functional")
    func.__spec__ = importlib.machinery.ModuleSpec("natten.functional", loader=None)
    func.__file__ = __file__
    func.na2d_qk = na2d_qk
    func.na2d_av = na2d_av
    mod.functional = func
    mod.na2d = _ShimNA2D
    mod.NeighborhoodAttention2D = _ShimNA2D
    for nm in (
        "has_cuda",
        "has_half",
        "has_gemm",
        "has_tf32",
        "has_64bit_indexing",
    ):
        setattr(
            mod,
            nm,
            lambda *a, **k: False,
        )

    mod.has_fp32 = lambda *a, **k: True
    mod.has_bfloat = lambda *a, **k: False
    for nm in ("set_kernel_choice", "set_memory_usage", "set_tiling", "reset_tiling", "reset_kernel_choice", "set_floating_point_ops_mode"):
        setattr(mod, nm, lambda *a, **k: None)
    sys.modules["natten"] = mod
    sys.modules["natten.functional"] = func
    print(
        "[natten-shim] installed pure-torch natten "
        "(layout=%s, chunk=%d, FP32 ONLY)"
        % (_LAYOUT, _CHUNK)
    )
