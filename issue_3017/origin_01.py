import triton
import triton.language as tl
import torch


@triton.jit
def test_kernel(
    # Inputs
    k,
    k_stride_i,
    k_stride_d,
    v,
    v_stride_i,
    v_stride_e,
    kv_out,
    kv_out_stride_i,
    kv_out_stride_j,
    kv_out_stride_k,
    # Outputs
    output,
    out_stride_i,
    out_stride_e,
):
    i = tl.arange(0, 16)

    k_loaded = tl.load(k + i[:, None] * k_stride_i + i[None, :] * k_stride_d, )
    v_loaded = tl.load(v + i[:, None] * v_stride_i + i[None, :] * v_stride_e, )

    kv = k_loaded[:, :, None] * v_loaded[:, None, :]

    context = tl.cumsum(kv, axis=0)

    out = tl.max(context, axis=1)

    tl.store(output + i[:, None] * out_stride_i + i[None, :] * out_stride_e, out)


def test_case():
    torch.random.manual_seed(1)

    T = 16
    k = torch.ones((T, T), device='cuda')
    v = torch.ones((T, T), device='cuda')
    out = torch.zeros((T, T), device='cuda')
    kv_out = torch.zeros((T, T, T), device='cuda')

    test_kernel[(1, )](
        k,
        *k.stride(),
        v,
        *v.stride(),
        kv_out,
        *kv_out.stride(),
        out,
        *out.stride(),
    )

    kv = k[:, :, None] * v[:, None, :]
    context = torch.cumsum(kv, dim=0)
    out_cmp, _ = torch.max(context, dim=1)

    print((out - out_cmp).abs().mean())  # 8.2349

    print(out)
    print(out_cmp)


if __name__ == "__main__":
    test_case()
