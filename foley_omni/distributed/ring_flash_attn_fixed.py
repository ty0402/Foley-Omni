from yunchang.ring.ring_flash_attn import ring_flash_attn_backward

def custom_backward(ctx, dout, *args):
    q, k, v, out, softmax_lse = ctx.saved_tensors
    dq, dk, dv = ring_flash_attn_backward(
        ctx.group,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        softmax_scale=ctx.softmax_scale,
        dropout_p=ctx.dropout_p,
        causal=ctx.causal,
        window_size=ctx.window_size,
        # softcap=ctx.softcap,  # 修补，不需要输入这个参数
        alibi_slopes=ctx.alibi_slopes,
        deterministic=ctx.deterministic,
        attn_type=ctx.attn_type,
    )
    return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None