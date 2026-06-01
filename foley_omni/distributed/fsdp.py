# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy,CPUOffload
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


# def shard_model(
#     model,
#     device_id,
#     param_dtype=torch.bfloat16,
#     reduce_dtype=torch.float32,
#     buffer_dtype=torch.float32,
#     process_group=None,
#     sharding_strategy=ShardingStrategy.FULL_SHARD,
#     sync_module_states=True,
# ):
#     model = FSDP(
#         module=model,
#         process_group=process_group,
#         sharding_strategy=sharding_strategy,
#         auto_wrap_policy=partial(
#             lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
#         mixed_precision=MixedPrecision(
#             param_dtype=param_dtype,
#             reduce_dtype=reduce_dtype,
#             buffer_dtype=buffer_dtype),
#         device_id=device_id,
#         sync_module_states=sync_module_states,
#         # cpu_offload=CPUOffload(offload_params=True),
#         )
#     return model

def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    # --- FSDP 包装策略 ---
    # FSDP 策略需要一个包含所有要包装的模块的列表
    # MusicModel 没有 .blocks，但它的子模块有
    # 我们将它们合并到一个列表中
    all_blocks_to_wrap = []
    
    if hasattr(model, 'audio_model') and hasattr(model.audio_model, 'blocks'):
        all_blocks_to_wrap.extend(list(model.audio_model.blocks))
        print(f"找到 {len(model.audio_model.blocks)} 个 audio blocks 用于 FSDP 包装。")
        
    if not all_blocks_to_wrap:
        print("警告: FSDP 在 'model.video_model.blocks' 或 'model.audio_model.blocks' 中没有找到任何可包装的块。")

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, 
            # --- 修改这一行，使用合并后的列表 ---
            lambda_fn=lambda m: m in all_blocks_to_wrap
        ),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        # cpu_offload=CPUOffload(offload_params=True),
        )
    return model