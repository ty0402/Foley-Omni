import torch
import torch.distributed as dist
from torch.autograd import Function

class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        ctx.group = group
        tensor = tensor.contiguous()
        world_size = dist.get_world_size(group=group)
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        group = ctx.group
        if dist.get_backend(group=group) is dist.Backend.NCCL:
            rank = dist.get_rank(group=group)
            gx = torch.empty_like(grad_outputs[rank])
            dist.reduce_scatter(gx, list(grad_outputs), op=dist.ReduceOp.SUM, group=group)
        else:
            # 对于非 NCCL 后端，使用 AlltoAll 模拟 ReduceScatter
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = dist.all_to_all(tensor_list, list(grad_outputs), group=group)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)
    
