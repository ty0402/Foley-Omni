import torch
from xfuser.core.distributed.group_coordinator import GroupCoordinator
# 从本地导入_AllGather实现
from .all_gather_with_grad import _AllGather

def patch_all_gather():
    """
    对xfuser包中的GroupCoordinator.all_gather方法应用猴子补丁，
    使用支持自动求导的_AllGather.apply实现
    """
    original_all_gather = GroupCoordinator.all_gather
    
    def patched_all_gather(
        self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
    ):
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        input_size = list(input_.size())
        input_size[0] *= world_size
        output_tensor = torch.empty(
            input_size, dtype=input_.dtype, device=input_.device
        )
        # 使用支持自动求导的_AllGather实现
        tensor_list = _AllGather.apply(self.device_group, input_)
        output_tensor = torch.cat(tensor_list, dim=0)

        if dim != 0:
            input_size[0] //= world_size
            output_tensor = output_tensor.reshape([world_size, ] + input_size)
            output_tensor = output_tensor.movedim(0, dim)

        if separate_tensors:
            tensor_list = [
                output_tensor.view(-1)
                .narrow(0, input_.numel() * i, input_.numel())
                .view_as(input_)
                for i in range(world_size)
            ]
            return tensor_list
        else:
            input_size = list(input_.size())
            input_size[dim] = input_size[dim] * world_size
            # Reshape
            output_tensor = output_tensor.reshape(input_size)
            return output_tensor
    
    # 应用补丁
    GroupCoordinator.all_gather = patched_all_gather 