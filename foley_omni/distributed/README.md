# 分布式模块补丁说明

## all_gather自动求导补丁

本模块包含了对xfuser包中`GroupCoordinator.all_gather`方法的补丁，使其支持自动求导功能。

### 问题背景

原始`xfuser`包中的`all_gather`函数使用了标准的`torch.distributed.all_gather_into_tensor`实现，该实现不支持梯度回传。在模型训练过程中，特别是在`xdit_context_parallel.py`中调用`get_sp_group().all_gather(x, dim=1)`时，这会导致梯度计算中断。

### 解决方案

我们实现了以下解决方案：

1. 创建了支持自动求导的`_AllGather`函数 (`all_gather_with_grad.py`)
2. 使用猴子补丁技术在运行时替换原始实现 (`monkey_patches.py`)
3. 在模块初始化时自动应用补丁 (`__init__.py`)

### 使用方法

补丁会在导入`wan.distributed`模块时自动应用，无需额外操作。

```python
# 正常导入即可
import wan.distributed

# 使用all_gather，自动使用支持梯度的版本
x = get_sp_group().all_gather(x, dim=1)
```

### 注意事项

1. 此补丁只影响项目内部使用，不会修改全局安装的`xfuser`包
2. 如果`xfuser`包未来版本修复了此问题，可以移除此补丁 