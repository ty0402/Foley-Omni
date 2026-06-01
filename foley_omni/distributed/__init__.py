# 导入并应用猴子补丁
from .monkey_patches import patch_all_gather

# 在导入模块时自动应用补丁
patch_all_gather()

# 导出其他模块内容
# ... existing code ...
