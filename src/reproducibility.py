"""集中设置 Python、NumPy 与 PyTorch 随机状态。"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定所有随机源，并按配置选择确定性或高性能 cuDNN 模式。

    非确定性模式仍固定伪随机种子，但 cuDNN benchmark 可能选择不同内核；
    该选项由实验配置记录，不能将两种模式的结果视作完全同协议复现。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
