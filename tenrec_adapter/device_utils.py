# MIT License - see LICENSE for details

"""
设备抽象层。

自动检测并适配 CUDA/MLU/CPU 设备，提供统一的设备管理接口。
支持寒武纪 MLU (torch_mlu) 和 NVIDIA CUDA。
"""

import logging
import os
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


# 设备类型常量
DEVICE_MLU = "mlu"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


def is_mlu_available() -> bool:
    """
    检查寒武纪 MLU 是否可用。
    
    Returns:
        True 如果 MLU 可用
    """
    try:
        import torch_mlu
        return torch.mlu.is_available()
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"MLU 检测失败: {e}")
        return False


def is_cuda_available() -> bool:
    """
    检查 NVIDIA CUDA 是否可用。
    
    Returns:
        True 如果 CUDA 可用
    """
    return torch.cuda.is_available()


def get_device(
    preferred: Optional[str] = None,
    device_id: int = 0,
) -> torch.device:
    """
    获取最佳可用设备。
    
    优先级：用户指定 > MLU > CUDA > CPU
    
    Args:
        preferred: 优先设备类型 ('mlu', 'cuda', 'cpu')
        device_id: 设备 ID
        
    Returns:
        torch.device 对象
    """
    if preferred:
        preferred = preferred.lower()
        if preferred == DEVICE_MLU:
            if is_mlu_available():
                return torch.device(f"{DEVICE_MLU}:{device_id}")
            else:
                logger.warning("MLU 不可用，回退到其他设备")
        elif preferred == DEVICE_CUDA:
            if is_cuda_available():
                return torch.device(f"{DEVICE_CUDA}:{device_id}")
            else:
                logger.warning("CUDA 不可用，回退到其他设备")
        elif preferred == DEVICE_CPU:
            return torch.device(DEVICE_CPU)
    
    # 自动检测
    if is_mlu_available():
        logger.info(f"使用寒武纪 MLU 设备 {device_id}")
        return torch.device(f"{DEVICE_MLU}:{device_id}")
    
    if is_cuda_available():
        logger.info(f"使用 NVIDIA CUDA 设备 {device_id}")
        return torch.device(f"{DEVICE_CUDA}:{device_id}")
    
    logger.info("使用 CPU 设备")
    return torch.device(DEVICE_CPU)


def get_device_count() -> int:
    """
    获取可用加速器数量。
    
    Returns:
        设备数量
    """
    if is_mlu_available():
        return torch.mlu.device_count()
    if is_cuda_available():
        return torch.cuda.device_count()
    return 0


def get_device_name(device_id: int = 0) -> str:
    """
    获取设备名称。
    
    Args:
        device_id: 设备 ID
        
    Returns:
        设备名称字符串
    """
    if is_mlu_available():
        try:
            import torch_mlu
            return torch.mlu.get_device_name(device_id)
        except:
            return f"MLU:{device_id}"
    
    if is_cuda_available():
        return torch.cuda.get_device_name(device_id)
    
    return "CPU"


def get_device_memory(device_id: int = 0) -> Tuple[int, int, int]:
    """
    获取设备内存信息。
    
    Args:
        device_id: 设备 ID
        
    Returns:
        (张量占用, 预留内存, 总内存) 单位为字节
        - 张量占用 (allocated): PyTorch 张量实际使用的显存
        - 预留内存 (reserved): PyTorch 缓存预留的显存（包含空闲缓存块）
        - 总内存 (total): 设备总显存
    """
    if is_mlu_available():
        try:
            import torch_mlu
            allocated = torch.mlu.memory_allocated(device_id)
            reserved = torch.mlu.memory_reserved(device_id)
            total = torch.mlu.get_device_properties(device_id).total_memory
            return allocated, reserved, total
        except Exception:
            return 0, 0, 0
    
    if is_cuda_available():
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        return allocated, reserved, total
    
    return 0, 0, 0


def get_device_memory_system(device_id: int = 0) -> Tuple[int, int]:
    """
    获取设备的系统级显存使用情况（真实占用）。
    
    通过系统命令获取 GPU/MLU 的实际显存占用，比 PyTorch API 更准确。
    
    Args:
        device_id: 设备 ID
        
    Returns:
        (已用显存, 总显存) 单位为字节
    """
    import subprocess
    
    if is_mlu_available():
        try:
            # 使用 cnmon 获取 MLU 显存使用情况
            result = subprocess.run(
                ['cnmon', 'info', '-m'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # 解析 cnmon 输出
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if f'Card {device_id}' in line or f'Device {device_id}' in line:
                        # 查找下一行的内存信息
                        continue
                    if 'MiB' in line and '/' in line:
                        # 格式: "Memory: 60000 MiB / 81920 MiB"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'MiB' in part and i > 0:
                                used_mib = float(parts[i-1])
                            if '/' in parts[i:i+2] if i+1 < len(parts) else False:
                                total_mib = float(parts[i+2]) if i+2 < len(parts) else 0
                        # 简化解析
                        import re
                        match = re.search(r'(\d+)\s*MiB\s*/\s*(\d+)\s*MiB', line)
                        if match:
                            used_bytes = int(match.group(1)) * 1024 * 1024
                            total_bytes = int(match.group(2)) * 1024 * 1024
                            return used_bytes, total_bytes
        except Exception as e:
            logger.debug(f"cnmon 获取显存失败: {e}")
        
        # 回退到 PyTorch API (使用 reserved 更接近真实值)
        try:
            import torch_mlu
            reserved = torch.mlu.memory_reserved(device_id)
            total = torch.mlu.get_device_properties(device_id).total_memory
            return reserved, total
        except Exception:
            pass
    
    if is_cuda_available():
        try:
            # 使用 nvidia-smi 获取 CUDA 显存使用情况
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                 '--format=csv,nounits,noheader', f'--id={device_id}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]
                used, total = map(int, line.split(','))
                return used * 1024 * 1024, total * 1024 * 1024  # MiB -> bytes
        except Exception as e:
            logger.debug(f"nvidia-smi 获取显存失败: {e}")
        
        # 回退到 PyTorch API
        reserved = torch.cuda.memory_reserved(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        return reserved, total
    
    return 0, 0


def to_device(
    tensor_or_module: Union[torch.Tensor, torch.nn.Module],
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, torch.nn.Module]:
    """
    将 tensor 或 module 移动到指定设备。
    
    Args:
        tensor_or_module: PyTorch tensor 或 module
        device: 目标设备，None 则自动选择
        
    Returns:
        移动后的 tensor 或 module
    """
    if device is None:
        device = get_device()
    
    return tensor_or_module.to(device)


def synchronize():
    """
    同步设备（等待所有操作完成）。
    """
    if is_mlu_available():
        torch.mlu.synchronize()
    elif is_cuda_available():
        torch.cuda.synchronize()


def empty_cache():
    """
    清空设备缓存。
    """
    if is_mlu_available():
        torch.mlu.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()


def set_device(device_id: int):
    """
    设置当前设备。
    
    Args:
        device_id: 设备 ID
    """
    if is_mlu_available():
        torch.mlu.set_device(device_id)
    elif is_cuda_available():
        torch.cuda.set_device(device_id)


class DeviceContext:
    """
    设备上下文管理器。
    
    用于临时切换设备。
    
    Example:
        with DeviceContext(device_id=1):
            # 在设备 1 上执行操作
            output = model(input)
    """
    
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.prev_device = None
    
    def __enter__(self):
        if is_mlu_available():
            self.prev_device = torch.mlu.current_device()
            torch.mlu.set_device(self.device_id)
        elif is_cuda_available():
            self.prev_device = torch.cuda.current_device()
            torch.cuda.set_device(self.device_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev_device is not None:
            if is_mlu_available():
                torch.mlu.set_device(self.prev_device)
            elif is_cuda_available():
                torch.cuda.set_device(self.prev_device)
        return False


def print_device_info():
    """
    打印设备信息摘要。
    """
    print("=" * 50)
    print("设备信息")
    print("=" * 50)
    
    if is_mlu_available():
        print(f"设备类型: 寒武纪 MLU")
        count = get_device_count()
        print(f"设备数量: {count}")
        for i in range(count):
            name = get_device_name(i)
            allocated, reserved, total = get_device_memory(i)
            sys_used, sys_total = get_device_memory_system(i)
            print(f"  MLU:{i} - {name}")
            print(f"    显存: {sys_used/1e9:.2f} GB / {sys_total/1e9:.2f} GB")
    elif is_cuda_available():
        print(f"设备类型: NVIDIA CUDA")
        count = get_device_count()
        print(f"设备数量: {count}")
        for i in range(count):
            name = get_device_name(i)
            allocated, reserved, total = get_device_memory(i)
            sys_used, sys_total = get_device_memory_system(i)
            print(f"  CUDA:{i} - {name}")
            print(f"    显存: {sys_used/1e9:.2f} GB / {sys_total/1e9:.2f} GB")
    else:
        print(f"设备类型: CPU")
        print("无加速器可用")
    
    print("=" * 50)


if __name__ == "__main__":
    print_device_info()
