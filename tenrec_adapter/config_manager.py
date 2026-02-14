# MIT License - see LICENSE for details

"""
配置管理模块。

支持 YAML 配置加载、合并和命令行覆盖。
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    配置管理类。

    支持：
    - YAML 配置文件加载
    - 配置继承 (_base_ 字段)
    - 命令行参数覆盖
    - 点号访问 (config.model.embed_dim)
    """

    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)

        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径。"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return self._config.copy()

    def update(self, other: Dict):
        """递归更新配置。"""
        self._config = self._deep_merge(self._config, other)

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """递归合并字典。"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str, base_dir: Optional[str] = None) -> Config:
    """
    加载 YAML 配置文件。

    支持 _base_ 字段指定父配置文件进行继承。

    Args:
        config_path: 配置文件路径
        base_dir: 基础目录（用于解析相对路径）

    Returns:
        Config 对象
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    if base_dir is None:
        base_dir = config_path.parent

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}

    # 处理继承
    if '_base_' in config_dict:
        base_path = Path(base_dir) / config_dict.pop('_base_')
        base_config = load_config(base_path, base_dir)
        base_config.update(config_dict)
        return base_config

    return Config(config_dict)


def merge_args_to_config(config: Config, args: argparse.Namespace) -> Config:
    """
    将命令行参数合并到配置中。

    Args:
        config: 配置对象
        args: 命令行参数

    Returns:
        合并后的 Config 对象
    """
    args_dict = vars(args)

    # 过滤掉 None 值（未指定的参数）
    args_dict = {k: v for k, v in args_dict.items() if v is not None}

    # 映射参数到配置结构
    overrides = {}

    # 数据参数
    if 'data_dir' in args_dict:
        overrides.setdefault('data', {})['data_dir'] = args_dict['data_dir']
    if 'scenario' in args_dict:
        overrides.setdefault('data', {})['scenario'] = args_dict['scenario']
    if 'max_users' in args_dict:
        overrides.setdefault('data', {})['max_users'] = args_dict['max_users']
    if 'user_attr_file' in args_dict:
        overrides.setdefault('data', {})['user_attr_file'] = args_dict['user_attr_file']
    if 'item_attr_file' in args_dict:
        overrides.setdefault('data', {})['item_attr_file'] = args_dict['item_attr_file']

    # 模型参数
    if 'embed_dim' in args_dict:
        overrides.setdefault('model', {})['embed_dim'] = args_dict['embed_dim']
    if 'hidden_dim' in args_dict:
        overrides.setdefault('model', {})['hidden_dim'] = args_dict['hidden_dim']
    if 'history_len' in args_dict:
        overrides.setdefault('model', {})['history_len'] = args_dict['history_len']

    # 训练参数
    if 'epochs' in args_dict:
        overrides.setdefault('training', {})['epochs'] = args_dict['epochs']
    if 'batch_size' in args_dict:
        overrides.setdefault('training', {})['batch_size'] = args_dict['batch_size']
    if 'lr' in args_dict:
        overrides.setdefault('training', {})['learning_rate'] = args_dict['lr']

    # 设备参数
    if 'device' in args_dict:
        overrides.setdefault('device', {})['type'] = args_dict['device']
    if 'use_amp' in args_dict:
        overrides.setdefault('device', {})['use_amp'] = args_dict['use_amp']


    # 分布式训练参数
    if 'use_ddp' in args_dict:
        overrides.setdefault('training', {})['use_ddp'] = args_dict['use_ddp']
    if 'local_rank' in args_dict:
        overrides.setdefault('training', {})['local_rank'] = args_dict['local_rank']

    # 其他
    if 'seed' in args_dict:
        overrides['seed'] = args_dict['seed']

    config.update(overrides)
    return config


def get_default_config() -> Config:
    """获取默认配置。"""
    default_config = {
        'data': {
            'data_dir': 'data/tenrec/Tenrec',
            'scenario': 'QB-video',
            'user_attr_file': 'user_attr.csv',
            'item_attr_file': 'video_attr.csv',
            'max_users': None,
            'use_cache': True,
            'num_workers': 0,
        },
        'model': {
            'embed_dim': 64,
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'history_len': 64,
            'num_actions': 4,  # Tenrec: click, like, share, follow
            'dropout': 0.1,
            'temperature': 0.07,
        },
        'training': {
            'epochs': 10,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'grad_accumulation': 1,
            'max_grad_norm': 1.0,
            'num_negatives': 4,
        },
        'device': {
            'type': 'auto',
            'use_amp': True,
            'amp_dtype': 'float16',
        },
        'logging': {
            'log_interval': 50,
            'eval_interval': 500,
            'use_tensorboard': True,
            'tensorboard_dir': 'runs',
        },
        'checkpoint': {
            'save_dir': 'checkpoints',
            'save_interval': 1000,
            'keep_last_n': 3,
        },
        'seed': 42,
    }
    return Config(default_config)
