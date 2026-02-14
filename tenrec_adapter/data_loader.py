# MIT License - see LICENSE for details

"""
Tenrec 数据集加载器。

加载和解析 Tenrec CSV 文件，构建用户历史序列。
支持 Pickle 缓存加速重复加载。
"""

import csv
import hashlib
import logging
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

logger = logging.getLogger(__name__)


def _safe_int(value, default: int = 0) -> int:
    """安全的 int 转换，处理 \\N（MySQL NULL）、空字符串等异常值。"""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    """安全的 float 转换。"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


@dataclass
class TenrecInteraction:
    """单条 Tenrec 交互记录。"""

    user_id: int
    item_id: int
    click: int  # 0 或 1
    like: int   # 0 或 1
    share: int  # 0 或 1
    follow: int  # 0 或 1
    reading_duration: float  # 阅读时长（秒），QB-video 有此字段
    timestamp: int  # Unix 时间戳（QB-video 有，QK-video 按行序合成）

    # Side Information（QK-video / ctr_data_1M 场景可用）
    video_category: int = 0     # 视频类别
    watching_times: int = 0     # 观看次数
    gender: int = 0             # 性别
    age: int = 0                # 年龄段

    # 预构建用户历史（ctr_data_1M 場景，hist_1~hist_10）
    hist_items: List[int] = field(default_factory=list)

    @property
    def has_positive_action(self) -> bool:
        """是否有任何正向交互。"""
        return self.click == 1 or self.like == 1 or self.share == 1 or self.follow == 1


@dataclass
class TenrecUserHistory:
    """用户的历史交互记录。"""

    user_id: int
    interactions: List[TenrecInteraction]

    def get_recent_items(self, max_len: int = 128) -> List[int]:
        """获取最近交互的 item_id 列表。"""
        sorted_interactions = sorted(self.interactions, key=lambda x: x.timestamp, reverse=True)
        return [inter.item_id for inter in sorted_interactions[:max_len]]

    def get_action_matrix(self, item_ids: List[int], num_actions: int = 4) -> np.ndarray:
        """
        获取指定 items 的 action 矩阵。

        Args:
            item_ids: item ID 列表
            num_actions: action 数量 (click, like, share, follow)

        Returns:
            action_matrix: [len(item_ids), num_actions] 的 numpy 数组
        """
        # 构建 item_id -> interaction 的映射
        item_to_interaction = {inter.item_id: inter for inter in self.interactions}

        action_matrix = np.zeros((len(item_ids), num_actions), dtype=np.float32)
        for i, item_id in enumerate(item_ids):
            if item_id in item_to_interaction:
                inter = item_to_interaction[item_id]
                action_matrix[i, 0] = inter.click
                action_matrix[i, 1] = inter.like
                action_matrix[i, 2] = inter.share
                action_matrix[i, 3] = inter.follow

        return action_matrix


# ========== 列式存储（高性能数据结构） ==========

class ColumnarInteractionStore:
    """
    列式交互数据存储。

    将 DataFrame 的每一列直接存为 numpy 数组，避免创建数百万个 Python 对象。
    10GB / 1.2亿行数据从 DataFrame 转入只需 ~10 秒（vs 创建 dataclass 对象需 30-60 分钟）。
    """

    def __init__(self, df: pd.DataFrame, has_hist: bool = False):
        """
        从 pandas DataFrame 零拷贝构建列式存储。

        Args:
            df: 已清洗的 DataFrame（列名已统一、NaN 已填充 0）
            has_hist: 是否包含 hist_1~hist_10 列
        """
        n = len(df)

        # 核心字段
        self.user_ids = df["user_id"].values.astype(np.int32)
        self.item_ids = df["item_id"].values.astype(np.int32)
        self.timestamps = df["timestamp"].values.astype(np.int64)

        # Action 字段
        self.clicks = df["click"].values.astype(np.int8) if "click" in df.columns else np.zeros(n, dtype=np.int8)
        self.likes = df["like"].values.astype(np.int8) if "like" in df.columns else np.zeros(n, dtype=np.int8)
        self.shares = df["share"].values.astype(np.int8) if "share" in df.columns else np.zeros(n, dtype=np.int8)
        self.follows = df["follow"].values.astype(np.int8) if "follow" in df.columns else np.zeros(n, dtype=np.int8)

        # reading_duration
        if "reading_duration" in df.columns:
            self.reading_durations = df["reading_duration"].values.astype(np.float32)
        elif "watching_times" in df.columns:
            self.reading_durations = df["watching_times"].values.astype(np.float32)
        else:
            self.reading_durations = np.zeros(n, dtype=np.float32)

        # Side Information
        self.video_categories = df["video_category"].values.astype(np.int32) if "video_category" in df.columns else np.zeros(n, dtype=np.int32)
        self.watching_times_col = df["watching_times"].values.astype(np.int32) if "watching_times" in df.columns else np.zeros(n, dtype=np.int32)
        self.genders = df["gender"].values.astype(np.int8) if "gender" in df.columns else np.zeros(n, dtype=np.int8)
        self.ages = df["age"].values.astype(np.int8) if "age" in df.columns else np.zeros(n, dtype=np.int8)

        # 预构建历史 (N, 10) 矩阵
        self.has_hist = has_hist
        if has_hist:
            self.hist_matrix = np.stack(
                [df[f"hist_{i}"].values.astype(np.int32) for i in range(1, 11)],
                axis=1,
            )  # shape: (N, 10)
        else:
            self.hist_matrix = None

        self.n_rows = n

        # 预构建用户集合和物品集合
        self.user_set = set(np.unique(self.user_ids).tolist())
        self.item_set = set(np.unique(self.item_ids).tolist())

        # 预构建用户 -> 正向物品集合（用于负采样）
        self._user_positive_items: Optional[Dict[int, set]] = None

    def __len__(self):
        return self.n_rows

    def build_user_positive_items(self) -> Dict[int, set]:
        """构建用户 -> 交互过的物品集合映射（用于负采样排除）。"""
        if self._user_positive_items is not None:
            return self._user_positive_items

        logger.info("构建用户-物品交互映射...")
        start = time.time()

        # 纯 numpy: argsort + unique 分组，避免 pandas groupby 开销
        # pandas groupby().apply(set) 在 1.2 亿行上需要 78s，这里只需 <5s
        sort_idx = np.argsort(self.user_ids)
        sorted_uids = self.user_ids[sort_idx]
        sorted_items = self.item_ids[sort_idx]
        unique_uids, start_positions = np.unique(sorted_uids, return_index=True)
        end_positions = np.append(start_positions[1:], len(sorted_uids))

        self._user_positive_items = {}
        for uid, s, e in zip(unique_uids, start_positions, end_positions):
            self._user_positive_items[int(uid)] = set(sorted_items[s:e].tolist())

        logger.info(f"用户-物品映射完成: {len(self._user_positive_items)} 用户, 耗时 {time.time() - start:.2f}s")
        return self._user_positive_items

    def get_sorted_indices(self) -> np.ndarray:
        """返回按 timestamp 排序的索引数组。"""
        return np.argsort(self.timestamps)

    def get_user_history_indices(self) -> Dict[int, np.ndarray]:
        """
        构建用户 -> 按时间排序的交互索引映射。

        Returns:
            {user_id: np.ndarray of sorted indices}
        """
        logger.info("构建用户历史索引...")
        start = time.time()

        # 先按 timestamp 排序
        sorted_order = np.argsort(self.timestamps)
        sorted_uids = self.user_ids[sorted_order]

        # 按 user_id 分组
        unique_uids, start_positions = np.unique(sorted_uids, return_index=True)
        # 计算每组结束位置
        end_positions = np.append(start_positions[1:], len(sorted_order))

        result = {}
        for uid, s, e in zip(unique_uids, start_positions, end_positions):
            result[int(uid)] = sorted_order[s:e]

        logger.info(f"用户历史索引完成: {len(result)} 用户, 耗时 {time.time() - start:.2f}s")
        return result


class InteractionView:
    """
    列式存储的轻量级视图。

    行为与 TenrecInteraction 完全一致，但不创建 Python 对象，
    而是按需从 numpy 数组读取。创建成本: ~0（仅存储 2 个引用）。
    """
    __slots__ = ('_store', '_idx')

    def __init__(self, store: ColumnarInteractionStore, idx: int):
        self._store = store
        self._idx = idx

    @property
    def user_id(self) -> int:
        return int(self._store.user_ids[self._idx])

    @property
    def item_id(self) -> int:
        return int(self._store.item_ids[self._idx])

    @property
    def click(self) -> int:
        return int(self._store.clicks[self._idx])

    @property
    def like(self) -> int:
        return int(self._store.likes[self._idx])

    @property
    def share(self) -> int:
        return int(self._store.shares[self._idx])

    @property
    def follow(self) -> int:
        return int(self._store.follows[self._idx])

    @property
    def reading_duration(self) -> float:
        return float(self._store.reading_durations[self._idx])

    @property
    def timestamp(self) -> int:
        return int(self._store.timestamps[self._idx])

    @property
    def video_category(self) -> int:
        return int(self._store.video_categories[self._idx])

    @property
    def watching_times(self) -> int:
        return int(self._store.watching_times_col[self._idx])

    @property
    def gender(self) -> int:
        return int(self._store.genders[self._idx])

    @property
    def age(self) -> int:
        return int(self._store.ages[self._idx])

    @property
    def hist_items(self) -> List[int]:
        if self._store.hist_matrix is not None:
            return self._store.hist_matrix[self._idx].tolist()
        return []

    @property
    def has_positive_action(self) -> bool:
        return self.click == 1 or self.like == 1 or self.share == 1 or self.follow == 1


class InteractionSlice:
    """
    交互数据切片。支持 len / __getitem__ / __iter__。

    作为 List[TenrecInteraction] 的 drop-in 替代，
    但底层由 ColumnarInteractionStore + 索引数组驱动。
    """

    def __init__(self, store: ColumnarInteractionStore, indices: np.ndarray):
        self._store = store
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return InteractionSlice(self._store, self._indices[idx])
        if isinstance(idx, (list, np.ndarray)):
            return InteractionSlice(self._store, self._indices[idx])
        return InteractionView(self._store, int(self._indices[idx]))

    def __iter__(self):
        for i in range(len(self._indices)):
            yield InteractionView(self._store, int(self._indices[i]))


class TenrecDataLoader:
    """
    Tenrec 数据集加载器。

    支持加载 QK-video, QK-article, QB-video, QB-article 等场景数据。
    支持 Pickle 缓存加速重复加载。
    """

    # Tenrec CSV 列名映射
    COLUMN_NAMES = [
        "user_id",
        "item_id",
        "click",
        "like",
        "share",
        "follow",
        "reading_duration",
        "timestamp",
    ]

    # v6: Columnar storage (Struct-of-Arrays)
    CACHE_VERSION = 6

    def __init__(
        self,
        data_dir: str,
        scenario: str = "QK-video",
        max_users: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
    ):
        """
        初始化数据加载器。

        Args:
            data_dir: 数据目录路径
            scenario: 场景名称 (QK-video, QK-article, QB-video, QB-article)
            max_users: 最大用户数（用于小规模测试）
            use_cache: 是否使用缓存
            cache_dir: 缓存目录（默认为 data_dir/.cache）
            force_reload: 强制重新加载（忽略缓存）
        """
        self.data_dir = Path(data_dir)
        self.scenario = scenario
        self.max_users = max_users
        self.use_cache = use_cache
        self.force_reload = force_reload

        # 缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.data_dir / ".cache"

        # 列式存储核心
        self.store: Optional[ColumnarInteractionStore] = None
        # 对外表现为 List[TenrecInteraction]，实际是 InteractionSlice
        self.interactions: Union[List[TenrecInteraction], InteractionSlice] = []

        self.user_histories: Dict[int, TenrecUserHistory] = {}
        self.item_set: set = set()
        self.user_set: set = set()
        self.item_category_map: Optional[np.ndarray] = None
        self._negative_pool: Optional[set] = None  # 负采样候选池（防止未来数据泄漏）

    def _get_cache_path(self) -> Path:
        """生成数据缓存文件路径。"""
        if self.max_users is None:
            cache_key = f"{self.scenario}_full_v{self.CACHE_VERSION}"
        else:
            cache_key = f"{self.scenario}_u{self.max_users}_v{self.CACHE_VERSION}"
        return self.cache_dir / f"{cache_key}.pkl"

    def _get_split_cache_path(self, seed: int) -> Path:
        """生成数据划分缓存文件路径。"""
        user_scope = "full" if self.max_users is None else f"u{self.max_users}"
        cache_key = f"{self.scenario}_split_{user_scope}_seed{seed}_v{self.CACHE_VERSION}"
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path: Path, csv_path: Path) -> bool:
        """检查缓存是否有效。"""
        if not cache_path.exists():
            return False

        # 检查缓存是否比源文件新
        cache_mtime = cache_path.stat().st_mtime
        csv_mtime = csv_path.stat().st_mtime

        return cache_mtime > csv_mtime

    def _load_from_cache(self, cache_path: Path) -> bool:
        """从缓存加载数据。"""
        try:
            start_time = time.time()
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # 验证缓存版本
            if cache_data.get("version") != self.CACHE_VERSION:
                logger.info("缓存版本不匹配，重新加载")
                return False
            cache_max_users = cache_data.get("max_users", None)
            if cache_max_users != self.max_users:
                logger.info(
                    f"缓存 max_users 不匹配 (cache={cache_max_users}, current={self.max_users})，重新加载"
                )
                return False

            # 恢复 Store
            self.store = cache_data["store"]
            self.interactions = InteractionSlice(self.store, np.arange(len(self.store)))

            # 恢复其他元数据
            self.user_histories = cache_data["user_histories"]
            self.item_set = cache_data["item_set"]
            self.user_set = cache_data["user_set"]

            elapsed = time.time() - start_time
            logger.info(f"从缓存加载完成: {len(self.interactions)} 条交互, 耗时 {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return False

    def _save_to_cache(self, cache_path: Path):
        """保存数据到缓存。"""
        try:
            os.makedirs(cache_path.parent, exist_ok=True)

            # 保存整个 Store (numpy arrays, 紧凑高效)
            cache_data = {
                "version": self.CACHE_VERSION,
                "store": self.store,
                "user_histories": self.user_histories,
                "item_set": self.item_set,
                "user_set": self.user_set,
                "max_users": self.max_users,
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"缓存已保存: {cache_path} ({cache_size_mb:.1f} MB)")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _get_required_columns(self) -> Set[str]:
        """Columns required by training and optional side features."""
        cols = {
            "user_id", "uid", "item_id", "iid",
            "timestamp",
            "click", "like", "share", "follow",
            "gender", "age",
            "video_category", "category",
        }
        cols.update({f"hist_{i}" for i in range(1, 65)})
        return cols

    def _finalize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric columns after CSV/Parquet loading."""
        for col in df.columns:
            if df[col].dtype in (float, "float64", "float32"):
                df[col] = df[col].fillna(0).astype(int)
            elif str(df[col].dtype).startswith("Int"):
                df[col] = df[col].fillna(0).astype(int)
            elif df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    def _read_parquet_fast(self, parquet_path: Path) -> pd.DataFrame:
        """Read Parquet with optional column pruning."""
        file_size_gb = parquet_path.stat().st_size / (1024**3)
        logger.info(f"使用 Parquet 引擎加载 ({file_size_gb:.1f} GB)...")

        usecols = None
        required_cols = self._get_required_columns()
        try:
            import pyarrow.parquet as pq  # type: ignore
            available_cols = pq.ParquetFile(parquet_path).schema.names
            usecols = [c for c in available_cols if c in required_cols or c.startswith("hist_")]
            if len(usecols) == 0:
                usecols = None
        except Exception:
            pass

        df = pd.read_parquet(parquet_path, columns=usecols)
        df = self._finalize_dataframe_types(df)
        logger.info(f"Parquet 加载完成: {len(df)} 行")
        return df

    def _read_csv_chunked_with_max_users(
        self,
        csv_path: Path,
        usecols: Optional[List[str]],
        dtype_map: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Stream large CSV in chunks and keep only first max_users users.
        This avoids one-shot allocation for very large CSV files.
        """
        assert self.max_users is not None

        chunks: List[pd.DataFrame] = []
        selected_users: Set[int] = set()
        chunksize = 1_000_000
        start_time = time.time()

        reader = pd.read_csv(
            csv_path,
            na_values=["\\N", ""],
            engine="c",
            low_memory=False,
            usecols=usecols if usecols else None,
            dtype=dtype_map if len(dtype_map) > 0 else None,
            memory_map=True,
            chunksize=chunksize,
        )

        kept_rows = 0
        for chunk_idx, chunk in enumerate(reader, start=1):
            if "uid" in chunk.columns and "user_id" not in chunk.columns:
                chunk = chunk.rename(columns={"uid": "user_id"})

            if "user_id" not in chunk.columns:
                continue

            chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce")
            chunk = chunk.dropna(subset=["user_id"])
            if len(chunk) == 0:
                continue
            chunk["user_id"] = chunk["user_id"].astype("int64")

            if len(selected_users) < self.max_users:
                for uid in chunk["user_id"].drop_duplicates().tolist():
                    uid_int = int(uid)
                    if uid_int not in selected_users:
                        selected_users.add(uid_int)
                        if len(selected_users) >= self.max_users:
                            break

            if len(selected_users) > 0:
                chunk = chunk[chunk["user_id"].isin(selected_users)]
                if len(chunk) > 0:
                    kept_rows += len(chunk)
                    chunks.append(chunk)

            if chunk_idx % 20 == 0:
                logger.info(
                    f"CSV 分块加载: chunk={chunk_idx}, selected_users={len(selected_users)}, kept_rows={kept_rows}"
                )

        if len(chunks) == 0:
            logger.warning("CSV 分块加载结果为空")
            df = pd.DataFrame(columns=usecols if usecols else [])
        else:
            df = pd.concat(chunks, axis=0, ignore_index=True)

        df = self._finalize_dataframe_types(df)
        elapsed = time.time() - start_time
        logger.info(
            f"CSV 分块加载完成: users={len(selected_users)}, rows={len(df)}, elapsed={elapsed:.2f}s"
        )
        return df

    def _read_csv_fast(self, csv_path: Path) -> pd.DataFrame:
        """
        高速 CSV 解析。

        优先使用 Polars（Rust 引擎），回退到 pandas（C 引擎）。
        大文件且 max_users 生效时，使用分块过滤降低内存峰值。
        """
        file_size_gb = csv_path.stat().st_size / (1024**3)
        required_cols = self._get_required_columns()

        try:
            available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except Exception:
            available_cols = []

        usecols = [c for c in available_cols if c in required_cols or c.startswith("hist_")]
        if len(usecols) == 0:
            usecols = None

        dtype_map: Dict[str, str] = {}
        if usecols is not None:
            for c in usecols:
                if c in {
                    "user_id", "uid", "item_id", "iid", "timestamp",
                    "click", "like", "share", "follow",
                    "gender", "age", "video_category", "category",
                } or c.startswith("hist_"):
                    dtype_map[c] = "Int32"

        if file_size_gb >= 1.0 and self.max_users is not None:
            logger.info(
                f"使用 pandas 分块加载 CSV ({file_size_gb:.1f} GB), max_users={self.max_users}..."
            )
            return self._read_csv_chunked_with_max_users(csv_path, usecols, dtype_map)

        if HAS_POLARS:
            logger.info(f"使用 Polars (Rust) 引擎解析 CSV ({file_size_gb:.1f} GB)...")
            try:
                df_pl = pl.read_csv(
                    str(csv_path),
                    null_values=["\\N", ""],
                    infer_schema_length=10000,
                    columns=usecols,
                )
                df = df_pl.fill_null(0).to_pandas()
                df = self._finalize_dataframe_types(df)
                logger.info(f"Polars 解析完成: {len(df)} 行")
                return df
            except Exception as e:
                logger.warning(f"Polars 解析失败，回退到 pandas: {e}")

        logger.info(f"使用 pandas (C) 引擎解析 CSV ({file_size_gb:.1f} GB)...")
        df = pd.read_csv(
            csv_path,
            na_values=["\\N", ""],
            engine="c",
            low_memory=False,
            usecols=usecols if usecols else None,
            dtype=dtype_map if len(dtype_map) > 0 else None,
            memory_map=True,
        )

        df = self._finalize_dataframe_types(df)
        logger.info(f"pandas 解析完成: {len(df)} 行")
        return df

    def load(self) -> "TenrecDataLoader":
        """
        加载数据集。

        优先从缓存加载，缓存无效时从 Parquet/CSV 加载并创建缓存。

        Returns:
            self (链式调用)
        """
        parquet_path = self.data_dir / f"{self.scenario}.parquet"
        csv_path = self.data_dir / f"{self.scenario}.csv"
        if parquet_path.exists():
            data_path = parquet_path
        elif csv_path.exists():
            data_path = csv_path
        else:
            raise FileNotFoundError(f"数据文件不存在: {parquet_path} or {csv_path}")

        cache_path = self._get_cache_path()

        # 尝试从缓存加载
        if self.use_cache and not self.force_reload:
            if self._is_cache_valid(cache_path, data_path):
                if self._load_from_cache(cache_path):
                    return self

        logger.info(f"加载 Tenrec 数据: {data_path}")
        start_time = time.time()

        if data_path.suffix == ".parquet":
            df = self._read_parquet_fast(data_path)
        else:
            df = self._read_csv_fast(data_path)

        # 自动检测数据集格式
        columns = set(df.columns)
        has_timestamp = "timestamp" in columns
        has_hist = "hist_1" in columns

        if "video_category" in columns:
            logger.info(f"检测到 Side Information 列")
        if has_hist:
            logger.info(f"检测到预构建历史列: hist_1~hist_10")
        if not has_timestamp:
            logger.info(f"无 timestamp 列，将按行序生成合成时间戳")
            df["timestamp"] = np.arange(len(df))  # 向量化生成合成时间戳

        # 兼容列名: uid -> user_id, iid -> item_id
        if "uid" in columns and "user_id" not in columns:
            df = df.rename(columns={"uid": "user_id"})
        if "iid" in columns and "item_id" not in columns:
            df = df.rename(columns={"iid": "item_id"})

        # max_users 过滤（在 DataFrame 层面高效执行）
        if self.max_users:
            unique_users = df["user_id"].unique()
            if len(unique_users) > self.max_users:
                selected_users = set(unique_users[:self.max_users])
                df = df[df["user_id"].isin(selected_users)]
                logger.info(f"max_users={self.max_users}, 过滤后: {len(df)} 条交互")

        # 确保关键列类型正确
        df["user_id"] = df["user_id"].astype(int)
        df["item_id"] = df["item_id"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)

        parse_time = time.time() - start_time
        logger.info(f"DataFrame 准备就绪，耗时 {parse_time:.2f}s")

        # ===== 构建列式存储 Store =====
        build_start = time.time()
        self.store = ColumnarInteractionStore(df, has_hist=has_hist)

        # 创建全量视图
        self.interactions = InteractionSlice(self.store, np.arange(len(self.store)))

        self.user_set = self.store.user_set
        self.item_set = self.store.item_set

        logger.info(f"Store 构建完成, 耗时 {time.time() - build_start:.2f}s")

        # 构建用户历史 (对于 ctr_data_1M，我们可以跳过昂贵的 history 构建，因为 hist_items 已存在)
        # 仅当没有预构建历史时才构建 (QB-video 等)
        if not has_hist:
            logger.info("构建用户动态历史...")
            user_hist_indices = self.store.get_user_history_indices()
            for uid, indices in user_hist_indices.items():
                self.user_histories[uid] = TenrecUserHistory(
                    user_id=uid,
                    interactions=InteractionSlice(self.store, indices)
                )
        else:
            logger.info("使用预构建历史 (hist_1~hist_10)，跳过动态历史构建")

        elapsed = time.time() - start_time
        logger.info(
            f"加载完成: {len(self.interactions)} 条交互, "
            f"{len(self.user_set)} 用户, {len(self.item_set)} items, "
            f"总耗时 {elapsed:.2f}s"
        )

        # 保存全量缓存
        if self.use_cache:
            self._save_to_cache(cache_path)

        return self

    def _filter_by_max_users(self):
        """
        从已加载的全量数据中筛选前 N 个用户。
        (注意：如果使用 use_cache=True，通常在 load() 内部就会处理 max_users，
         这个方法主要用于从全量缓存加载后进行二次筛选)
        """
        # 暂不支持基于 Store 的后处理筛选（因为 Store 是不可变的 numpy 数组）
        # 如果需要支持，可以重建 Store 或仅筛选 interactions slice
        # 但考虑到 max_users 主要用于开发调试，且 load() 中已有 DataFrame 级筛选，
        # 这里仅实现基于 interactions slice 的筛选（不修改 store）

        # 按用户首次交互时间排序，取前 N 个
        user_first_time = {}
        # 为了速度，直接遍历 numpy 数组
        uids = self.store.user_ids
        ts = self.store.timestamps

        for i in range(len(uids)):
            uid = uids[i]
            t = ts[i]
            if uid not in user_first_time:
                user_first_time[uid] = t
            else:
                if t < user_first_time[uid]:
                    user_first_time[uid] = t

        sorted_users = sorted(user_first_time.keys(), key=lambda u: user_first_time[u])
        selected_users = set(sorted_users[:self.max_users])

        # 过滤 interactions slice
        original_count = len(self.interactions)

        # 找出属于 selected_users 的索引
        valid_mask = np.isin(self.store.user_ids, list(selected_users))
        valid_indices = np.where(valid_mask)[0]

        self.interactions = InteractionSlice(self.store, valid_indices)

        # 更新用户历史
        self.user_histories = {u: h for u, h in self.user_histories.items() if u in selected_users}

        # 更新集合
        self.user_set = selected_users
        self.item_set = {int(self.store.item_ids[i]) for i in valid_indices}

        logger.info(f"用户筛选: {original_count} -> {len(self.interactions)} 条交互 (max_users={self.max_users})")

    def get_statistics(self) -> Dict:
        """获取数据集统计信息。"""
        # 使用 numpy 向量化计算
        if self.store:
            # 如果是全量 slice
            if len(self.interactions) == len(self.store):
                total_clicks = np.sum(self.store.clicks)
                total_likes = np.sum(self.store.likes)
                total_shares = np.sum(self.store.shares)
                total_follows = np.sum(self.store.follows)
            else:
                # 切片统计
                indices = self.interactions._indices
                total_clicks = np.sum(self.store.clicks[indices])
                total_likes = np.sum(self.store.likes[indices])
                total_shares = np.sum(self.store.shares[indices])
                total_follows = np.sum(self.store.follows[indices])
        else:
            total_clicks = sum(1 for i in self.interactions if i.click == 1)
            total_likes = sum(1 for i in self.interactions if i.like == 1)
            total_shares = sum(1 for i in self.interactions if i.share == 1)
            total_follows = sum(1 for i in self.interactions if i.follow == 1)

        return {
            "total_interactions": len(self.interactions),
            "total_users": len(self.user_set),
            "total_items": len(self.item_set),
            "total_clicks": int(total_clicks),
            "total_likes": int(total_likes),
            "total_shares": int(total_shares),
            "total_follows": int(total_follows),
            "click_rate": float(total_clicks) / len(self.interactions) if len(self.interactions) > 0 else 0,
            "like_rate": float(total_likes) / len(self.interactions) if len(self.interactions) > 0 else 0,
            "share_rate": float(total_shares) / len(self.interactions) if len(self.interactions) > 0 else 0,
            "follow_rate": float(total_follows) / len(self.interactions) if len(self.interactions) > 0 else 0,
        }

    def split_train_val_test(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        use_cache: bool = True,
    ) -> Tuple[Any, Any, Any]: # Tuple[InteractionSlice, InteractionSlice, InteractionSlice]
        """
        按时间划分训练/验证/测试集。支持缓存划分结果。

        Returns:
            (train, val, test) 三个 InteractionSlice
        """
        # 尝试从缓存加载划分结果
        if use_cache and self.use_cache:
            split_cache_path = self._get_split_cache_path(seed)
            if split_cache_path.exists():
                try:
                    with open(split_cache_path, "rb") as f:
                        split_data = pickle.load(f)

                    if (split_data.get("total_users") == len(self.user_set) and
                        split_data.get("max_users") == self.max_users):

                        train_indices = split_data["train_indices"]
                        val_indices = split_data["val_indices"]
                        test_indices = split_data["test_indices"]

                        train = InteractionSlice(self.store, train_indices)
                        val = InteractionSlice(self.store, val_indices)
                        test = InteractionSlice(self.store, test_indices)

                        logger.info(f"从缓存加载数据划分: train={len(train)}, val={len(val)}, test={len(test)}")
                        return train, val, test
                except Exception as e:
                    logger.warning(f"划分缓存加载失败: {e}")

        # 按时间排序的索引
        # 如果 self.interactions 是全量 Slice (即 range(n))，直接用 store 的 sorted_indices
        if self.store and len(self.interactions) == len(self.store):
            sorted_indices = self.store.get_sorted_indices()
        else:
            # 如果是筛选后的子集 (max_users)，需要对子集进行排序
            # 这里 InteractionSlice.__iter__ 返回 InteractionView，可以用于排序
            # 但为了性能，我们要对 indices 进行排序。
            # 获取所有 View，按 timestamp 排序
            # 这是一个相对慢的操作 (~1M items)，但只做一次
            current_indices = self.interactions._indices
            timestamps = self.store.timestamps[current_indices]
            # argsort 获取排序后的相对索引
            relative_sorted = np.argsort(timestamps)
            # 映射回绝对索引
            sorted_indices = current_indices[relative_sorted]

        n = len(sorted_indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_indices = sorted_indices[:train_end]
        val_indices = sorted_indices[train_end:val_end]
        test_indices = sorted_indices[val_end:]

        train = InteractionSlice(self.store, train_indices)
        val = InteractionSlice(self.store, val_indices)
        test = InteractionSlice(self.store, test_indices)

        logger.info(f"数据划分: train={len(train)}, val={len(val)}, test={len(test)}")

        # 保存划分缓存
        if use_cache and self.use_cache:
            try:
                split_cache_path = self._get_split_cache_path(seed)
                os.makedirs(split_cache_path.parent, exist_ok=True)
                split_data = {
                    "version": self.CACHE_VERSION,
                    "total_users": len(self.user_set),
                    "max_users": self.max_users,
                    # 只保存 numpy 索引数组，非常紧凑
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                    "seed": seed,
                }
                with open(split_cache_path, "wb") as f:
                    pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"数据划分已缓存: {split_cache_path}")
            except Exception as e:
                logger.warning(f"划分缓存保存失败: {e}")

        return train, val, test

    def get_user_history(self, user_id: int) -> Optional[TenrecUserHistory]:
        """获取指定用户的历史记录。"""
        return self.user_histories.get(user_id)

    def get_all_items(self) -> List[int]:
        """获取所有 item ID。"""
        return list(self.item_set)

    def set_negative_pool(self, item_ids: set):
        """
        设置负采样候选池（用于防止未来数据泄漏）。

        在时间切分场景下，训练和验证阶段的负采样应仅使用训练集出现过的物品，
        避免将测试集的未来物品作为负样本导致验证指标虚高。

        Args:
            item_ids: 允许作为负样本的物品 ID 集合（通常为训练集物品）
        """
        self._negative_pool = item_ids
        # 缓存 numpy 数组用于快速采样（避免每次 list(set) 的开销）
        self._negative_pool_array = np.array(list(item_ids))
        logger.info(f"负采样候选池已设置: {len(item_ids)} items (全量: {len(self.item_set)} items)")

    def clear_negative_pool(self):
        """清除负采样候选池限制，恢复使用全量物品。"""
        self._negative_pool = None
        self._negative_pool_array = None
        logger.info("负采样候选池已清除，恢复使用全量物品集")

    def sample_negative_items(
        self,
        user_id: int,
        num_negatives: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """
        为用户采样负样本 items。

        Args:
            user_id: 用户 ID
            num_negatives: 负样本数量
            rng: 随机数生成器

        Returns:
            负样本 item ID 列表
        """
        if rng is None:
            rng = np.random.default_rng()

        positive_items = set()

        # 优先使用 Store 的预构建映射 (O(1) 查找)
        if self.store:
            user_pos_map = self.store.build_user_positive_items()
            if user_id in user_pos_map:
                positive_items = user_pos_map[user_id]

        # 回退到 user_histories (兼容旧逻辑 / 小数据集)
        elif user_id in self.user_histories:
            positive_items = {inter.item_id for inter in self.user_histories[user_id].interactions}

        # 决定采样源
        use_pool = self._negative_pool is not None
        pool_size = len(self._negative_pool) if use_pool else len(self.item_set)

        # 策略 1: 小候选集 -> 差集采样 (精确，无碰撞)
        if pool_size < 50000:
            pool = self._negative_pool if use_pool else self.item_set
            candidate_items = list(pool - positive_items)

            if len(candidate_items) < num_negatives:
                return list(rng.choice(candidate_items, size=num_negatives, replace=True)) if candidate_items else []
            return list(rng.choice(candidate_items, size=num_negatives, replace=False))

        # 策略 2: 大候选集 -> 向量化拒绝采样 (Vectorized Rejection Sampling)
        # 避免构建 list(pool - positive_items) 的巨大开销 (O(N))
        else:
            # 准备源数组
            if use_pool:
                source_array = self._negative_pool_array
            else:
                # 懒加载全量 item 数组
                if not hasattr(self, '_all_items_array') or self._all_items_array is None:
                    self._all_items_array = np.array(list(self.item_set))
                source_array = self._all_items_array

            # 向量化拒绝采样：一次性采样大量候选，用 np.isin 批量过滤
            positive_array = np.array(list(positive_items), dtype=np.int32) if positive_items else np.array([], dtype=np.int32)
            result = np.empty(0, dtype=np.int32)
            max_attempts = 5

            for _ in range(max_attempts):
                needed = num_negatives - len(result)
                if needed <= 0:
                    break

                # 过采样系数 3.0 以确保足够的有效候选
                candidates = rng.choice(source_array, size=needed * 3, replace=True)

                # 向量化过滤：排除正样本
                if len(positive_array) > 0:
                    mask = ~np.isin(candidates, positive_array)
                    candidates = candidates[mask]

                # 去重（同时排除已有结果）
                if len(result) > 0:
                    mask2 = ~np.isin(candidates, result)
                    candidates = candidates[mask2]
                _, unique_idx = np.unique(candidates, return_index=True)
                candidates = candidates[np.sort(unique_idx)]

                result = np.concatenate([result, candidates[:needed]])

            # 兜底：如果尝试多次仍不够（极罕见），回退到差集逻辑
            if len(result) < num_negatives:
                pool = self._negative_pool if use_pool else self.item_set
                candidate_items = list(pool - positive_items)
                remaining = num_negatives - len(result)
                if candidate_items:
                    fallback_add = rng.choice(np.array(candidate_items), size=remaining, replace=True)
                    result = np.concatenate([result, fallback_add])

            return result[:num_negatives].astype(int).tolist()

    def batch_sample_negative_items(
        self,
        user_ids: np.ndarray,
        num_negatives: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        批量为多个用户采样负样本（向量化实现）。

        对于大批量（如 eval cache 构建），比逐个调用 sample_negative_items 快 100x+。

        Args:
            user_ids: 用户 ID 数组 [N]
            num_negatives: 每个用户的负样本数量
            rng: 随机数生成器

        Returns:
            负样本 item ID 数组 [N, num_negatives]
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(user_ids)

        # 准备源数组
        use_pool = self._negative_pool is not None
        if use_pool:
            source_array = self._negative_pool_array
        else:
            if not hasattr(self, '_all_items_array') or self._all_items_array is None:
                self._all_items_array = np.array(list(self.item_set))
            source_array = self._all_items_array

        # 第一步：批量随机采样 (N, num_negatives) — 不考虑碰撞
        # 对于 210 万物品池 + 127 负样本，碰撞率 < 0.006%，极低
        neg_indices = rng.integers(0, len(source_array), size=(n, num_negatives))
        result = source_array[neg_indices]  # [N, num_negatives]

        # 第二步：修复碰撞样本（用户已交互的物品）
        # 只需处理极少数碰撞 case，避免全量逐用户循环
        user_pos_map = self.store.build_user_positive_items() if self.store else {}

        # 找出需要修复的 (i, j) 位置
        # 优化：按用户去重，相同用户只查一次 positive_items
        unique_uids, uid_inverse = np.unique(user_ids, return_inverse=True)

        for uid_idx, uid in enumerate(unique_uids):
            pos_items = user_pos_map.get(int(uid))
            if not pos_items:
                continue

            # 找出该用户的所有行
            row_mask = uid_inverse == uid_idx
            row_indices = np.where(row_mask)[0]

            # 检查这些行中是否有碰撞
            pos_array = np.array(list(pos_items), dtype=np.int32)
            sub_result = result[row_indices]  # [num_rows, num_negatives]
            collision_mask = np.isin(sub_result, pos_array)

            if not collision_mask.any():
                continue

            # 只修复碰撞位置
            collision_rows, collision_cols = np.where(collision_mask)
            for r, c in zip(collision_rows, collision_cols):
                # 重采样直到不碰撞
                for _ in range(100):
                    new_item = source_array[rng.integers(0, len(source_array))]
                    if int(new_item) not in pos_items:
                        result[row_indices[r], c] = new_item
                        break

        return result
