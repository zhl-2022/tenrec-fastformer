#!/usr/bin/env python
# MIT License - see LICENSE for details

"""
推理演示脚本。

在本地 CPU 上运行推荐系统的完整推理流程，用于验证模型架构和数据管道。
不需要 GPU 或训练好的权重 — 使用随机初始化权重演示数据流。

Usage:
    python tenrec_adapter/inference_demo.py --data_dir data/tenrec/Tenrec --scenario QB-video
    python tenrec_adapter/inference_demo.py --scenario QB-video --max_users 100 --encoder_type transformer
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tenrec_adapter.data_loader import TenrecDataLoader
from tenrec_adapter.models import TwoTowerModel
from tenrec_adapter.ranking_model import RankingModel
from tenrec_adapter.metrics import compute_auc, compute_ndcg_at_k, compute_hit_rate_at_k, compute_mrr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_batch(
    data_loader: TenrecDataLoader,
    interactions,
    batch_size: int = 32,
    num_negatives: int = 15,
    history_seq_len: int = 10,
):
    """从交互数据构建推理 batch。"""
    # 取前 batch_size 条正样本
    user_ids = []
    history_item_ids = []
    candidate_item_ids = []
    labels = []

    all_items = list(data_loader.item_set)
    user_positive = data_loader.store.build_user_positive_items()

    count = 0
    seen_users = set()

    for inter in interactions:
        uid = inter.user_id
        if uid in seen_users:
            continue
        seen_users.add(uid)

        # 用户历史
        if data_loader.store.has_hist:
            hist = inter.hist_items[:history_seq_len]
        elif uid in data_loader.user_histories:
            hist = data_loader.user_histories[uid].get_recent_items(history_seq_len)
        else:
            continue

        if len(hist) == 0:
            continue

        # Padding history
        hist_padded = hist + [0] * (history_seq_len - len(hist))

        # 候选: 1 正样本 + num_negatives 负样本
        pos_item = inter.item_id
        pos_items = user_positive.get(uid, set())

        neg_items = []
        attempts = 0
        while len(neg_items) < num_negatives and attempts < num_negatives * 5:
            neg = all_items[np.random.randint(0, len(all_items))]
            if neg not in pos_items:
                neg_items.append(neg)
            attempts += 1

        # Padding negatives if not enough
        while len(neg_items) < num_negatives:
            neg_items.append(all_items[np.random.randint(0, len(all_items))])

        candidates = [pos_item] + neg_items
        label = [1] + [0] * num_negatives

        user_ids.append(uid)
        history_item_ids.append(hist_padded)
        candidate_item_ids.append(candidates)
        labels.append(label)

        count += 1
        if count >= batch_size:
            break

    if count == 0:
        return None

    num_actions = 4  # click, like, share, follow
    C = num_negatives + 1

    batch = {
        "user_ids": torch.tensor(user_ids, dtype=torch.long),
        "history_item_ids": torch.tensor(history_item_ids, dtype=torch.long),
        "history_actions": torch.zeros(count, history_seq_len, num_actions),
        "candidate_item_ids": torch.tensor(candidate_item_ids, dtype=torch.long),
        "labels": torch.zeros(count, C, num_actions),
    }

    # 设置正样本 label (click=1)
    for i in range(count):
        batch["labels"][i, 0, 0] = 1.0  # 第一个候选是正样本，click=1

    return batch, labels


def run_retrieval_demo(model, batch, labels):
    """运行召回模型推理。"""
    logger.info("=" * 50)
    logger.info("Stage 1: Retrieval (Two-Tower)")
    logger.info("=" * 50)

    with torch.no_grad():
        start = time.time()
        outputs = model(batch)
        elapsed = (time.time() - start) * 1000

    logits = outputs["logits"]  # [B, C, num_actions]
    scores = logits[:, :, 0]  # click scores

    B, C = scores.shape
    logger.info(f"  Input:  {B} users × {C} candidates")
    logger.info(f"  Output: logits {list(logits.shape)}")
    logger.info(f"  Time:   {elapsed:.1f}ms")

    # labels from build_batch is List[List[int]] — already binary (click only)
    all_labels = np.array(labels)  # [B, C]
    all_scores = scores.numpy()

    auc = compute_auc(all_labels.flatten(), all_scores.flatten())
    ndcg = compute_ndcg_at_k(all_labels, all_scores, k=5)
    hit = compute_hit_rate_at_k(all_labels, all_scores, k=1)
    mrr = compute_mrr(all_labels, all_scores)

    logger.info(f"  ---")
    logger.info(f"  AUC:      {auc:.4f}  (random baseline: 0.5000)")
    logger.info(f"  NDCG@5:   {ndcg:.4f}")
    logger.info(f"  Hit@1:    {hit:.4f}")
    logger.info(f"  MRR:      {mrr:.4f}")

    # Top-3 user 推荐结果
    logger.info(f"  ---")
    logger.info(f"  Sample Recommendations (Top-3 per user):")
    for i in range(min(3, B)):
        sorted_idx = torch.argsort(scores[i], descending=True)[:3]
        items = batch["candidate_item_ids"][i][sorted_idx].tolist()
        s = scores[i][sorted_idx].tolist()
        logger.info(f"    User {batch['user_ids'][i].item()}: "
                     f"items={items}, scores=[{', '.join(f'{x:.3f}' for x in s)}]")

    return auc


def run_ranking_demo(model, batch, labels):
    """运行精排模型推理。"""
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Stage 2: Ranking ({model.encoder_type})")
    logger.info("=" * 50)

    with torch.no_grad():
        start = time.time()
        outputs = model(batch)
        elapsed = (time.time() - start) * 1000

    logits = outputs["logits"]
    scores_tensor = outputs["scores"]

    B, C = scores_tensor.shape
    logger.info(f"  Input:  {B} users × {C} candidates")
    logger.info(f"  Output: logits {list(logits.shape)}, scores {list(scores_tensor.shape)}")
    logger.info(f"  Time:   {elapsed:.1f}ms")
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # 计算指标
    all_labels = np.array(labels)
    all_scores = scores_tensor.numpy()

    auc = compute_auc(all_labels.flatten(), all_scores.flatten())
    ndcg = compute_ndcg_at_k(all_labels, all_scores, k=5)
    hit = compute_hit_rate_at_k(all_labels, all_scores, k=1)
    mrr = compute_mrr(all_labels, all_scores)

    logger.info(f"  ---")
    logger.info(f"  AUC:      {auc:.4f}  (random baseline: 0.5000)")
    logger.info(f"  NDCG@5:   {ndcg:.4f}")
    logger.info(f"  Hit@1:    {hit:.4f}")
    logger.info(f"  MRR:      {mrr:.4f}")

    # Top-3 user 推荐结果
    logger.info(f"  ---")
    logger.info(f"  Sample Recommendations (Top-3 per user):")
    for i in range(min(3, B)):
        sorted_idx = torch.argsort(scores_tensor[i], descending=True)[:3]
        items = batch["candidate_item_ids"][i][sorted_idx].tolist()
        s = scores_tensor[i][sorted_idx].tolist()
        logger.info(f"    User {batch['user_ids'][i].item()}: "
                     f"items={items}, scores=[{', '.join(f'{x:.3f}' for x in s)}]")

    return auc


def main():
    parser = argparse.ArgumentParser(description="Tenrec-Fastformer Inference Demo")
    parser.add_argument("--data_dir", type=str, default="data/tenrec/Tenrec")
    parser.add_argument("--scenario", type=str, default="QB-video",
                        choices=["QB-video", "QK-video", "ctr_data_1M"])
    parser.add_argument("--max_users", type=int, default=500,
                        help="Max users to load (for fast demo)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_negatives", type=int, default=15)
    parser.add_argument("--history_seq_len", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding dim (smaller for CPU demo)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Encoder layers")
    parser.add_argument("--encoder_type", type=str, default="fastformer",
                        choices=["fastformer", "transformer"],
                        help="Ranking encoder type")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (optional)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("  Tenrec-Fastformer Inference Demo")
    logger.info("=" * 60)
    logger.info(f"  Scenario:     {args.scenario}")
    logger.info(f"  Encoder:      {args.encoder_type}")
    logger.info(f"  Embed Dim:    {args.embed_dim}")
    logger.info(f"  Num Layers:   {args.num_layers}")
    logger.info(f"  Device:       CPU")
    if args.checkpoint:
        logger.info(f"  Checkpoint:   {args.checkpoint}")
    else:
        logger.info(f"  Checkpoint:   None (random weights)")
    logger.info("")

    # 1. 加载数据
    logger.info("[1/4] Loading data...")
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # 尝试项目根目录下
        data_dir = PROJECT_ROOT / args.data_dir
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error(f"Please download Tenrec dataset and place CSV files in: {args.data_dir}")
        sys.exit(1)

    loader = TenrecDataLoader(
        data_dir=str(data_dir),
        scenario=args.scenario,
        max_users=args.max_users,
        use_cache=True,
    ).load()

    stats = loader.get_statistics()
    logger.info(f"  Loaded: {stats['total_interactions']:,} interactions, "
                f"{stats['total_users']:,} users, {stats['total_items']:,} items")
    logger.info(f"  Click rate: {stats['click_rate']:.2%}")

    # 2. 划分数据
    logger.info("\n[2/4] Splitting data...")
    train, val, test = loader.split_train_val_test(seed=args.seed)

    # 3. 构建 batch
    logger.info(f"\n[3/4] Building inference batch (batch_size={args.batch_size})...")
    result = build_batch(
        loader, test,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        history_seq_len=args.history_seq_len,
    )
    if result is None:
        logger.error("Failed to build batch — not enough data")
        sys.exit(1)

    batch, binary_labels = result

    # 4. 推理
    logger.info(f"\n[4/4] Running inference...")
    num_users = max(loader.user_set) + 2  # +2 for padding idx 0
    num_items = max(loader.item_set) + 2

    # --- Retrieval ---
    retrieval_model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        hidden_dim=args.embed_dim * 2,
        num_heads=4,
    )
    retrieval_model.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        retrieval_model.load_state_dict(ckpt.get("retrieval_model", ckpt.get("model_state_dict", {})), strict=False)
        logger.info("  Loaded retrieval checkpoint")

    run_retrieval_demo(retrieval_model, batch, binary_labels)

    # --- Ranking ---
    ranking_model = RankingModel(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        hidden_dim=args.embed_dim * 2,
        num_heads=4,
        num_layers=args.num_layers,
        encoder_type=args.encoder_type,
    )
    ranking_model.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ranking_model.load_state_dict(ckpt.get("ranking_model", ckpt.get("model_state_dict", {})), strict=False)
        logger.info("  Loaded ranking checkpoint")

    run_ranking_demo(ranking_model, batch, binary_labels)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Demo Complete!")
    logger.info("=" * 60)
    logger.info(f"  Note: Results use {'trained' if args.checkpoint else 'random'} weights.")
    if not args.checkpoint:
        logger.info(f"  AUC ~0.5 is expected with random weights.")
        logger.info(f"  Train the model to get meaningful predictions:")
        logger.info(f"    bash tenrec_adapter/scripts/run_ctr1m_ranking.sh")


if __name__ == "__main__":
    main()
