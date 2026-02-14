import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Lazy import torch inside functions to avoid Streamlit file watcher issues
# from tenrec_adapter.models import TwoTowerModel
# from tenrec_adapter.ranking_model import RankingModel

st.set_page_config(
    page_title="Tenrec Data & Inference",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Constants =====
# 与训练配置保持一致
HASH_TABLE_SIZE = 100001
HISTORY_SEQ_LEN = 10  # 训练时使用的长度
NUM_NEGATIVES = 127   # 训练时使用的负样本数

# ===== Sidebar Config =====
st.sidebar.title("Tenrec Dashboard")

# Dataset Selection
DATA_DIR = st.sidebar.text_input("Data Directory", value="data/tenrec/Tenrec")
SCENARIO = st.sidebar.selectbox(
    "Dataset Scenario",
    ["QB-video", "QK-video", "ctr_data_1M", "QK-article", "QB-article"],
    index=2  # Default to ctr_data_1M
)

@st.cache_resource
def load_data(data_dir, scenario, max_rows=100000, use_parquet=True):
    """Load dataset for visualization (supports Parquet > CSV)."""
    # 1. Try Parquet (Fastest)
    parquet_path = Path(data_dir) / f"{scenario}.parquet"
    if use_parquet and parquet_path.exists():
        try:
            if max_rows is None:
                return pd.read_parquet(parquet_path)
            else:
                return pd.read_parquet(parquet_path).head(max_rows)
        except Exception as e:
            st.warning(f"Failed to read parquet: {e}")

    # 2. Try CSV (Slower)
    csv_path = Path(data_dir) / f"{scenario}.csv"
    if not csv_path.exists():
        return None

    # Try polars for speed
    try:
        import polars as pl
        if max_rows:
            df = pl.scan_csv(str(csv_path), null_values=["\\N", ""]).head(max_rows).collect().to_pandas()
        else:
            df = pl.read_csv(str(csv_path), null_values=["\\N", ""]).to_pandas()
    except ImportError:
        df = pd.read_csv(csv_path, nrows=max_rows, na_values=["\\N", ""])

    # Fill NA
    for col in df.columns:
        if df[col].dtype in (float, 'float64'):
            df[col] = df[col].fillna(0).astype(int)  # ID类通常是int

    return df

@st.cache_resource
def load_model(ckpt_path):
    """Load Ranking Model from checkpoint."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, "File not found"

    try:
        import torch
        from tenrec_adapter.ranking_model import RankingModel

        # Load Checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Get Config
        config = ckpt.get("config", {})
        model_sd = ckpt.get("model_state_dict", {})

        # 如果是 DDP 训练的模型，key 会有 "module." 前缀
        if any(k.startswith("module.") for k in model_sd.keys()):
            model_sd = {k.replace("module.", ""): v for k, v in model_sd.items()}

        # Params
        embed_dim = config.get("embed_dim", 512)
        hidden_dim = config.get("hidden_dim", 1024)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("ranking_num_layers", 12)  # Ranking key
        encoder_type = config.get("encoder_type", "fastformer")

        # history_len 必须与训练一致
        history_len = config.get("history_seq_len", HISTORY_SEQ_LEN)

        # num_candidates: 训练时通常是 (num_negatives + 1)
        # 但模型初始化时不仅取决于训练时的负样本数，还取决于 Position Embedding 的大小
        # 我们从 checkpoint 的 pos_encoding 权重推断 max_len
        if "pos_encoding.pe" in model_sd:
            pe_len = model_sd["pos_encoding.pe"].shape[1]
            # max_len = max(history_len, num_candidates) + 2
            # 这里的 num_candidates 应该是模型能处理的最大值
            max_candidates = pe_len - 2 - history_len # Approximate logic or just use a safe large number
            if max_candidates < 1: max_candidates = 128
        else:
            max_candidates = 128  # Default safe value

        # Init Model
        model = RankingModel(
            num_users=HASH_TABLE_SIZE + 1,
            num_items=HASH_TABLE_SIZE + 1,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            history_len=history_len,
            num_candidates=max_candidates, # 只要 <= 训练时的配置即可
            encoder_type=encoder_type
        )

        # Load Weights
        # strict=False 因为可能有一些辅助参数不匹配
        keys = model.load_state_dict(model_sd, strict=False)
        if keys.missing_keys:
             # 过滤掉 unnecessary keys 引起的警告
             missing = [k for k in keys.missing_keys if "pos_encoding" not in k] # pos_encoding buffer
             if missing:
                 print(f"Warning: Missing keys: {missing}")

        model.eval()
        return model, config

    except Exception as e:
        return None, str(e)


def convert_csv_to_parquet(data_dir, scenario):
    """Convert CSV to Parquet for faster loading."""
    csv_path = Path(data_dir) / f"{scenario}.csv"
    parquet_path = Path(data_dir) / f"{scenario}.parquet"

    status = st.empty()
    status.info("Converting... This may take a moment.")

    try:
        # Use Polars for super fast conversion if available
        try:
            import polars as pl
            pl.scan_csv(str(csv_path), null_values=["\\N", ""]).collect().write_parquet(str(parquet_path))
        except ImportError:
            df = pd.read_csv(csv_path, na_values=["\\N", ""])
            df.to_parquet(parquet_path)
        status.success(f"Converted {scenario}.parquet! Refreshing...")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        status.error(f"Conversion failed: {e}")

# ===== UI Logic =====

# Check Parquet availability
parquet_exists = (Path(DATA_DIR) / f"{SCENARIO}.parquet").exists()

st.sidebar.markdown("---")
st.sidebar.subheader("Data Optimization")

use_full_data = st.sidebar.checkbox("Load Full Dataset", value=False, disabled=not parquet_exists)
max_rows = None if use_full_data else 100000

if parquet_exists:
    st.sidebar.success("⚡ Parquet available")
    if st.sidebar.button("Re-convert Parquet"):
         convert_csv_to_parquet(DATA_DIR, SCENARIO)
else:
    st.sidebar.warning("🐢 Using slow CSV")
    if st.sidebar.button("Convert to Parquet (Recommended)"):
        convert_csv_to_parquet(DATA_DIR, SCENARIO)

# Load Data
df = load_data(DATA_DIR, SCENARIO, max_rows=max_rows, use_parquet=True)


if df is None:
    st.error(f"Dataset file not found: {Path(DATA_DIR) / f'{SCENARIO}.csv'}")
    st.stop()

# Basic Stats
# num_users = df['user_id'].max() + 2  # Incorrect for sampled data
# num_items = df['item_id'].max() + 2

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🤖 Model Inference", "📈 Batch Eval", "🔬 Checkpoint Compare"])

# ===== TAB 1: Data Explorer =====
with tab1:
    st.header(f"Exploratory Data Analysis: {SCENARIO}")

    # Top Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (Sampled)", f"{len(df):,}")
    c2.metric("Unique Users (In Sample)", f"{df['user_id'].nunique():,}")
    c3.metric("Unique Items (In Sample)", f"{df['item_id'].nunique():,}")

    if "click" in df.columns:
        ctr = df["click"].mean()
        c4.metric("CTR (In Sample)", f"{ctr:.2%}")

    # Plots System
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Interaction Distribution")
        actions = [c for c in ["click", "like", "share", "follow"] if c in df.columns]
        if actions:
            counts = df[actions].sum().reset_index()
            counts.columns = ["Action", "Count"]
            fig = px.bar(counts, x="Action", y="Count", color="Action", title="Total Interactions by Type")
            st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("Item Popularity (Top 20)")
        pop = df['item_id'].value_counts().reset_index().head(20)
        pop.columns = ["Item ID", "Count"]
        # Convert Item ID to string for categorical axis
        pop["Item ID"] = pop["Item ID"].astype(str)
        fig = px.bar(pop, x="Item ID", y="Count", title="Top 20 Popular Items")
        st.plotly_chart(fig, use_container_width=True)

    # Raw Data Table
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(100), use_container_width=True)

# ===== TAB 2: Inference =====
with tab2:
    st.header("Single User Inference")

    c_side, c_main = st.columns([1, 3])

    with c_side:
        st.subheader("Configuration")

        # --- Model Source Selection ---
        ckpt_dir = Path("checkpoints/ranking")
        available_ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        available_ckpts = sorted([str(p) for p in available_ckpts], key=os.path.getmtime, reverse=True)

        ckpt_path = st.selectbox(
            "Select Checkpoint",
            available_ckpts + ["Custom Path"],
            index=0 if available_ckpts else 0
        )

        if ckpt_path == "Custom Path":
            ckpt_path = st.text_input("Checkpoint Path (.pt)", "checkpoints/ranking/best_model.pt")

        # User Selection
        # Filter for users with history (hist_len > 0 or in dataframe)
        sample_users = df['user_id'].unique()[:100]
        user_input = st.selectbox("Select User ID", sample_users)

        top_k = st.slider("Top K Recommendations", 5, 50, 10)

        btn_run = st.button("Run Inference", type="primary")

    with c_main:
        if btn_run:
            import torch

            # Load Model
            with st.spinner(f"Loading model from `{ckpt_path}`..."):
                model, config = load_model(ckpt_path)

            if model is None:
                st.error(f"Failed to load model: {config}") # config holds error msg here
            else:
                st.success(f"✅ Model loaded! Encoder: {model.encoder_type}, Layers: {model.num_layers}")

                # 1. Prepare Features
                # Find user history in dataframe
                user_rows = df[df['user_id'] == user_input].sort_values("timestamp", ascending=False)

                if len(user_rows) == 0:
                    st.warning("User not found in dataset sample.")
                else:
                    # Construct History
                    # Try to find hist_item_id columns first
                    hist_cols = [c for c in df.columns if c.startswith("hist_item_")]
                    if hist_cols:
                        # Use pre-built history from first row
                        row = user_rows.iloc[0]
                        hist_items = [row[c] for c in hist_cols if row[c] > 0]
                    else:
                        # Construct from interaction history
                        hist_items = user_rows['item_id'].tolist()

                    # Truncate/Pad
                    hist_items = hist_items[:HISTORY_SEQ_LEN]
                    hist_len = len(hist_items)
                    hist_padded = (hist_items + [0] * HISTORY_SEQ_LEN)[:HISTORY_SEQ_LEN]

                    st.write("### 1. User History")
                    st.write(f"History Length: {hist_len}")
                    st.write(hist_items)

                    # 2. Candidate Generation
                    # In a real system, this comes from Retrieval (TwoTower).
                    # Here we mock it by picking some popular items + random items
                    popular_items = df['item_id'].value_counts().head(50).index.tolist()
                    random_items = np.random.randint(1, HASH_TABLE_SIZE, 50).tolist()
                    candidates = list(set(popular_items + random_items))[:100]

                    # Ensure candidates > top_k
                    while len(candidates) < top_k:
                        candidates.append(np.random.randint(1, HASH_TABLE_SIZE))

                    # 3. Build Batch
                    # RankingModel expects:
                    # user_id: [B], history_item_ids: [B, S], candidate_item_ids: [B, C]
                    batch = {
                        "user_id": torch.tensor([user_input]),
                        "history_item_ids": torch.tensor([hist_padded]),
                        "candidate_item_ids": torch.tensor([candidates]),
                    }

                    # 4. Inference
                    with torch.no_grad():
                        outputs = model(batch)
                        scores = outputs["scores"][0].cpu().numpy() # [C]

                    # 5. Show Results
                    st.write("### 2. Recommended Items")

                    # Sort scores
                    sorted_idx = np.argsort(scores)[::-1][:top_k]
                    top_items = [candidates[i] for i in sorted_idx]
                    top_scores = [float(scores[i]) for i in sorted_idx]

                    rec_df = pd.DataFrame({
                        "Rank": range(1, top_k + 1),
                        "Item ID": top_items,
                        "Score": top_scores,
                    })

                    st.dataframe(rec_df.style.background_gradient(subset=["Score"], cmap="Viridis"))

                    # Histogram
                    fig = px.bar(rec_df, x="Item ID", y="Score", color="Score", title="Recommendation Scores")
                    # Force categorical x-axis
                    fig.update_xaxes(type='category')
                    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: Batch Eval =====
with tab3:
    st.header("Batch Evaluation (On Test Set)")

    st.info("Runs evaluation on a subset of the loaded data to estimate AUC/NDCG.")

    col_e1, col_e2 = st.columns([1, 3])
    with col_e1:
        eval_ckpt = st.selectbox("Checkpoint", available_ckpts, key="eval_ckpt")
        sample_size = st.slider("Sample Size", 100, 5000, 1000)
        btn_eval = st.button("Run Evaluation", type="primary")

    with col_e2:
        if btn_eval:
            import torch
            from tenrec_adapter.metrics import compute_auc, compute_ndcg_at_k

            with st.spinner("Loading model..."):
                model, _ = load_model(eval_ckpt)

            if model:
                # Prepare Eval Batch
                # Group by User to form (User, pos_item, neg_items) structure is complex in pure Pandas
                # Simplified approach: Pointwise eval (AUC only)
                # For Listwise (NDCG), we need (User, List[candidates], List[labels])

                with st.spinner(f"Evaluating on {sample_size} samples..."):
                    # 1. Sample data
                    eval_df = df.sample(n=sample_size, random_state=42)

                    # 2. Construct Batches (Batch size = 128)
                    bs = 128
                    all_scores = []
                    all_labels = []

                    # Mock negatives for pointwise AUC:
                    # For each positive (User, Item, Click=1), we need a negative (User, NegItem, Click=0)
                    # Simplified: Just predict on the row. If click=1, label=1.
                    # But ranking model is Listwise. It expects [B, C].
                    # Let's construct [B, 2]: [PosItem, NegItem]

                    user_ids_list = eval_df['user_id'].tolist()
                    item_ids_list = eval_df['item_id'].tolist()
                    clicks_list = eval_df['click'].tolist() if 'click' in eval_df else [1]*len(eval_df)

                    # Get History (Slow step, optimize later)
                    # Hack: use constant history for speed in this demo
                    dummy_hist = [0] * HISTORY_SEQ_LEN

                    # Random Negatives
                    all_neg_items = np.random.randint(1, HASH_TABLE_SIZE, size=len(eval_df)).tolist()

                    prog_bar = st.progress(0)

                    for i in range(0, len(eval_df), bs):
                        end = min(i + bs, len(eval_df))
                        curr_bs = end - i

                        u_ids = user_ids_list[i:end]
                        pos_items = item_ids_list[i:end]
                        neg_items = all_neg_items[i:end]

                        # Candidates: [Pos, Neg]
                        # Shape: [B, 2]
                        batch_cands = []
                        for p, n in zip(pos_items, neg_items):
                            batch_cands.append([p, n])

                        batch_tensor = {
                            "user_id": torch.tensor(u_ids),
                            "history_item_ids": torch.tensor([dummy_hist] * curr_bs), # Mock history
                            "candidate_item_ids": torch.tensor(batch_cands),
                        }

                        with torch.no_grad():
                            out = model(batch_tensor)
                            s = out["scores"].cpu().numpy() # [B, 2]

                        # Labels: Pos=1, Neg=0
                        # But wait, if original row had click=0, then Pos is actually Neg.
                        # True Label for PosItem = row['click']. True Label for NegItem = 0.
                        for k in range(curr_bs):
                            clk = clicks_list[i+k]
                            all_labels.append(1 if clk > 0 else 0)     # Label for PosItem
                            all_scores.append(s[k, 0])                 # Score for PosItem

                            all_labels.append(0)                       # Label for NegItem
                            all_scores.append(s[k, 1])                 # Score for NegItem

                        prog_bar.progress((i + bs) / len(eval_df))

                    # Metrics
                    auc = compute_auc(np.array(all_labels), np.array(all_scores))

                    st.metric("AUC (Estimated)", f"{auc:.4f}")
                    st.success("Evaluation Complete!")
                    st.caption("Note: This uses simplified Pointwise construction (1 Positive vs 1 Random Negative) and Mock History.")


# ===== TAB 4: Compare =====
with tab4:
    st.header("Epoch Comparison")

    if len(available_ckpts) < 1:
        st.warning("Need at least 1 checkpoint to compare.")
    else:
        # Load metrics from filename or log (Mocking for now)
        # Assuming filename format: checkpoint_epochX_stepY.pt
        data = []
        for ckpt in available_ckpts:
            name = os.path.basename(ckpt)
            # Try to extract epoch
            epoch = 0
            if "epoch" in name:
                import re
                m = re.search(r"epoch(\d+)", name)
                if m: epoch = int(m.group(1))

            # TODO: Load real metrics from checkpoint dict if possible
            # metrics = torch.load(ckpt, map_location="cpu")["metrics"]

            data.append({"Checkpoint": name, "Epoch": epoch})

        cmp_df = pd.DataFrame(data).sort_values("Epoch")

        st.dataframe(cmp_df)
        st.line_chart(cmp_df, x="Epoch", y="Epoch") # Placeholder chart
