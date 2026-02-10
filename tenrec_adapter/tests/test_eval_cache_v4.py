
import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from tenrec_adapter.eval_cache import CachedEvalDataset
from tenrec_adapter.data_loader import InteractionView

# Mock classes to simulate the environment
class MockStore:
    def __init__(self, n=100):
        self.user_ids = np.arange(n)
        self.item_ids = np.arange(n) * 2
        self.clicks = np.ones(n)
        self.likes = np.zeros(n)
        self.shares = np.zeros(n)
        self.follows = np.zeros(n)
        self.has_hist = True
        self.hist_matrix = np.ones((n, 10), dtype=np.int32)

class MockDataLoader:
    def __init__(self, store):
        self.store = store
    
    def batch_sample_negative_items(self, user_ids, num_neg, rng):
        # Return mock negatives: [N, num_neg]
        return np.zeros((len(user_ids), num_neg), dtype=np.int32)

class MockAdapter:
    def __init__(self, store):
        self.data_loader = MockDataLoader(store)
        self.rng = np.random.default_rng(42)
        self.hash_table_size = 1000
        self.num_item_hashes = 2
        self.num_user_hashes = 2
        self.num_author_hashes = 2
        self.history_seq_len = 5
        self.num_actions = 4
        self.product_surface_vocab_size = 10

    def _batch_generate_multi_hash(self, ids, num_hashes, table_size):
        # Simple mock hash: id + offset
        # Returns [shape] + [num_hashes]
        return np.stack([ids] * num_hashes, axis=-1) % table_size

    def create_training_batch(self, interactions, num_negatives):
        # Not used in vectorized path, but needed for interface
        pass

class MockInteractionSlice:
    def __init__(self, indices):
        self._indices = indices
    def __len__(self):
        return len(self._indices)

def collate_fn(batch):
    return {
        "user_ids": torch.stack([b["user_ids"] for b in batch]),
        "history_item_ids": torch.stack([b["history_item_ids"] for b in batch]),
        "history_actions": torch.stack([b["history_actions"] for b in batch]),
        "candidate_item_ids": torch.stack([b["candidate_item_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

class TestEvalCacheV4:
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    def test_vectorized_build_and_load(self, temp_dir):
        # Setup
        n = 100
        num_neg = 4
        store = MockStore(n)
        adapter = MockAdapter(store)
        interactions = MockInteractionSlice(np.arange(n))
        cache_path = Path(temp_dir) / "test_cache.pt"

        # 1. Build Cache
        dataset = CachedEvalDataset(
            cache_path=str(cache_path),
            interactions=interactions,
            adapter=adapter,
            num_negatives=num_neg,
            seed=42,
            force_rebuild=True
        )

        # Verify length
        assert len(dataset) == n
        assert dataset.length == n
        
        # Verify data structure (Dict[str, Tensor])
        assert isinstance(dataset.data, dict)
        assert "user_ids" in dataset.data
        assert isinstance(dataset.data["user_ids"], torch.Tensor)
        assert dataset.data["user_ids"].shape == (n,)
        
        # Verify candidate shape [N, 1+num_neg]
        assert dataset.data["candidate_item_ids"].shape == (n, 1 + num_neg)

        # 2. Check __getitem__
        item0 = dataset[0]
        assert isinstance(item0, dict)
        assert "user_ids" in item0
        assert item0["user_ids"].dim() == 0  # scalar
        assert item0["candidate_item_ids"].shape == (1 + num_neg,)

        # 3. Save is automatic in __init__, check file exists
        assert cache_path.exists()

        # 4. Load Cache
        dataset_loaded = CachedEvalDataset(
            cache_path=str(cache_path),
            interactions=interactions, # Indices don't matter for load
            adapter=adapter,
            num_negatives=num_neg,
            seed=42,
            force_rebuild=False
        )
        
        assert len(dataset_loaded) == n
        # Check if data matches
        assert torch.equal(dataset.data["user_ids"], dataset_loaded.data["user_ids"])

    def test_dataloader_integration(self, temp_dir):
        # Verify compatibility with DataLoader and collate_fn
        n = 100
        store = MockStore(n)
        adapter = MockAdapter(store)
        interactions = MockInteractionSlice(np.arange(n))
        cache_path = Path(temp_dir) / "test_dl.pt"

        dataset = CachedEvalDataset(
            cache_path=str(cache_path),
            interactions=interactions,
            adapter=adapter,
            force_rebuild=True
        )

        dl = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
        
        batch = next(iter(dl))
        assert "user_ids" in batch
        assert batch["user_ids"].shape == (10,)
        assert batch["candidate_item_ids"].shape == (10, 5) # 1+4 negs
