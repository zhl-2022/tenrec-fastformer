//! Thread-safe in-memory item store, inspired by Thunder's PostStore.
//!
//! Uses DashMap for lock-free concurrent read/write access.
//! Designed for sub-microsecond user→item history lookups in online serving.

use dashmap::DashMap;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Minimal item record stored in user timelines.
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ItemRecord {
    #[pyo3(get)]
    pub item_id: i64,
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub item_type: u8, // 0=original, 1=interaction, 2=video
}

#[pymethods]
impl ItemRecord {
    #[new]
    fn new(item_id: i64, timestamp: i64, item_type: u8) -> Self {
        ItemRecord {
            item_id,
            timestamp,
            item_type,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ItemRecord(item_id={}, ts={}, type={})",
            self.item_id, self.timestamp, self.item_type
        )
    }
}

/// Thread-safe in-memory store for user→item histories.
///
/// Inspired by x-algorithm/thunder PostStore:
///   - DashMap<user_id, VecDeque<ItemRecord>> for O(1) user lookups
///   - Automatic retention-based trimming
///   - Concurrent insert/query without GIL
///
/// Python usage:
///   store = PostStore(retention_seconds=86400*2)  # 2 days
///   store.insert(user_id=1, item_id=100, timestamp=now, item_type=0)
///   items = store.get_user_history(user_id=1, max_items=20)
///   store.trim_old()
#[pyclass]
pub struct PostStore {
    /// user_id → timeline of ItemRecord (newest last)
    items_by_user: Arc<DashMap<i64, VecDeque<ItemRecord>>>,
    /// Global item_id → bool for soft-deleted items
    deleted_items: Arc<DashMap<i64, bool>>,
    /// Retention period in seconds
    retention_seconds: u64,
    /// Max items per user
    max_per_user: usize,
}

#[pymethods]
impl PostStore {
    /// Create a new PostStore.
    ///
    /// Args:
    ///     retention_seconds: how long to keep items (default 172800 = 2 days)
    ///     max_per_user: max items per user timeline (default 500)
    #[new]
    #[pyo3(signature = (retention_seconds=172800, max_per_user=500))]
    fn new(retention_seconds: u64, max_per_user: usize) -> Self {
        PostStore {
            items_by_user: Arc::new(DashMap::new()),
            deleted_items: Arc::new(DashMap::new()),
            retention_seconds,
            max_per_user,
        }
    }

    /// Insert an item into a user's timeline.
    fn insert(&self, user_id: i64, item_id: i64, timestamp: i64, item_type: u8) {
        if self.deleted_items.contains_key(&item_id) {
            return;
        }

        let record = ItemRecord {
            item_id,
            timestamp,
            item_type,
        };

        let mut entry = self.items_by_user.entry(user_id).or_default();
        let deque = entry.value_mut();

        // Avoid duplicates
        if deque.iter().any(|r| r.item_id == item_id) {
            return;
        }

        deque.push_back(record);

        // Enforce max per user
        while deque.len() > self.max_per_user {
            deque.pop_front();
        }
    }

    /// Batch insert multiple items.
    ///
    /// Args:
    ///     user_ids: list of user IDs
    ///     item_ids: list of item IDs (same length)
    ///     timestamps: list of timestamps (same length)
    ///     item_types: list of item types (same length)
    fn insert_batch(
        &self,
        user_ids: Vec<i64>,
        item_ids: Vec<i64>,
        timestamps: Vec<i64>,
        item_types: Vec<u8>,
    ) {
        for i in 0..user_ids.len() {
            self.insert(user_ids[i], item_ids[i], timestamps[i], item_types[i]);
        }
    }

    /// Retrieve a user's history (most recent first).
    ///
    /// Args:
    ///     user_id: the user to look up
    ///     max_items: maximum items to return (default 20)
    ///
    /// Returns:
    ///     List of ItemRecord, newest first.
    #[pyo3(signature = (user_id, max_items=20))]
    fn get_user_history(&self, user_id: i64, max_items: usize) -> Vec<ItemRecord> {
        match self.items_by_user.get(&user_id) {
            Some(entry) => {
                let deque = entry.value();
                deque
                    .iter()
                    .rev()
                    .filter(|r| !self.deleted_items.contains_key(&r.item_id))
                    .take(max_items)
                    .cloned()
                    .collect()
            }
            None => Vec::new(),
        }
    }

    /// Get histories for multiple users at once.
    ///
    /// Returns a list of lists (one per user_id, in order).
    #[pyo3(signature = (user_ids, max_per_user=20))]
    fn get_batch_histories(
        &self,
        user_ids: Vec<i64>,
        max_per_user: usize,
    ) -> Vec<Vec<ItemRecord>> {
        user_ids
            .iter()
            .map(|uid| self.get_user_history(*uid, max_per_user))
            .collect()
    }

    /// Get only item_ids for a user (convenience method).
    #[pyo3(signature = (user_id, max_items=20))]
    fn get_user_item_ids(&self, user_id: i64, max_items: usize) -> Vec<i64> {
        self.get_user_history(user_id, max_items)
            .into_iter()
            .map(|r| r.item_id)
            .collect()
    }

    /// Mark items as deleted (soft-delete).
    fn mark_deleted(&self, item_ids: Vec<i64>) {
        for id in item_ids {
            self.deleted_items.insert(id, true);
        }
    }

    /// Trim items older than retention_seconds from ALL users.
    ///
    /// Returns the total number of items trimmed.
    fn trim_old(&self) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let cutoff = now - self.retention_seconds as i64;
        let mut total_trimmed = 0;
        let mut empty_users = Vec::new();

        for mut entry in self.items_by_user.iter_mut() {
            let user_id = *entry.key();
            let deque = entry.value_mut();

            while let Some(front) = deque.front() {
                if front.timestamp < cutoff {
                    deque.pop_front();
                    total_trimmed += 1;
                } else {
                    break;
                }
            }

            if deque.is_empty() {
                empty_users.push(user_id);
            }
        }

        // Clean up empty entries
        for uid in empty_users {
            self.items_by_user.remove_if(&uid, |_, v| v.is_empty());
        }

        total_trimmed
    }

    /// Sort all user timelines by timestamp (call after bulk loading).
    fn sort_all(&self) {
        for mut entry in self.items_by_user.iter_mut() {
            let deque = entry.value_mut();
            let mut vec: Vec<_> = deque.drain(..).collect();
            vec.sort_unstable_by_key(|r| r.timestamp);
            for item in vec {
                deque.push_back(item);
            }
        }
    }

    /// Get statistics about the store.
    fn stats(&self) -> (usize, usize, usize) {
        let user_count = self.items_by_user.len();
        let total_items: usize = self
            .items_by_user
            .iter()
            .map(|e| e.value().len())
            .sum();
        let deleted = self.deleted_items.len();
        (user_count, total_items, deleted)
    }

    /// Clear all data.
    fn clear(&self) {
        self.items_by_user.clear();
        self.deleted_items.clear();
    }

    fn __repr__(&self) -> String {
        let (users, items, deleted) = self.stats();
        format!(
            "PostStore(users={}, items={}, deleted={}, retention={}s)",
            users, items, deleted, self.retention_seconds
        )
    }
}
