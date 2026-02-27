/**
 * PTO Runtime2 - TensorMap Interface
 *
 * TensorMap provides producer lookup for dependency discovery:
 * - Maps Tensor -> producer task ID
 * - Used by pto_submit_task() to find dependencies
 *
 * Key design features:
 * 1. Ring buffer pool for entries (no malloc/free)
 * 2. Lazy invalidation (entries become stale when producer retires)
 * 3. Chain truncation optimization (truncate on first stale entry)
 * 4. Per-task entry tracking for efficient cleanup
 * 5. OVERLAP DETECTION: Detects dependencies for overlapping sub-regions
 *
 * Hash table with chaining:
 * - buckets[] array of head offsets
 * - Entries linked via next_in_bucket
 * - Insert at head (newest first) for sorted chains
 *
 * CRITICAL: Hash only by base_ptr
 * ==============================
 * For overlap detection to work, ALL sub-regions of the same base tensor
 * MUST be in the SAME hash bucket. This allows lookup to compare all
 * potentially overlapping regions.
 *
 * Overlap detection: Two regions create a dependency if:
 *   1. Same base_ptr (raw tensor pointer)
 *   2. Byte ranges [offset, offset+size) intersect
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#pragma once

#include "common.h"
#include "pto_runtime2_types.h"
#include "tensor.h"

struct PTO2OrchestratorState;  // forward declare

// =============================================================================
// TensorMap Structure
// =============================================================================

/**
 * TensorMap entry structure
 * Maps tensor region -> producer task ID
 *
 * Stored in ring buffer pool with lazy invalidation:
 * - Entry is valid only if producer_task_id >= last_task_alive
 * - Stale entries ignored during lookup
 * - Pool wraps around, overwriting stale entries
 *
 * Chain truncation optimization:
 * - Entries in bucket chains sorted by task_id (newest first)
 * - When lookup hits stale entry, truncate rest of chain
 */
struct PTO2TensorMapEntry {
    bool with_alloc{true};     // True if entry is task output, False if entry is task inout
    int32_t producer_task_id;  // Task that produces this region
    int32_t next_in_bucket;    // Offset to next entry in hash bucket (-1 = end)
    int32_t prev_in_bucket;    // Offset to prev entry in hash bucket (-1 = head is buckets[bucket])
    int32_t next_in_task;      // Offset to next entry for same task (-1 = end)
    int32_t prev_in_task;      // Offset to prev entry for same task (-1 = head is task_entry_head[slot])
    int32_t bucket_index;      // != -1 if entry is linked in a bucket chain
                               // CRITICAL: Must be set -1 before overwriting!
    uint64_t addr;
    Tensor tensor;             // Tensor descriptor key
};

/**
 * Stack-allocated lookup result (avoids heap allocation per lookup)
 */
#define PTO2_LOOKUP_MAX_RESULTS 16
// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================
struct PTO2LookupResult {
    struct Entry {
        int32_t entry_idx;
        OverlapStatus overlap_status;
    };
    Entry entries[PTO2_LOOKUP_MAX_RESULTS];
    int32_t count{0};

    void push(int32_t entry_idx, OverlapStatus s) {
        if (count < PTO2_LOOKUP_MAX_RESULTS) {
            entries[count++] = {entry_idx, s};
        }
    }
};

/**
 * TensorMap structure
 *
 * Hash table with ring buffer entry pool and lazy invalidation.
 */
struct PTO2TensorMap {
    // Hash table buckets (fixed size, power of 2)
    int32_t* buckets;     // Array of offsets into entry_pool (-1 = empty)
    int32_t num_buckets;  // Must be power of 2 for fast modulo

    // Entry pool as ring buffer
    PTO2TensorMapEntry* entry_pool;  // Ring buffer of entries
    int32_t* free_entry_list;        // free entry ids
    int32_t pool_size;               // Total pool capacity
    int32_t next_entry_idx;          // id when next entry insert
    int32_t free_num;                // free entry number in entry pool

    // Per-task entry tracking (for efficient bucket cleanup)
    int32_t* task_entry_head;  // Per-task head offset (-1 = no entries)
                               // Indexed by task_id % TASK_WINDOW_SIZE

    // Validity threshold (for lazy invalidation)
    int32_t last_task_alive;  // Cached value from shared memory

    PTO2OrchestratorState* orch{nullptr};

    int32_t new_entry() {
        if (free_num > 0) {
            int32_t res = free_entry_list[--free_num];
            debug_assert(entry_pool[res].bucket_index == -1);
            return res;
        }
        if (next_entry_idx < pool_size) {
            int32_t res = next_entry_idx++;
            debug_assert(entry_pool[res].bucket_index == -1);
            return res;
        }

        size_t wait_count = 0;
        while (free_num == 0) {
            sync_tensormap(true);
            always_assert(wait_count++ <= 1000000000UL);
        }
        debug_assert(free_num > 0);
        int32_t res = free_entry_list[--free_num];
        debug_assert(entry_pool[res].bucket_index != -1);
        return res;
    }
    void free_entry(int32_t entry_idx) {
        auto entry = &entry_pool[entry_idx];
        if (entry->bucket_index == -1) {
            return;  // Already removed
        }

        // Update predecessor's next pointer (O(1) via prev_in_bucket)
        if (entry->prev_in_bucket == -1) {
            // Entry is the head of its bucket chain, update bucket head
            // Must compute hash BEFORE clearing tensor (tensor.data() needs valid tensor_pool)
            buckets[entry->bucket_index] = entry->next_in_bucket;
        } else {
            entry_pool[entry->prev_in_bucket].next_in_bucket = entry->next_in_bucket;
        }

        // Update successor's prev pointer
        if (entry->next_in_bucket >= 0) {
            entry_pool[entry->next_in_bucket].prev_in_bucket = entry->prev_in_bucket;
        }

        // Clear tensor AFTER bucket chain manipulation (hash computation needs valid tensor)
        entry->tensor = Tensor();

        free_entry_list[free_num++] = entry_idx;
        entry->bucket_index = -1;
        entry->next_in_bucket = -1;
        entry->prev_in_bucket = -1;
        entry->next_in_task = -1;
        entry->prev_in_task = -1;
    }

    // =============================================================================
    // TensorMap API
    // =============================================================================

    /**
     * Initialize TensorMap
     *
     * @param num_buckets Number of hash buckets (must be power of 2)
     * @param pool_size   Size of entry pool
     * @return true on success, false on allocation failure
     */
    bool init(int32_t num_buckets, int32_t pool_size);

    /**
     * Initialize TensorMap with default sizes
     */
    bool init_default();

    /**
     * Destroy TensorMap and free resources
     */
    void destroy();

    /**
     * Reset TensorMap to empty state
     */
    void reset();

    /**
     * Update validity threshold from shared memory
     * Called periodically to refresh the lazy invalidation threshold.
     *
     * @param last_task_alive  Current value from shared memory
     */
    void sync_validity(int32_t last_task_alive) { this->last_task_alive = last_task_alive; }

    /**
     * Lookup producer for a tensor region
     *
     * Searches the hash table for a matching region.
     * Returns producer entry if found and valid.
     *
     * Chain truncation: When first stale entry is found, truncates
     * the rest of the chain (all subsequent entries are also stale).
     *
     * @param tensor  Tensor to look up
     * @param result  Output: stack-allocated result buffer
     */
    void lookup(const Tensor& tensor, PTO2LookupResult* result) {
        auto& query_tensor_data = tensor.data();
        uint32_t bucket_index = hash(query_tensor_data.buffer.addr);
        int32_t* prev_ptr = &buckets[bucket_index];  // For truncation
        int32_t entry_index = *prev_ptr;

        result->count = 0;

        while (entry_index >= 0) {
            PTO2TensorMapEntry& entry = entry_pool[entry_index];

            // Check validity first
            if (!entry_valid(entry)) {
                // ========== STALE ENTRY: Truncate chain here ==========
                // All subsequent entries are guaranteed to be stale too!
                // Truncate: unlink this and all following entries
                *prev_ptr = -1;  // Terminate chain at previous entry

                // Mark truncated entries as not in bucket (for correct reuse)
                while (entry_index >= 0) {
                    PTO2TensorMapEntry& stale = entry_pool[entry_index];
                    int32_t next = stale.next_in_bucket;
                    stale.bucket_index = -1;
                    stale.next_in_bucket = -1;
                    stale.prev_in_bucket = -1;
                    entry_index = next;
                }
                return;
            }

            // Entry is valid - check if regions OVERLAP (not just exact match)
            // Since we hash only by base_ptr, all entries in this bucket have
            // potential to overlap. We must check actual byte-range overlap.
            if (query_tensor_data.buffer.addr == entry.addr) {
                auto overlap_status = query_tensor_data.is_overlap(entry.tensor.data());
                if (overlap_status != OverlapStatus::NO_OVERLAP) {
                    result->push(entry_index, overlap_status);
                }
            }

            // Move to next entry
            prev_ptr = &entry.next_in_bucket;
            entry_index = *prev_ptr;
        }
    }

    /**
     * Insert a new entry (called when task produces output)
     *
     * Allocates from ring buffer pool, may overwrite stale entries.
     * Inserts at head of hash bucket chain (maintains task_id ordering).
     *
     * @param tm                TensorMap
     * @param tensor            Tensor produced
     * @param producer_task_id  Task ID of producer
     */
    void insert(const Tensor& tensor, int32_t producer_task_id, bool with_alloc) {
        // Allocate entry from ring buffer pool
        int32_t entry_index = new_entry();
        PTO2TensorMapEntry& entry = entry_pool[entry_index];

        // Initialize new entry
        entry.tensor = tensor;
        entry.producer_task_id = producer_task_id;
        entry.with_alloc = with_alloc;

        // Insert at head of hash bucket (maintains task_id descending order)
        entry.addr = tensor.data().buffer.addr;
        entry.bucket_index = hash(entry.addr);
        entry.next_in_bucket = buckets[entry.bucket_index];
        entry.prev_in_bucket = -1;  // New head has no predecessor
        // Update old head's prev pointer
        if (entry.next_in_bucket >= 0) {
            entry_pool[entry.next_in_bucket].prev_in_bucket = entry_index;
        }
        buckets[entry.bucket_index] = entry_index;

        // Link to task's entry list (for cleanup)
        int32_t task_slot = producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
        entry.next_in_task = task_entry_head[task_slot];
        entry.prev_in_task = -1;  // New head has no predecessor
        // Update old head's prev pointer
        if (entry.next_in_task >= 0) {
            entry_pool[entry.next_in_task].prev_in_task = entry_index;
        }
        task_entry_head[task_slot] = entry_index;
    }

    /**
     * Cleanup stale entries for retired tasks
     *
     * Called periodically by Orchestrator when last_task_alive advances.
     * Removes entries from bucket chains for tasks in [old, new) range.
     *
     * @param tm                   TensorMap
     * @param old_last_task_alive  Previous threshold
     * @param new_last_task_alive  New threshold
     */
    void cleanup_retired(int32_t old_last_task_alive, int32_t new_last_task_alive) {
        // Iterate through retired tasks and remove their entries from bucket chains
        for (int32_t task_id = old_last_task_alive; task_id < new_last_task_alive; task_id++) {
            int32_t task_slot = task_id & (PTO2_TASK_WINDOW_SIZE - 1);
            int32_t offset = task_entry_head[task_slot];

            while (offset >= 0) {
                PTO2TensorMapEntry* entry = &entry_pool[offset];
                int32_t next = entry->next_in_task;  // Save before clearing
                // Only remove if this entry belongs to the retiring task
                // (slot may have been reused by a newer task)
                if (entry->producer_task_id == task_id) {
                    // Clear task chain pointers (entire chain is being destroyed)
                    free_entry(offset);
                }
                offset = next;
            }

            // Clear task's entry head (slot will be reused by task_id + TASK_WINDOW_SIZE)
            task_entry_head[task_slot] = -1;
        }
    }

    // =============================================================================
    // Internal Helpers (exposed for testing)
    // =============================================================================

    /**
     * Compute hash for tensor addr
     */
    uint32_t hash(uint64_t key) {
        // Improve distribution by mixing bits (pointers often have aligned low bits)
        key = key ^ (key >> 16);
        key = key ^ (key >> 32);

        // Use bitwise AND for power-of-2 modulo (faster than %)
        return (uint32_t)(key & (num_buckets - 1));
    }

    /**
     * Check if entry is valid (producer has not retired)
     */
    bool entry_valid(const PTO2TensorMapEntry& entry) const {
        return entry.producer_task_id >= last_task_alive;
    }

    void remove_entry(int32_t entry_idx) {
        remove_from_task(entry_idx);
        free_entry(entry_idx);
    }

    /**
     * Remove entry from its task chain (O(1) with prev pointer)
     * Called during pool wrap-around to unlink reused entries.
     */
    void remove_from_task(int32_t entry_idx) {
        auto entry = &entry_pool[entry_idx];
        // Update predecessor's next pointer (O(1) via prev_in_task)
        if (entry->prev_in_task == -1) {
            // Entry is the head of its task chain, update task_entry_head
            int32_t task_slot = entry->producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
            task_entry_head[task_slot] = entry->next_in_task;
        } else {
            entry_pool[entry->prev_in_task].next_in_task = entry->next_in_task;
        }

        // Update successor's prev pointer
        if (entry->next_in_task >= 0) {
            entry_pool[entry->next_in_task].prev_in_task = entry->prev_in_task;
        }

        entry->next_in_task = -1;
        entry->prev_in_task = -1;
    }

    // =============================================================================
    // Debug Utilities
    // =============================================================================

    /**
     * Print TensorMap statistics
     */
    void print_stats();

    /**
     * Get count of valid entries
     */
    int32_t valid_count();

    // =============================================================================
    // TensorMap Synchronization
    // =============================================================================

    /**
     * Sync TensorMap validity threshold from shared memory
     *
     * Called periodically to refresh the lazy invalidation threshold.
     * Also triggers cleanup if threshold has advanced significantly.
     */
    void sync_tensormap(bool force = false);
};
