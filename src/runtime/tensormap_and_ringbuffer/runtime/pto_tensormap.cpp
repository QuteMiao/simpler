/**
 * PTO Runtime2 - TensorMap Implementation
 *
 * Implements TensorMap with ring buffer pool, lazy invalidation,
 * and chain truncation optimization.
 *
 * Key features:
 * 1. O(1) insert at bucket head
 * 2. O(valid_entries) lookup with chain truncation
 * 3. Automatic stale entry cleanup during lookup
 * 4. Periodic explicit cleanup for long chains
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_tensormap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "pto_orchestrator.h"

// =============================================================================
// Initialization and Destruction
// =============================================================================

bool PTO2TensorMap::init(int32_t new_num_buckets, int32_t new_pool_size) {
    // Validate power of 2 for fast modulo
    if ((new_num_buckets & (new_num_buckets - 1)) != 0) {
        return false;  // num_buckets must be power of 2
    }

    // Allocate buckets
    buckets = (PTO2TensorMapEntry**)malloc(new_num_buckets * sizeof(PTO2TensorMapEntry*));
    if (!buckets) {
        return false;
    }

    // Initialize all buckets to empty (-1)
    for (int32_t i = 0; i < new_num_buckets; i++) {
        buckets[i] = nullptr;
    }

    num_buckets = new_num_buckets;

    // Allocate entry pool
    entry_pool = (PTO2TensorMapEntry*)calloc(new_pool_size, sizeof(PTO2TensorMapEntry));
    if (!entry_pool) {
        free(buckets);
        buckets = NULL;
        return false;
    }

    // Allocate free entry list
    free_entry_list = (PTO2TensorMapEntry**)calloc(new_pool_size, sizeof(PTO2TensorMapEntry*));
    if (!free_entry_list) {
        free(buckets);
        free(entry_pool);
        buckets = NULL;
        entry_pool = NULL;
        return false;
    }

    pool_size = new_pool_size;
    next_entry_idx = 0;
    free_num = 0;

    // Initialize all entries as not in bucket
    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool[i].bucket_index = -1;
        entry_pool[i].next_in_bucket = nullptr;
        entry_pool[i].prev_in_bucket = nullptr;
        entry_pool[i].next_in_task = nullptr;
        entry_pool[i].prev_in_task = nullptr;
        entry_pool[i].producer_task_id = -1;
    }

    // Allocate per-task entry tracking
    task_entry_head = (PTO2TensorMapEntry**)malloc(new_pool_size * sizeof(PTO2TensorMapEntry*));
    if (!task_entry_head) {
        free(entry_pool);
        free(buckets);
        free(free_entry_list);
        entry_pool = NULL;
        buckets = NULL;
        free_entry_list = NULL;
        return false;
    }

    // Initialize all task entry heads to -1 (no entries)
    for (int32_t i = 0; i < PTO2_TASK_WINDOW_SIZE; i++) {
        task_entry_head[i] = nullptr;
    }

    last_task_alive = 0;

    return true;
}

bool PTO2TensorMap::init_default() {
    return init(PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE);
}

void PTO2TensorMap::destroy() {
    if (buckets) {
        free(buckets);
        buckets = NULL;
    }

    if (entry_pool) {
        free(entry_pool);
        entry_pool = NULL;
    }

    if (task_entry_head) {
        free(task_entry_head);
        task_entry_head = NULL;
    }

    if (free_entry_list) {
        free(free_entry_list);
        free_entry_list = NULL;
    }
}

void PTO2TensorMap::reset() {
    // Reset all buckets to empty
    for (int32_t i = 0; i < num_buckets; i++) {
        buckets[i] = nullptr;
    }

    // Reset all entries
    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool[i].bucket_index = -1;
        entry_pool[i].next_in_bucket = nullptr;
        entry_pool[i].prev_in_bucket = nullptr;
        entry_pool[i].next_in_task = nullptr;
        entry_pool[i].prev_in_task = nullptr;
        entry_pool[i].producer_task_id = -1;
    }

    // Reset per-task entry tracking
    for (int32_t i = 0; i < pool_size; i++) {
        task_entry_head[i] = nullptr;
    }

    next_entry_idx = 0;
    free_num = 0;
    last_task_alive = 0;
}

// =============================================================================
// Debug Utilities
// =============================================================================

void PTO2TensorMap::print_stats() {
    int32_t valid = 0;
    int32_t stale = 0;
    int32_t empty_buckets = 0;
    int32_t max_chain = 0;
    int64_t total_chain = 0;
    int32_t non_empty_buckets = 0;

    // Count entries
    for (int32_t i = 0; i < pool_size; i++) {
        if (entry_pool[i].bucket_index != -1) {
            if (entry_valid(entry_pool[i])) {
                valid++;
            } else {
                stale++;
            }
        }
    }

    // Count bucket stats
    for (int32_t b = 0; b < num_buckets; b++) {
        int32_t chain_len = 0;
        auto cur_entry = buckets[b];

        while (cur_entry != nullptr) {
            chain_len++;
            cur_entry = cur_entry->next_in_bucket;
        }

        if (chain_len == 0) {
            empty_buckets++;
        } else {
            non_empty_buckets++;
            total_chain += chain_len;
            if (chain_len > max_chain) {
                max_chain = chain_len;
            }
        }
    }

    printf("=== TensorMap Statistics ===\n");
    printf("Pool size:           %d\n", pool_size);
    printf("Pool next entry idx: %d\n", next_entry_idx);
    printf("Pool free_num:       %d\n", free_num);
    printf("Num buckets:         %d\n", num_buckets);
    printf("Valid entries:       %d\n", valid);
    printf("Stale entries:       %d\n", stale);
    printf("Empty buckets:       %d\n", empty_buckets);
    printf("Max chain len:       %d\n", max_chain);
    printf("Avg chain len:       %.2f\n", non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0);
    printf("Last task alive:     %d\n", last_task_alive);
    printf("============================\n");
}

int32_t PTO2TensorMap::valid_count() {
    int32_t count = 0;

    for (int32_t i = 0; i < pool_size; i++) {
        if (entry_pool[i].bucket_index != -1 && entry_valid(entry_pool[i])) {
            count++;
        }
    }

    return count;
}

void PTO2TensorMap::sync_tensormap() {
    constexpr int MIN_FREE_NUM = 1024;
    always_assert(orch != nullptr);
    while(true) {
        // Read current last_task_alive from shared memory
        int32_t new_last_task_alive = PTO2_LOAD_ACQUIRE(&orch->sm_handle->header->last_task_alive);
        sync_validity(new_last_task_alive);
        if ((pool_size - next_entry_idx + free_num < MIN_FREE_NUM) || new_last_task_alive - orch->tensormap_last_cleanup >= PTO2_TENSORMAP_CLEANUP_INTERVAL) {
            cleanup_retired(orch->tensormap_last_cleanup, new_last_task_alive);
            orch->tensormap_last_cleanup = new_last_task_alive;
        } else {
            break;
        }
    }
}
