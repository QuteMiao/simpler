# Runtime Logic: tensormap_and_ringbuffer

## Overview
The tensormap_and_ringbuffer runtime (RT2) builds the graph on device using a TensorMap for dependency discovery and ring buffers for task slots, dependency lists, and packed output memory. Orchestration and scheduling communicate through PTO2 shared memory, and AICPU scheduler threads dispatch tasks to AICore using PTO2 dispatch payloads.

## Core Components
- `Runtime` holds handshake buffers, orchestration state, shared memory pointers, and the kernel address map. See `src/runtime/tensormap_and_ringbuffer/runtime/runtime.h`.
- `PTO2SharedMemory` stores task descriptors, dependency lists, and ring-buffer state. See `src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`.
- `PTO2OrchestratorState` owns the TensorMap, ring buffers, and scope stack for buffer lifetimes. See `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`.
- `PTO2SchedulerState` reads shared memory and drives ready-queue scheduling. See `src/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`.
- `PTO2DispatchPayload` is the per-core handoff structure used for AICore dispatch. See `src/runtime/tensormap_and_ringbuffer/runtime/pto2_dispatch_payload.h`.

## Host Init Flow (Device Orchestration)
1. `init_runtime_impl` registers kernel binaries and stores function addresses in `Runtime::func_id_to_addr_`. See `src/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`.
2. Input and output tensor pointers are converted to device pointers using `arg_types` and `arg_sizes`. Input buffers are copied to device, output buffers are recorded for copy-back.
3. The orchestration SO is embedded in `Runtime` for AICPU-side `dlopen`, and a device copy is allocated for cleanup tracking.
4. GM heap and PTO2 shared memory are allocated on device. Pointers are stored via `Runtime::set_pto2_gm_heap` and `Runtime::set_pto2_gm_sm_ptr`.
5. Optional overrides are read from environment. `PTO2_READY_QUEUE_SHARDS` controls scheduler queue sharding. `PTO2_RING_TASK_WINDOW`, `PTO2_RING_HEAP`, and `PTO2_RING_DEP_POOL` override ring sizes.

## Device Orchestration Flow
1. AICPU thread 3 loads the orchestration SO and resolves `aicpu_orchestration_entry` (and optional `aicpu_orchestration_config`). See `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`.
2. The executor creates a `PTO2Runtime` from shared memory with `pto2_runtime_create_from_sm`, which wires the orchestrator and scheduler to the shared-memory rings.
3. Orchestration runs inside an outer scope. The orchestration code uses `pto_orchestration_api.h` helpers to submit tasks and manage scopes.
4. Each `pto2_submit_task` performs a fixed sequence. It syncs TensorMap validity, allocates a task slot, collects input dependencies via TensorMap lookup, allocates packed output buffers from the heap ring, registers outputs, finalizes the fanin list, and updates shared memory. See `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`.
5. The executor signals orchestration completion in shared memory and exposes an orchestrator-ready queue to scheduler threads for fast wakeups.

## Scheduling And Dispatch Flow
1. Scheduler threads read `PTO2TaskDescriptor` and dependency lists from shared memory and build a `PTO2DispatchPayload` for each task.
2. Ready tasks are assigned to AICore via the per-core handshake, just like other runtimes, but using the PTO2 payload rather than a `Task` struct.
3. Completion updates per-task fanin refcounts and frees ring-buffer slots as `last_task_alive` advances.

## Finalize And Cleanup
`validate_runtime_impl` copies output tensors back to host. If a packed output buffer is present in the shared memory header (`graph_output_ptr`), it is used for the first output tensor. All device allocations recorded in tensor pairs are freed afterward. See `src/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`.

## Key Files
- `src/runtime/tensormap_and_ringbuffer/runtime/runtime.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- `src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.cpp`
- `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
