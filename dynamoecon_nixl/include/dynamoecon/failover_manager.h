#pragma once

#include "dynamoecon/crossover_scheduler.h"
#include "dynamoecon/radix_kv_indexer.h"
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace dynamoecon {

// ==============================================================================
// NFR-2.3: Sub-10ms Decode Worker Heartbeat Failure & KV-Session Re-Anchoring
//
// Tracks active inference sessions anchored on Decode workers alongside their
// source Prefill worker and KV prefix block hashes. When a Decode worker fails
// its heartbeat check, `handle_worker_failure()` atomically re-anchors every
// affected session onto the lowest-queue healthy secondary Decode worker and
// migrates Radix-indexed KV ownership within <= 10ms with ZERO request loss!
// ==============================================================================

struct ActiveKVSession {
    uint64_t              session_id         = 0;
    int                   prefill_worker_id  = 0;
    int                   decode_worker_id   = 0;
    std::vector<uint64_t> block_hashes;
    int                   generated_tokens   = 0;
};

struct FailoverReport {
    int      failed_decode_worker     = -1;
    int      sessions_reanchored      = 0;
    int      sessions_lost            = 0;   // NFR-2.3: MUST equal 0
    int      kv_blocks_migrated       = 0;
    int      primary_backup_worker    = -1;
    double   failover_latency_ms      = 0.0; // NFR-2.3: MUST be <= 10.0 ms
};

class DYNAMOECON_API FailoverManager {
public:
    explicit FailoverManager(ConcurrentRadixKVIndexer& indexer);

    void register_session(const ActiveKVSession& session);
    void complete_session(uint64_t session_id);
    size_t active_session_count() const;

    // Triggers immediate heartbeat failure recovery for `failed_decode_worker`
    FailoverReport handle_worker_failure(
        int failed_decode_worker,
        std::vector<WorkerQueueState>& cluster_workers,
        double now_s);

    std::vector<ActiveKVSession> sessions_on_worker(int decode_worker_id) const;

private:
    ConcurrentRadixKVIndexer& indexer_;
    mutable std::mutex        mu_;
    std::unordered_map<uint64_t, ActiveKVSession> sessions_;
};

} // namespace dynamoecon
