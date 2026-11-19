#include "dynamoecon/radix_kv_indexer.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <mutex>

namespace dynamoecon {

ConcurrentRadixKVIndexer::ConcurrentRadixKVIndexer(
    int num_workers,
    uint64_t worker_hbm_capacity_bytes,
    int block_tokens,
    EvictionPolicy policy,
    KVEventBus* event_bus)
    : num_workers_(std::clamp(num_workers, 1, 64))
    , worker_capacity_bytes_(worker_hbm_capacity_bytes)
    , watermark_90pct_bytes_(static_cast<uint64_t>(static_cast<double>(worker_hbm_capacity_bytes) * 0.90))
    , block_tokens_(std::max(1, block_tokens))
    , policy_(policy)
    , event_bus_(event_bus)
{
    for (auto& wb : worker_bytes_) {
        wb.store(0ULL, std::memory_order_relaxed);
    }
}

double ConcurrentRadixKVIndexer::compute_retention_score(
    double flops_recompute,
    double p_reuse,
    uint64_t size_bytes,
    double delta_t_idle_s) noexcept
{
    double denom = static_cast<double>(std::max<uint64_t>(1ULL, size_bytes)) *
                   (1.0 + std::max(0.0, delta_t_idle_s));
    return (std::max(0.0, flops_recompute) * std::clamp(p_reuse, 1e-4, 1.0)) / denom;
}

void ConcurrentRadixKVIndexer::insert_prefix_blocks(
    int worker_id,
    const uint64_t* block_hashes,
    int num_blocks,
    uint64_t bytes_per_block,
    double base_flops_per_block,
    double now_s)
{
    if (worker_id < 0 || worker_id >= num_workers_ || num_blocks <= 0 || block_hashes == nullptr) {
        return;
    }

    const uint64_t wbit = (1ULL << worker_id);
    const size_t s_idx  = stripe_index(block_hashes[0]);

    {
        std::unique_lock<std::shared_mutex> lk(stripes_[s_idx].mu);
        Stripe& st = stripes_[s_idx];

        RadixNode* curr = nullptr;
        uint64_t parent_h = 0ULL;

        for (int b = 0; b < num_blocks; ++b) {
            uint64_t h = block_hashes[b];
            auto* map_ptr = (b == 0) ? &st.roots : &curr->children;
            auto it = map_ptr->find(h);

            // Positional prefill FLOPs grow super-linearly with depth due to causal attention!
            double positional_flops = base_flops_per_block * (1.0 + 0.12 * static_cast<double>(b + 1));

            if (it == map_ptr->end()) {
                auto node = std::make_unique<RadixNode>();
                node->block_hash      = h;
                node->parent_hash     = parent_h;
                node->worker_mask     = wbit;
                node->size_bytes      = bytes_per_block;
                node->flops_recompute = positional_flops;
                node->p_reuse_ema     = 0.35;
                node->last_access_s   = now_s;
                node->depth_blocks    = b + 1;
                curr = node.get();
                st.by_hash[h] = curr;
                (*map_ptr)[h] = std::move(node);

                worker_bytes_[worker_id].fetch_add(bytes_per_block, std::memory_order_relaxed);
                if (event_bus_) {
                    DynamoKVEvent ev{};
                    ev.type            = KVEventType::BLOCK_STORED;
                    ev.worker_id       = worker_id;
                    ev.block_hash      = h;
                    ev.parent_hash     = parent_h;
                    ev.token_count     = block_tokens_;
                    ev.size_bytes      = bytes_per_block;
                    ev.retention_score = compute_retention_score(positional_flops, 0.35, bytes_per_block, 0.0);
                    ev.timestamp_s     = now_s;
                    event_bus_->publish(ev);
                }
            } else {
                curr = it->second.get();
                bool newly_placed = ((curr->worker_mask & wbit) == 0ULL);
                curr->worker_mask |= wbit;
                curr->p_reuse_ema   = std::min(1.0, 0.80 * curr->p_reuse_ema + 0.20 * 1.0);
                curr->last_access_s = std::max(curr->last_access_s, now_s);
                if (newly_placed) {
                    worker_bytes_[worker_id].fetch_add(curr->size_bytes, std::memory_order_relaxed);
                    if (event_bus_) {
                        DynamoKVEvent ev{};
                        ev.type            = KVEventType::BLOCK_STORED;
                        ev.worker_id       = worker_id;
                        ev.block_hash      = h;
                        ev.parent_hash     = parent_h;
                        ev.token_count     = block_tokens_;
                        ev.size_bytes      = curr->size_bytes;
                        ev.retention_score = compute_retention_score(curr->flops_recompute, curr->p_reuse_ema, curr->size_bytes, 0.0);
                        ev.timestamp_s     = now_s;
                        event_bus_->publish(ev);
                    }
                }
            }
            parent_h = h;
        }
    }

    if (worker_bytes_[worker_id].load(std::memory_order_relaxed) >= watermark_90pct_bytes_) {
        evict_if_above_watermark(worker_id, now_s);
    }
}

void ConcurrentRadixKVIndexer::evict_if_above_watermark(int worker_id, double now_s) {
    const uint64_t wbit       = (1ULL << worker_id);
    // Evict down to 85% hysteresis band once 90% HBM watermark is breached
    const uint64_t target_cap = static_cast<uint64_t>(static_cast<double>(worker_capacity_bytes_) * 0.85);
    if (worker_bytes_[worker_id].load(std::memory_order_relaxed) <= target_cap) return;

    struct EvictCandidate {
        double   metric;
        size_t   stripe;
        uint64_t hash;
    };

    // Up to 4 multi-level leaf sweeps (handles pruning entire multi-block branches cleanly in O(N)!)
    for (int pass = 0; pass < 4 && worker_bytes_[worker_id].load(std::memory_order_relaxed) > target_cap; ++pass) {
        std::vector<EvictCandidate> candidates;
        candidates.reserve(256);

        for (size_t s = 0; s < kNumStripes; ++s) {
            std::shared_lock<std::shared_mutex> slk(stripes_[s].mu);
            for (const auto& kv : stripes_[s].by_hash) {
                const RadixNode* nd = kv.second;
                if ((nd->worker_mask & wbit) == 0ULL) continue;

                bool has_child_on_worker = false;
                for (const auto& ch : nd->children) {
                    if ((ch.second->worker_mask & wbit) != 0ULL) {
                        has_child_on_worker = true;
                        break;
                    }
                }
                if (has_child_on_worker) continue;

                double metric = 0.0;
                if (policy_ == EvictionPolicy::TOKCONOMICS_S_RETAIN) {
                    double idle = std::max(0.0, now_s - nd->last_access_s);
                    metric = compute_retention_score(nd->flops_recompute, nd->p_reuse_ema, nd->size_bytes, idle);
                } else {
                    metric = nd->last_access_s;
                }
                candidates.push_back({metric, s, nd->block_hash});
            }
        }

        if (candidates.empty()) break;

        std::sort(candidates.begin(), candidates.end(),
                  [](const EvictCandidate& a, const EvictCandidate& b) { return a.metric < b.metric; });

        for (const auto& cand : candidates) {
            if (worker_bytes_[worker_id].load(std::memory_order_relaxed) <= target_cap) break;

            std::unique_lock<std::shared_mutex> ulk(stripes_[cand.stripe].mu);
            auto it = stripes_[cand.stripe].by_hash.find(cand.hash);
            if (it == stripes_[cand.stripe].by_hash.end()) continue;

            RadixNode* vnode = it->second;
            if ((vnode->worker_mask & wbit) == 0ULL) continue;

            double idle = std::max(0.0, now_s - vnode->last_access_s);
            double s_ret = compute_retention_score(vnode->flops_recompute, vnode->p_reuse_ema, vnode->size_bytes, idle);
            vnode->worker_mask &= ~wbit;
            uint64_t cur_b = worker_bytes_[worker_id].load(std::memory_order_relaxed);
            uint64_t sub_b = std::min(cur_b, vnode->size_bytes);
            worker_bytes_[worker_id].fetch_sub(sub_b, std::memory_order_relaxed);

            total_evictions_.fetch_add(1, std::memory_order_relaxed);
            evicted_bytes_.fetch_add(vnode->size_bytes, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> mlk(evict_metric_mu_);
                sum_evicted_s_retain_ += s_ret;
            }
            if (event_bus_) {
                DynamoKVEvent ev{};
                ev.type            = KVEventType::BLOCK_REMOVED;
                ev.worker_id       = worker_id;
                ev.block_hash      = vnode->block_hash;
                ev.parent_hash     = vnode->parent_hash;
                ev.token_count     = block_tokens_;
                ev.size_bytes      = vnode->size_bytes;
                ev.retention_score = s_ret;
                ev.timestamp_s     = now_s;
                event_bus_->publish(ev);
            }
        }
    }
}

RadixLookupResult ConcurrentRadixKVIndexer::match_prefix(
    const uint64_t* block_hashes,
    int num_blocks,
    uint64_t candidate_worker_mask,
    double now_s,
    bool update_ema_on_hit)
{
    RadixLookupResult res{};
    if (num_blocks <= 0 || block_hashes == nullptr || candidate_worker_mask == 0ULL) {
        return res;
    }

    total_lookups_.fetch_add(1, std::memory_order_relaxed);
    total_requested_tokens_.fetch_add(static_cast<uint64_t>(num_blocks) * block_tokens_, std::memory_order_relaxed);

    const size_t s_idx = stripe_index(block_hashes[0]);
    uint64_t alive_mask = candidate_worker_mask;

    if (update_ema_on_hit) {
        std::unique_lock<std::shared_mutex> lk(stripes_[s_idx].mu);
        auto root_it = stripes_[s_idx].roots.find(block_hashes[0]);
        if (root_it == stripes_[s_idx].roots.end()) return res;

        RadixNode* curr = root_it->second.get();
        for (int b = 0; b < num_blocks && curr != nullptr; ++b) {
            alive_mask &= curr->worker_mask;
            if (alive_mask == 0ULL) break;

            curr->p_reuse_ema   = std::min(1.0, 0.85 * curr->p_reuse_ema + 0.15 * 1.0);
            curr->last_access_s = std::max(curr->last_access_s, now_s);

            uint64_t tmp = alive_mask;
            while (tmp != 0ULL) {
                int w = std::countr_zero(tmp);
                tmp &= (tmp - 1ULL);
                res.worker_matched_tokens[w] = (b + 1) * block_tokens_;
                if (res.worker_matched_tokens[w] > res.matched_tokens_best) {
                    res.matched_tokens_best = res.worker_matched_tokens[w];
                    res.best_prefill_worker = w;
                }
            }
            if (b + 1 < num_blocks) {
                auto ch_it = curr->children.find(block_hashes[b + 1]);
                curr = (ch_it != curr->children.end()) ? ch_it->second.get() : nullptr;
            }
        }
    } else {
        // Pure lock-shared reader fast path: zero writer contention across concurrent router threads!
        std::shared_lock<std::shared_mutex> slk(stripes_[s_idx].mu);
        auto root_it = stripes_[s_idx].roots.find(block_hashes[0]);
        if (root_it == stripes_[s_idx].roots.end()) return res;

        const RadixNode* curr = root_it->second.get();
        for (int b = 0; b < num_blocks && curr != nullptr; ++b) {
            alive_mask &= curr->worker_mask;
            if (alive_mask == 0ULL) break;

            uint64_t tmp = alive_mask;
            while (tmp != 0ULL) {
                int w = std::countr_zero(tmp);
                tmp &= (tmp - 1ULL);
                res.worker_matched_tokens[w] = (b + 1) * block_tokens_;
                if (res.worker_matched_tokens[w] > res.matched_tokens_best) {
                    res.matched_tokens_best = res.worker_matched_tokens[w];
                    res.best_prefill_worker = w;
                }
            }
            if (b + 1 < num_blocks) {
                auto ch_it = curr->children.find(block_hashes[b + 1]);
                curr = (ch_it != curr->children.end()) ? ch_it->second.get() : nullptr;
            }
        }
    }

    total_matched_tokens_.fetch_add(static_cast<uint64_t>(res.matched_tokens_best), std::memory_order_relaxed);
    return res;
}

int ConcurrentRadixKVIndexer::migrate_worker_blocks(int dead_worker_id, int backup_worker_id, double now_s) {
    if (dead_worker_id < 0 || dead_worker_id >= num_workers_ ||
        backup_worker_id < 0 || backup_worker_id >= num_workers_ ||
        dead_worker_id == backup_worker_id) {
        return 0;
    }

    const uint64_t dead_bit   = (1ULL << dead_worker_id);
    const uint64_t backup_bit = (1ULL << backup_worker_id);
    int migrated = 0;

    for (size_t s = 0; s < kNumStripes; ++s) {
        std::unique_lock<std::shared_mutex> lk(stripes_[s].mu);
        for (auto& kv : stripes_[s].by_hash) {
            RadixNode* nd = kv.second;
            if ((nd->worker_mask & dead_bit) != 0ULL) {
                nd->worker_mask &= ~dead_bit;
                if ((nd->worker_mask & backup_bit) == 0ULL) {
                    nd->worker_mask |= backup_bit;
                    worker_bytes_[backup_worker_id].fetch_add(nd->size_bytes, std::memory_order_relaxed);
                }
                nd->last_access_s = std::max(nd->last_access_s, now_s);
                ++migrated;
            }
        }
    }
    worker_bytes_[dead_worker_id].store(0ULL, std::memory_order_relaxed);
    return migrated;
}

uint64_t ConcurrentRadixKVIndexer::worker_used_bytes(int worker_id) const noexcept {
    if (worker_id < 0 || worker_id >= num_workers_) return 0ULL;
    return worker_bytes_[worker_id].load(std::memory_order_relaxed);
}

double ConcurrentRadixKVIndexer::worker_hbm_utilization(int worker_id) const noexcept {
    if (worker_capacity_bytes_ == 0ULL) return 0.0;
    return static_cast<double>(worker_used_bytes(worker_id)) / static_cast<double>(worker_capacity_bytes_);
}

EvictionStats ConcurrentRadixKVIndexer::stats() const noexcept {
    EvictionStats st{};
    st.total_evictions        = total_evictions_.load(std::memory_order_relaxed);
    st.evicted_bytes          = evicted_bytes_.load(std::memory_order_relaxed);
    st.total_lookups          = total_lookups_.load(std::memory_order_relaxed);
    st.total_matched_tokens   = total_matched_tokens_.load(std::memory_order_relaxed);
    st.total_requested_tokens = total_requested_tokens_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> mlk(evict_metric_mu_);
        st.avg_evicted_s_retain = (st.total_evictions > 0)
            ? (sum_evicted_s_retain_ / static_cast<double>(st.total_evictions))
            : 0.0;
    }
    return st;
}

} // namespace dynamoecon
