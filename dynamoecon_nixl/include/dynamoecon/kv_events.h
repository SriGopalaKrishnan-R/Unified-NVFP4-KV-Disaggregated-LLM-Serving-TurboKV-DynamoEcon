#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
    #if defined(DYNAMOECON_BUILDING_DLL)
        #define DYNAMOECON_API __declspec(dllexport)
    #else
        #define DYNAMOECON_API
    #endif
#else
    #define DYNAMOECON_API __attribute__((visibility("default")))
#endif

namespace dynamoecon {

// ==============================================================================
// Section 3.4: NVIDIA Dynamo 1.x `--kv-events-config` Compatible Event Schema
// Matches Dynamo's KVPublisher / ZMQ KVEventBatch wire schema:
//   - BlockStored
//   - BlockRemoved
//   - AllBlocksCleared
// ==============================================================================

enum class KVEventType : uint8_t {
    BLOCK_STORED       = 0,
    BLOCK_REMOVED      = 1,
    ALL_BLOCKS_CLEARED = 2
};

struct DynamoKVEvent {
    uint64_t    event_id        = 0;
    KVEventType type            = KVEventType::BLOCK_STORED;
    int         worker_id       = 0;
    uint64_t    block_hash      = 0;
    uint64_t    parent_hash     = 0;
    int         token_count     = 16;
    uint64_t    size_bytes      = 0;
    double      retention_score = 0.0;
    double      timestamp_s     = 0.0;

    std::string to_dynamo_json() const {
        const char* tname = (type == KVEventType::BLOCK_STORED) ? "BlockStored"
                          : (type == KVEventType::BLOCK_REMOVED) ? "BlockRemoved"
                                                                  : "AllBlocksCleared";
        std::ostringstream os;
        os << "{\"event_id\":" << event_id
           << ",\"event_type\":\"" << tname << "\""
           << ",\"worker_id\":" << worker_id
           << ",\"block_hash\":\"" << std::hex << block_hash << std::dec << "\""
           << ",\"parent_hash\":\"" << std::hex << parent_hash << std::dec << "\""
           << ",\"token_count\":" << token_count
           << ",\"size_bytes\":" << size_bytes
           << ",\"s_retain\":" << retention_score
           << ",\"timestamp_s\":" << timestamp_s << "}";
        return os.str();
    }
};

class KVEventBus {
public:
    explicit KVEventBus(size_t max_history = 4096)
        : max_history_(max_history) {}

    void publish(DynamoKVEvent ev) {
        std::function<void(const DynamoKVEvent&)> cb;
        {
            std::lock_guard<std::mutex> lk(mu_);
            ev.event_id = ++next_id_;
            if (ring_.size() >= max_history_) {
                ring_.pop_front();
            }
            ring_.push_back(ev);
            cb = subscriber_;
        }
        if (cb) {
            cb(ev);
        }
    }

    void set_subscriber(std::function<void(const DynamoKVEvent&)> cb) {
        std::lock_guard<std::mutex> lk(mu_);
        subscriber_ = std::move(cb);
    }

    std::vector<DynamoKVEvent> recent_events(size_t max_count = 64) const {
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = std::min(max_count, ring_.size());
        return std::vector<DynamoKVEvent>(ring_.end() - static_cast<std::ptrdiff_t>(n), ring_.end());
    }

    uint64_t total_published() const {
        std::lock_guard<std::mutex> lk(mu_);
        return next_id_;
    }

private:
    size_t max_history_;
    mutable std::mutex mu_;
    uint64_t next_id_ = 0;
    std::deque<DynamoKVEvent> ring_;
    std::function<void(const DynamoKVEvent&)> subscriber_;
};

} // namespace dynamoecon
