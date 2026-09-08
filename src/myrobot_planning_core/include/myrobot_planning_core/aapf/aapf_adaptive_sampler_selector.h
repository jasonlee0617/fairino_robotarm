#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <string>

#include "myrobot_planning_core/aapf/aapf_guided_sampler.h"

namespace fairino_planning {

inline constexpr int kAapfSampleSourceCount = 2;

inline int aapfSampleSourceIndex(AapfSampleSource source) {
    return static_cast<int>(source);
}

inline const char* aapfSampleSourceName(AapfSampleSource source) {
    switch (source) {
    case AapfSampleSource::kGuided: return "guided";
    case AapfSampleSource::kGlobal: return "global";
    }
    return "global";
}

class AapfAdaptiveSamplerSelector {
public:
    AapfAdaptiveSamplerSelector(int window, double exploration, int global_min_period)
        : window_(std::max(1, window))
        , exploration_(std::max(0.0, exploration))
        , global_min_period_(std::max(0, global_min_period)) {}

    AapfSampleSource choose() {
        ++selection_count_;
        if (global_min_period_ > 0 && selection_count_ % global_min_period_ == 0) {
            return AapfSampleSource::kGlobal;
        }
        int warmup_source = -1;
        for (int i = 0; i < kAapfSampleSourceCount; ++i) {
            if (history_[i].size() < kWarmupPulls &&
                (warmup_source < 0 || history_[i].size() < history_[warmup_source].size())) {
                warmup_source = i;
            }
        }
        if (warmup_source >= 0) {
            return static_cast<AapfSampleSource>(warmup_source);
        }

        int total = 0;
        double max_rate = 0.0;
        std::array<double, kAapfSampleSourceCount> rates{};
        for (int i = 0; i < kAapfSampleSourceCount; ++i) {
            double utility = 0.0;
            double elapsed_ms = 0.0;
            for (const auto& sample : history_[i]) {
                utility += sample.utility;
                elapsed_ms += sample.elapsed_ms;
            }
            rates[i] = utility / std::max(elapsed_ms, 1e-6);
            max_rate = std::max(max_rate, rates[i]);
            total += static_cast<int>(history_[i].size());
        }

        int best = 0;
        double best_score = -1.0;
        for (int i = 0; i < kAapfSampleSourceCount; ++i) {
            const double exploitation = max_rate > 0.0 ? rates[i] / max_rate : 0.0;
            const double exploration = exploration_ * std::sqrt(
                std::log(static_cast<double>(total) + 1.0) /
                static_cast<double>(history_[i].size()));
            const double score = exploitation + exploration;
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        return static_cast<AapfSampleSource>(best);
    }

    void observe(AapfSampleSource source, double utility, double elapsed_ms) {
        auto& samples = history_[aapfSampleSourceIndex(source)];
        samples.push_back({std::clamp(utility, 0.0, 1.0), std::max(0.0, elapsed_ms)});
        while (static_cast<int>(samples.size()) > window_) {
            samples.pop_front();
        }
    }

    int selectionCount() const { return selection_count_; }

private:
    struct Observation {
        double utility;
        double elapsed_ms;
    };

    static constexpr std::size_t kWarmupPulls = 2;
    int window_;
    double exploration_;
    int global_min_period_;
    int selection_count_{0};
    std::array<std::deque<Observation>, kAapfSampleSourceCount> history_{};
};

}  // namespace fairino_planning
