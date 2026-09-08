#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "myrobot_planning_core/request/plan_request_core.hpp"
#include "myrobot_planning_core/result/plan_result.hpp"

namespace fairino_planning {

class SearchRuntime {
public:
    SearchRuntime(
        const PlanRequestCore* request,
        std::chrono::steady_clock::time_point started)
        : request_(request), started_(request && request->started != std::chrono::steady_clock::time_point{}
              ? request->started : started) {
        if (request_) {
            checkpoints_ = request_->anytime_checkpoints_s;
            checkpoints_.erase(
                std::remove_if(checkpoints_.begin(), checkpoints_.end(),
                    [](double value) { return !std::isfinite(value) || value <= 0.0; }),
                checkpoints_.end());
            std::sort(checkpoints_.begin(), checkpoints_.end());
            checkpoints_.erase(std::unique(checkpoints_.begin(), checkpoints_.end()), checkpoints_.end());
        }
    }

    bool shouldStop() const {
        return request_ && request_->shouldStop();
    }

    bool deadlineReached() const {
        return request_ && request_->hasDeadline() &&
            std::chrono::steady_clock::now() >= request_->deadline;
    }

    double elapsedSeconds() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_).count();
    }

    void capture(PlanResult& result, bool has_solution, double incumbent_cost, int nodes = 0) {
        const double elapsed = elapsedSeconds();
        while (next_checkpoint_ < checkpoints_.size() &&
               elapsed + 1e-12 >= checkpoints_[next_checkpoint_]) {
            AnytimeTracePoint point;
            point.elapsed_s = checkpoints_[next_checkpoint_++];
            point.has_solution = has_solution;
            point.incumbent_joint_cost_rad =
                has_solution ? incumbent_cost : std::numeric_limits<double>::infinity();
            point.iterations = result.iterations;
            point.sample_attempts = result.sample_attempts;
            point.accepted_samples = result.accepted_samples;
            point.num_nodes = nodes;
            point.work_units = result.effort_stats.work_units > 0
                ? result.effort_stats.work_units : result.iterations;
            result.anytime_trace.push_back(point);
        }
    }

    void markStopReason(PlanResult& result, bool safety_guard) const {
        if (request_ && request_->cancel_requested && request_->cancel_requested()) {
            result.stop_reason = "cancelled";
        } else if (deadlineReached()) {
            result.stop_reason = result.success ? "deadline_incumbent" : "deadline_no_solution";
        } else if (result.effort_stats.post_solution_budget_complete) {
            result.stop_reason = "post_solution_budget_complete";
        } else if (safety_guard) {
            result.stop_reason = "iteration_guard";
        } else if (result.lower_bound_certified) {
            result.stop_reason = "lower_bound_certified";
        } else {
            result.stop_reason = result.success ? "completed_with_incumbent" : "no_solution";
        }
    }

private:
    const PlanRequestCore* request_{nullptr};
    std::chrono::steady_clock::time_point started_{};
    std::vector<double> checkpoints_;
    std::size_t next_checkpoint_{0U};
};

}  // namespace fairino_planning
