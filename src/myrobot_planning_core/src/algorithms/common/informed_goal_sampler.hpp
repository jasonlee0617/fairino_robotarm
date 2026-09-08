#pragma once

#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include "myrobot_planning_core/config/planning_params.hpp"

namespace fairino_planning::informed_sampling {

inline JointConfig unitBall(std::mt19937& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    JointConfig sample;
    for (int joint = 0; joint < NUM_JOINTS; ++joint) sample[joint] = normal(rng);
    const double norm = sample.norm();
    if (norm < 1e-12) {
        sample.setZero();
        sample[0] = 1.0;
    } else {
        sample /= norm;
    }
    return sample * std::pow(uniform(rng), 1.0 / NUM_JOINTS);
}

inline JointConfig ellipsoid(
    const JointConfig& start, const JointConfig& goal, double best_cost, std::mt19937& rng) {
    const JointConfig delta = goal - start;
    const double c_min = delta.norm();
    if (best_cost <= c_min + 1e-12) return goal;
    JointConfig scaled = unitBall(rng);
    scaled[0] *= 0.5 * best_cost;
    const double minor = 0.5 * std::sqrt(std::max(0.0, best_cost * best_cost - c_min * c_min));
    for (int joint = 1; joint < NUM_JOINTS; ++joint) scaled[joint] *= minor;
    JointConfig axis = JointConfig::Zero();
    axis[0] = 1.0;
    Eigen::Matrix<double, NUM_JOINTS, NUM_JOINTS> rotation =
        Eigen::Matrix<double, NUM_JOINTS, NUM_JOINTS>::Identity();
    if (c_min > 1e-12) {
        JointConfig reflector = axis - delta / c_min;
        if (reflector.norm() > 1e-12) {
            reflector.normalize();
            rotation -= 2.0 * reflector * reflector.transpose();
        }
    }
    return 0.5 * (start + goal) + rotation * scaled;
}

inline std::optional<JointConfig> singleRoot(
    const JointConfig& start, const JointConfig& goal, double best_cost,
    const JointLimits& limits, std::mt19937& rng) {
    if ((goal - start).norm() + 1e-12 >= best_cost) return std::nullopt;
    const JointConfig sample = ellipsoid(start, goal, best_cost, rng);
    return limits.isWithin(sample) ? std::optional<JointConfig>{sample} : std::nullopt;
}

inline std::optional<JointConfig> multiRootUnion(
    const JointConfig& start, const std::vector<JointConfig>& goals, double best_cost,
    const JointLimits& limits, std::mt19937& rng) {
    std::vector<const JointConfig*> eligible;
    for (const auto& goal : goals) {
        if ((goal - start).norm() + 1e-12 < best_cost) eligible.push_back(&goal);
    }
    if (eligible.empty()) return std::nullopt;
    std::uniform_int_distribution<size_t> select(0U, eligible.size() - 1U);
    for (int attempt = 0; attempt < 64; ++attempt) {
        const JointConfig sample = ellipsoid(start, *eligible[select(rng)], best_cost, rng);
        if (limits.isWithin(sample)) return sample;
    }
    return std::nullopt;
}

}  // namespace fairino_planning::informed_sampling
