#pragma once
#include "myrobot_planning_core/types.h"
#include "myrobot_planning_core/dh_kinematics.h"
#include "myrobot_planning_core/ik/ik_candidate.h"
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace fairino_planning {

// ==========================================================================
// ConstraintLevel
// ==========================================================================
enum class ConstraintLevel {
    CriticalHard, IndustrialHard, SoftPreference, Diagnostic
};

enum class IKTaskProfile {
    Grasp,
    Continuous,
};
const char* toString(IKTaskProfile profile);

// ==========================================================================
// IKSelectParams — 5 groups, all loaded from ik_params.yaml
// ==========================================================================
struct IKSelectParams {
    IKTaskProfile task_profile = IKTaskProfile::Grasp;

    // ── Group 1: manipulability ──
    double mu_eps               = 1.0e-4;
    double alpha_manipulability = 2.0;
    double sigma_hard_flange    = 0.03;
    double sigma_hard_gripper   = 0.035;
    double cond_hard_max        = 400.0;
    double sigma_min_threshold  = 0.001;

    // ── Group 2: continuity ──
    Eigen::Matrix<double, NUM_JOINTS, NUM_JOINTS> W_move =
        (Eigen::Matrix<double,6,1>() << 8.0, 3.0, 4.0, 6.0, 8.0, 5.0).finished().asDiagonal();
    double alpha_continuity = 0.5;
    double cost_eps         = 1.0;
    double lexicographic_eps = 0.005;
    bool   enable_continuity_guard = true;
    double max_joint_step_rad = 1.20;
    double max_wrist_step_rad = 0.90;
    bool   branch_switch_hard_reject = true;
    double branch_switch_min_step_rad = 0.25;
    double hint_seed_sync_max_rad = 0.50;
    double cartesian_stream_max_pos_step_m = 0.03;
    double cartesian_stream_max_rot_step_rad = 0.35;
    int continuous_max_goal_roots = 4;
    double continuous_goal_root_min_separation_rad = 0.3;

    // ── Group 3: posture ──
    double upper_arm_min_z_soft   = 0.12;
    double upper_arm_min_z_hard   = 0.04;
    double forearm_min_z_soft     = 0.10;
    double forearm_min_z_hard     = 0.035;
    double wrist_chain_min_z_soft = 0.08;
    double wrist_chain_min_z_hard = 0.03;
    double anti_gravity_soft      = 0.5;
    double anti_gravity_hard      = 0.2;

    // ── Group 4: wrist ──
    double q4_inner_soft_start    = 0.20;
    double q4_inner_hard_max      = 1.35;
    double alpha_q4_inner         = 4.0;
    double q4_positive_weight     = 2.0;
    double q5_ref                 = M_PI / 2.0;
    double q5_ref_weight          = 0.3;
    double forearm_tool_angle_soft = 1.05;  // 60 deg
    double forearm_tool_angle_hard = 0.65;  // 37 deg
    double alpha_wrist_fold       = 4.0;

    // ── Group 5: joint safety ──
    double joint_margin_hard_rad  = 0.05;
    double wrist_sin_min          = 0.08;
    double elbow_sin_min          = 0.06;
    double base_radius_min        = 0.03;

    // ── seed delta (cross-cutting) ──
    bool   enable_seed_delta_hard_filter = false;
    bool   allow_large_motion_fallback   = true;
    std::array<double, NUM_JOINTS> seed_delta_soft_start{{2.0, 1.3, 1.8, 2.0, 2.0, 2.0}};
    std::array<double, NUM_JOINTS> seed_delta_soft_weight{{4.0, 2.0, 3.0, 4.0, 5.0, 4.0}};

    // ── scoring weights (dimensionless [0,1] range each) ──
    double S1_continuity      = 0.30;
    double S2_manipulability  = 0.25;
    double S3_posture         = 0.30;
    double S4_joint_safety    = 0.15;

    // ── rejection ──
    bool reject_q2_positive = false;
    bool reject_q4_positive = false;

    // ── debug ──
    bool debug_log_all_candidates    = true;
    int  debug_max_candidates_to_log = 8;
    bool debug_print_degrees         = true;
    int  debug_log_every_n_calls     = 10;
    bool debug_always_log_failures   = true;

    // ── task profiles ──
    bool   grasp_hard_reject_low_arm = true;
    bool   grasp_hard_reject_wrist_fold = true;
    bool   grasp_allow_industrial_fallback = false;
    double grasp_upper_arm_min_z_hard = 0.08;
    double grasp_forearm_min_z_hard = 0.08;
    double grasp_wrist_chain_min_z_hard = 0.05;
    double grasp_q4_inner_hard_max = 0.20;
    double grasp_forearm_tool_angle_hard = 0.80;
    bool   continuous_enforce_branch_guard = true;
    bool   continuous_enforce_consistency_limits = true;
    ToolParams gripper_tool = ToolParams::gripper();
};

// ==========================================================================
// ScoreBreakdown — 4 normalized dimensions, all in [0,1]
// ==========================================================================
struct ScoreBreakdown {
    double S1_continuity{0.0};
    double S2_manipulability{0.0};
    double S3_posture{0.0};
    double S4_joint_safety{0.0};
    double total{0.0};
};

struct RankedCandidate {
    IKCandidate candidate;
    ScoreBreakdown score;
    bool critical_pass{true};
    int  industrial_violation_count{0};
    bool branch_changed{false};
};

// ==========================================================================
// IKRejectReason + IKCandidateDiagnostic
// ==========================================================================
enum class IKRejectReason {
    kAccepted = 0, kOutsideLimits, kSigmaTooSmall, kConditionTooLarge,
    kJointMarginTooSmall, kWristSinTooSmall, kElbowSinTooSmall,
    kBaseRadiusTooSmall, kSeedDeltaTooLarge, kRejectQ4InnerFold,
    kRejectQ2Positive, kRejectQ4Positive, kContinuityJump, kBranchSwitch,
    kConsistencyLimit, kLowArmHeight, kWristFold,
};
const char* toString(IKRejectReason reason);

struct IKCandidateDiagnostic {
    JointConfig q{JointConfig::Zero()};
    IKQualityMetrics metrics{};
    bool passed_hard_filter{false};
    IKRejectReason reject_reason{IKRejectReason::kAccepted};
    bool wrist_flip{false};
    double S1{0}, S2{0}, S3{0}, S4{0};
    double total_cost{std::numeric_limits<double>::infinity()};
    double dq_norm{0.0};
    double max_abs_dq{0.0};
    bool branch_changed{false};
    bool selected{false};
};

struct IKSelectionRequest {
    const std::vector<JointConfig>* solutions{nullptr};
    JointConfig seed{JointConfig::Zero()};
    Transform4d target_pose{Transform4d::Identity()};
    ToolModel tool_model{ToolModel::FLANGE};
    IKTaskProfile task_profile{IKTaskProfile::Grasp};
    const IKBranchHint* hint{nullptr};
    std::vector<double> consistency_limits;
};

struct IKSelectionResult {
    std::optional<JointConfig> selected;
    std::vector<IKCandidateDiagnostic> diagnostics;
    IKQualityMetrics metrics{};
};

// ==========================================================================
// IKSelector
// ==========================================================================
class IKSelector {
public:
    IKSelector();
    explicit IKSelector(const IKSelectParams& params);
    void setToolTransform(const Transform4d& flange_to_tool);

    IKSelectionResult select(const IKSelectionRequest& request) const;

    static JointConfig unwrapNearSeed(
        const JointConfig& q_wrapped, const JointConfig& q_seed, const JointLimits& limits);

private:
    IKSelectParams params_;
    DHKinematics   fk_;
    JointLimits    limits_;

    // metrics
    IKQualityMetrics evaluateMetrics(const JointConfig& q, ToolModel model) const;
    ManipulabilityFeatures computeManipulability(const IKQualityMetrics& m) const;

    // features
    LinkPostureFeatures computeLinkPosture(const JointConfig& q) const;
    WristPostureFeatures computeWristPosture(const JointConfig& q, ToolModel model) const;
    MotionFeatures computeMotion(const JointConfig& q, const JointConfig& q_seed) const;

    // scoring (4 dimensions, each ∈ [0,1])
    double scoreS1_continuity(const MotionFeatures& m) const;
    double scoreS2_manipulability(const ManipulabilityFeatures& mu) const;
    double scoreS3_posture(const JointConfig& q, const LinkPostureFeatures& l,
                           const WristPostureFeatures& w, const Transform4d& target,
                           ToolModel model) const;
    double scoreS4_jointSafety(const JointConfig& q) const;

    ScoreBreakdown scoreCandidate(const IKCandidate& c, const Transform4d& target,
                                  ToolModel model) const;
    bool better(const RankedCandidate& a, const RankedCandidate& b,
                IKTaskProfile profile) const;

    // filtering
    bool passCriticalHardFilters(const JointConfig& q, ToolModel model,
        const IKQualityMetrics& m, IKRejectReason* rr) const;

    // utils
    static double segmentMinZ(const Eigen::Vector3d& a, const Eigen::Vector3d& b, int samples);
    static double softBarrierBelow(double value, double soft, double hard);
};

}  // namespace fairino_planning
