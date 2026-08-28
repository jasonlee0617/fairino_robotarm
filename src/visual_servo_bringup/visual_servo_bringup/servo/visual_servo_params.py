"""Validated, axis-symmetric runtime parameters for position servo."""

from __future__ import annotations

from dataclasses import dataclass

from myrobot_common.utils.params import param


def _int_list(node, name: str, default: list[int]) -> set[int]:
    raw = param(node, name, default)
    if isinstance(raw, (int, float, str)):
        raw = [raw]
    return {int(value) for value in raw}


def _resolve_controller_type(controller_type: str) -> tuple[str, str, str]:
    controller_type = str(controller_type).strip().upper()
    if controller_type in {"PID", "PD", "PI_FF", "ADAPTIVE_PID"}:
        return controller_type, "PID", controller_type
    if controller_type in {"LADRC", "NLADRC", "MPC"}:
        return controller_type, controller_type, "NONE"
    raise RuntimeError("servo_controller_type must be PID, PD, PI_FF, ADAPTIVE_PID, LADRC, NLADRC or MPC")


@dataclass(frozen=True)
class ServoRuntimeConfig:
    servo_controller_type: str
    servo_controller_family: str
    pid_variant: str
    servo_align_xyz_tol: float
    servo_status_decel_codes: set[int]
    servo_status_halt_codes: set[int]
    predict_lead_sec: float
    max_predict_horizon: float
    cmd_lpf_alpha: float
    servo_detection_timeout: float
    vel_ff_gain: float
    rel_vel_damping_gain: float
    ff_vel_ema_alpha: float
    max_target_speed: float
    target_vxyz_clip: float
    meas_jump_clip_xyz: float
    ee_vel_ema_alpha: float
    rel_vel_clip: float
    ff_term_clip: float
    v_xyz_max: float
    a_xyz_max: float
    target_accel_ema_alpha: float
    aligned_stable_count: int
    ladrc_wc: float
    ladrc_wo: float
    ladrc_b0: float
    ladrc_ff_mix_gain: float
    nladrc_wc: float
    nladrc_wo: float
    nladrc_b0: float
    nladrc_alpha_obs: float
    nladrc_alpha_obs2: float
    nladrc_delta_obs: float
    nladrc_obs_error_clip: float
    nladrc_obs_transition: float
    nladrc_z2_clip: float
    nladrc_u_fb_clip: float
    nladrc_z2_decay_band: float
    nladrc_z2_decay_gain: float
    nladrc_z2_gain: float
    nladrc_ff_mix_gain: float
    nladrc_u_clip: float
    nladrc_u_rate_max: float
    nladrc_u_ema_alpha: float
    mpc_ts: float
    mpc_horizon: int
    mpc_tau: float
    mpc_delay_sec: float
    mpc_delay_steps: int
    mpc_q_e: float
    mpc_q_v: float
    mpc_q_terminal: float
    mpc_r_u: float
    mpc_r_du: float
    mpc_u_max: float
    mpc_du_max: float
    mpc_norm_clip: float
    ff_age_start_sec: float
    ff_age_ref_sec: float
    ff_age_window_sec: float
    ff_age_floor_scale: float
    ff_err_norm_threshold: float
    ff_large_err_scale: float
    slew_dv_trigger: float
    slew_alpha_high: float
    slew_alpha_low: float
    twist_norm_max: float
    status1_speed_scale: float
    servo_handoff_zero_twist_count: int
    handoff_target_delta_max: float
    handoff_target_speed_max: float
    pid_kp: float
    pid_ki: float
    pid_kd: float
    pid_d_ema_alpha: float
    pid_derivative_clip: float
    pid_integral_limit: float
    pid_integral_active_radius: float
    pid_integral_decay: float
    pid_u_max: float
    adaptive_pid_kp: float
    adaptive_pid_ki: float
    adaptive_pid_kd: float
    adaptive_pid_schedule_alpha: float
    adaptive_pid_kp_min: float
    adaptive_pid_kp_max: float
    adaptive_pid_kd_min: float
    adaptive_pid_kd_max: float
    adaptive_pid_err_low: float
    adaptive_pid_err_high: float
    adaptive_pid_derr_low: float
    adaptive_pid_derr_high: float

    @classmethod
    def from_node(cls, node) -> "ServoRuntimeConfig":
        controller_type, family, variant = _resolve_controller_type(param(node, "servo_controller_type", "LADRC"))
        values = dict(
            servo_controller_type=controller_type, servo_controller_family=family, pid_variant=variant,
            servo_align_xyz_tol=float(param(node, "servo_align_xyz_tol", 0.003)),
            servo_status_decel_codes=_int_list(node, "servo_status_decel_codes", [1, 6]),
            servo_status_halt_codes=_int_list(node, "servo_status_halt_codes", [2, 4, 5]),
            predict_lead_sec=float(param(node, "predict_lead_sec", 0.025)), max_predict_horizon=float(param(node, "max_predict_horizon", 0.12)),
            cmd_lpf_alpha=float(param(node, "cmd_lpf_alpha", 0.7)), servo_detection_timeout=float(param(node, "servo_detection_timeout", 0.20)),
            vel_ff_gain=float(param(node, "vel_ff_gain", 0.9)), rel_vel_damping_gain=float(param(node, "rel_vel_damping_gain", 1.2)),
            ff_vel_ema_alpha=float(param(node, "ff_vel_ema_alpha", 0.65)), max_target_speed=float(param(node, "max_target_speed", 1.5)),
            target_vxyz_clip=float(param(node, "target_vxyz_clip", 1.5)), meas_jump_clip_xyz=float(param(node, "meas_jump_clip_xyz", 0.004)),
            ee_vel_ema_alpha=float(param(node, "ee_vel_ema_alpha", 0.7)), rel_vel_clip=float(param(node, "rel_vel_clip", 1.5)),
            ff_term_clip=float(param(node, "ff_term_clip", 1.5)), v_xyz_max=float(param(node, "v_xyz_max", 1.5)),
            a_xyz_max=float(param(node, "a_xyz_max", 1.5)), target_accel_ema_alpha=float(param(node, "target_accel_ema_alpha", 0.25)),
            aligned_stable_count=int(param(node, "aligned_stable_count", 40)),
            ladrc_wc=float(param(node, "ladrc_wc", 10.0)), ladrc_wo=float(param(node, "ladrc_wo", 25.0)),
            ladrc_b0=float(param(node, "ladrc_b0", 0.5)), ladrc_ff_mix_gain=float(param(node, "ladrc_ff_mix_gain", 0.2)),
            nladrc_wc=float(param(node, "nladrc_wc", 12.5)), nladrc_wo=float(param(node, "nladrc_wo", 25.0)),
            nladrc_b0=float(param(node, "nladrc_b0", 0.5)), nladrc_alpha_obs=float(param(node, "nladrc_alpha_obs", 0.98)),
            nladrc_alpha_obs2=float(param(node, "nladrc_alpha_obs2", 0.95)), nladrc_delta_obs=float(param(node, "nladrc_delta_obs", 0.004)),
            nladrc_obs_error_clip=float(param(node, "nladrc_obs_error_clip", 0.02)), nladrc_obs_transition=float(param(node, "nladrc_obs_transition", 0.012)),
            nladrc_z2_clip=float(param(node, "nladrc_z2_clip", 0.14)), nladrc_u_fb_clip=float(param(node, "nladrc_u_fb_clip", 1.5)),
            nladrc_z2_decay_band=float(param(node, "nladrc_z2_decay_band", 0.004)), nladrc_z2_decay_gain=float(param(node, "nladrc_z2_decay_gain", 3.0)),
            nladrc_z2_gain=float(param(node, "nladrc_z2_gain", 1.0)), nladrc_ff_mix_gain=float(param(node, "nladrc_ff_mix_gain", 0.3)),
            nladrc_u_clip=float(param(node, "nladrc_u_clip", 1.5)), nladrc_u_rate_max=float(param(node, "nladrc_u_rate_max", 1.5)),
            nladrc_u_ema_alpha=float(param(node, "nladrc_u_ema_alpha", 1.0)),
            mpc_ts=float(param(node, "mpc_ts", 0.005)), mpc_horizon=int(param(node, "mpc_horizon", 16)), mpc_tau=float(param(node, "mpc_tau", 0.01)),
            mpc_delay_sec=float(param(node, "mpc_delay_sec", 0.005)), mpc_delay_steps=int(param(node, "mpc_delay_steps", 1)),
            mpc_q_e=float(param(node, "mpc_q_e", 320.0)), mpc_q_v=float(param(node, "mpc_q_v", 2.0)), mpc_q_terminal=float(param(node, "mpc_q_terminal", 480.0)),
            mpc_r_u=float(param(node, "mpc_r_u", 0.16)), mpc_r_du=float(param(node, "mpc_r_du", 2.0)), mpc_u_max=float(param(node, "mpc_u_max", 1.5)),
            mpc_du_max=float(param(node, "mpc_du_max", 0.0075)), mpc_norm_clip=float(param(node, "mpc_norm_clip", 1.5)),
            ff_age_start_sec=float(param(node, "ff_age_start_sec", 0.08)), ff_age_ref_sec=float(param(node, "ff_age_ref_sec", 0.09)),
            ff_age_window_sec=float(param(node, "ff_age_window_sec", 0.05)), ff_age_floor_scale=float(param(node, "ff_age_floor_scale", 0.7)),
            ff_err_norm_threshold=float(param(node, "ff_err_norm_threshold", 0.0065)), ff_large_err_scale=float(param(node, "ff_large_err_scale", 0.6)),
            slew_dv_trigger=float(param(node, "slew_dv_trigger", 0.03)), slew_alpha_high=float(param(node, "slew_alpha_high", 1.0)),
            slew_alpha_low=float(param(node, "slew_alpha_low", 0.7)), twist_norm_max=float(param(node, "twist_norm_max", 1.5)),
            status1_speed_scale=float(param(node, "status1_speed_scale", 0.4)), servo_handoff_zero_twist_count=int(param(node, "servo_handoff_zero_twist_count", 5)),
            handoff_target_delta_max=float(param(node, "handoff_target_delta_max", 0.01)), handoff_target_speed_max=float(param(node, "handoff_target_speed_max", 0.005)),
            pid_kp=float(param(node, "pid_kp", 15.0)), pid_ki=float(param(node, "pid_ki", 6.0)), pid_kd=float(param(node, "pid_kd", 0.0)),
            pid_d_ema_alpha=float(param(node, "pid_d_ema_alpha", 0.8)), pid_derivative_clip=float(param(node, "pid_derivative_clip", 1.0)),
            pid_integral_limit=float(param(node, "pid_integral_limit", 0.003)), pid_integral_active_radius=float(param(node, "pid_integral_active_radius", 0.01)),
            pid_integral_decay=float(param(node, "pid_integral_decay", 0.94)), pid_u_max=float(param(node, "pid_u_max", 1.5)),
            adaptive_pid_kp=float(param(node, "adaptive_pid_kp", 50.0)), adaptive_pid_ki=float(param(node, "adaptive_pid_ki", 0.0)),
            adaptive_pid_kd=float(param(node, "adaptive_pid_kd", 10.0)), adaptive_pid_schedule_alpha=float(param(node, "adaptive_pid_schedule_alpha", 0.8)),
            adaptive_pid_kp_min=float(param(node, "adaptive_pid_kp_min", 9.0)), adaptive_pid_kp_max=float(param(node, "adaptive_pid_kp_max", 9.0)),
            adaptive_pid_kd_min=float(param(node, "adaptive_pid_kd_min", 0.0)), adaptive_pid_kd_max=float(param(node, "adaptive_pid_kd_max", 0.0)),
            adaptive_pid_err_low=float(param(node, "adaptive_pid_err_low", 0.004)), adaptive_pid_err_high=float(param(node, "adaptive_pid_err_high", 0.025)),
            adaptive_pid_derr_low=float(param(node, "adaptive_pid_derr_low", 0.0001)), adaptive_pid_derr_high=float(param(node, "adaptive_pid_derr_high", 0.006)),
        )
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _resolve_controller_type(self.servo_controller_type)
        positive = (
            "servo_align_xyz_tol", "v_xyz_max", "a_xyz_max", "twist_norm_max", "max_target_speed", "target_vxyz_clip",
            "meas_jump_clip_xyz", "ladrc_wc", "ladrc_wo", "ladrc_b0", "nladrc_wc", "nladrc_wo", "nladrc_b0",
            "nladrc_delta_obs", "nladrc_obs_error_clip", "nladrc_obs_transition", "nladrc_z2_clip", "nladrc_u_fb_clip",
            "nladrc_z2_decay_band", "nladrc_u_clip", "nladrc_u_rate_max", "mpc_ts", "mpc_u_max", "mpc_du_max", "mpc_norm_clip",
            "pid_derivative_clip", "pid_u_max",
        )
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise RuntimeError(f"{name} must be > 0")
        if self.mpc_horizon <= 0 or self.aligned_stable_count <= 0 or self.servo_handoff_zero_twist_count <= 0:
            raise RuntimeError("MPC horizon, aligned_stable_count and servo_handoff_zero_twist_count must be > 0")
        if not (0.0 < self.nladrc_alpha_obs <= 1.0 and 0.0 < self.nladrc_alpha_obs2 <= 1.0 and 0.0 < self.nladrc_u_ema_alpha <= 1.0):
            raise RuntimeError("NLADRC alpha parameters must be in (0, 1]")
        if not (0.0 <= self.pid_d_ema_alpha <= 1.0 and 0.0 <= self.pid_integral_decay <= 1.0):
            raise RuntimeError("PID smoothing parameters must be in [0, 1]")
