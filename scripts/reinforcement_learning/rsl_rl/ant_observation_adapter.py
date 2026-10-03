"""Shared opt-in Ant scan selection and checkpoint input validation for inference."""

PATTERN_ALIASES = {
    "original": "original", "original_9x7": "original",
    "dense": "dense", "dense_13x9": "dense",
    "forward_dense": "forward_dense", "forward_dense_13x7": "forward_dense",
}
SCAN_DIMS = {"original": 63, "dense": 117, "forward_dense": 91}


def canonical_pattern(name):
    if name not in PATTERN_ALIASES:
        raise ValueError(f"Unknown height scan pattern: {name}")
    return PATTERN_ALIASES[name]


def configure_height_scan(env_cfg, enabled=False, pattern=None):
    """Modify only this runtime observation/sensor config; legacy flag means original."""
    if not enabled and pattern is None:
        return
    from isaaclab_tasks.manager_based.classic.ant.ant_terrain_heightscan_env_cfg import add_height_scan_for_evaluation

    selected = canonical_pattern(pattern or "original")
    add_height_scan_for_evaluation(env_cfg)
    if selected != "original":
        from isaaclab_tasks.manager_based.classic.ant.ant_heightscan_variants_env_cfg import (
            dense_height_scan_pattern_cfg,
            forward_dense_height_scan_pattern_cfg,
        )
        factory = dense_height_scan_pattern_cfg if selected == "dense" else forward_dense_height_scan_pattern_cfg
        env_cfg.scene.height_scanner.pattern_cfg = factory()


def configure_compact_terrain(env_cfg, enabled=False, pattern=None):
    """Opt-in training-identical compact inputs; never add a second scanner.

    Call after configure_height_scan. Without --height_scan the policy gets only
    compact inputs; with it, the original raw scan is retained before compact.
    """
    if not enabled:
        return
    if pattern is not None and canonical_pattern(pattern) != "original":
        raise ValueError("Compact descriptors require the Original 9x7 pattern, not dense/forward_dense.")
    if getattr(env_cfg.observations.policy, "foot_contacts", None) is not None:
        raise ValueError("Compact policies in this experiment do not use explicit Contact inputs.")
    from isaaclab_tasks.manager_based.classic.ant.ant_compact_terrain_env_cfg import (
        AntCompactObservationsCfg, AntHeightScanCompactObservationsCfg,
    )
    from isaaclab_tasks.manager_based.classic.ant.ant_terrain_heightscan_env_cfg import height_scanner_cfg

    raw_enabled = getattr(env_cfg.observations.policy, "height_scan", None) is not None
    env_cfg.scene.height_scanner = height_scanner_cfg()
    env_cfg.scene.clone_in_fabric = False
    env_cfg.observations = AntHeightScanCompactObservationsCfg() if raw_enabled else AntCompactObservationsCfg()


def validate_compact_scan(env, policy_observation):
    """Diagnostic-only check: the manager's last 18 inputs equal the training helper."""
    import torch
    from isaaclab.envs.mdp import height_scan
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.classic.ant.ant_compact_terrain_observations import compact_from_height_scan

    unclipped = height_scan(env, SceneEntityCfg("height_scanner"), offset=0.5)
    # Native raycast_mesh returns inf for missed hits (e.g. GapPath). Training
    # already uses this EXACT clipping; do not add a replacement/sanitization.
    raw = unclipped.clamp(-1.0, 1.0)
    descriptor = compact_from_height_scan(raw)
    assert raw.shape == (env.num_envs, 63) and descriptor.shape == (env.num_envs, 18)
    assert torch.isfinite(raw).all() and torch.isfinite(descriptor).all()
    assert torch.equal(policy_observation[:, -18:], descriptor)
    return {"raw_scan_finite": True, "compact_descriptor_finite": True,
            "compact_matches_training_helper": True,
            "raw_scan_definition": "training Original normalized/clipped 63-D observation",
            "unclipped_scan_finite": bool(torch.isfinite(unclipped).all()),
            "missed_ray_count": int(torch.isinf(unclipped).sum())}


def scan_metadata(env, selected=None):
    """Check native sensor buffers and requested pattern outside the observation path."""
    import torch

    raw_enabled = getattr(env.cfg.observations.policy, "height_scan", None) is not None
    compact_enabled = getattr(env.cfg.observations.policy, "compact_terrain", None) is not None
    if not raw_enabled and not compact_enabled:
        return {"height_scan_enabled": False, "height_scan_pattern": None, "scan_dim": 0}
    scanner = env.scene["height_scanner"]
    pattern = selected or next((name for name, count in SCAN_DIMS.items() if count == scanner.num_rays), None)
    if pattern is None:
        raise ValueError(f"Unrecognized Ant height scan ray count: {scanner.num_rays}")
    pattern = canonical_pattern(pattern)
    assert scanner.num_rays == SCAN_DIMS[pattern]
    assert scanner._num_envs == env.num_envs
    assert tuple(scanner._timestamp_last_update.shape) == (env.num_envs,)
    assert tuple(scanner.data.ray_hits_w.shape) == (env.num_envs, SCAN_DIMS[pattern], 3)
    expected, directions = scanner.cfg.pattern_cfg.func(scanner.cfg.pattern_cfg, scanner.device)
    expected += torch.tensor(scanner.cfg.offset.pos, dtype=expected.dtype, device=expected.device)
    assert torch.equal(scanner.ray_starts[0], expected)
    result = {"height_scan_enabled": raw_enabled, "height_scan_pattern": pattern,
              "scan_dim": scanner.num_rays if raw_enabled else 0}
    if compact_enabled:
        from isaaclab_tasks.manager_based.classic.ant.ant_compact_terrain_observations import DESCRIPTOR_ORDER
        assert pattern == "original"
        result.update(compact_enabled=True, compact_dim=18, internal_scanner="original_63",
                      internal_scanner_rays=63, compact_descriptor_order=list(DESCRIPTOR_ORDER),
                      raw_heightscan_dim_for_policy=63 if raw_enabled else 0)
    return result


def validate_checkpoint_inputs(checkpoint_path, env):
    """Fail explicitly on dimension mismatch; never pad observations or migrate weights."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    actor_dim = int(state["actor.0.weight"].shape[1])
    critic_dim = int(state["critic.0.weight"].shape[1])
    observation_dim = int(env.observation_manager.group_obs_dim["policy"][0])
    if actor_dim != observation_dim or critic_dim != observation_dim:
        raise ValueError(
            f"Ant checkpoint input mismatch: actor={actor_dim}, critic={critic_dim}, "
            f"environment policy observation={observation_dim}. Select the matching height-scan/contact inputs."
        )
    return {"actor_input_dim": actor_dim, "critic_input_dim": critic_dim, "observation_dim": observation_dim}
