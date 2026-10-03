"""External GraspNet imports and model lifecycle, isolated from ROS callbacks."""

import gc
import os
import sys

from ament_index_python.packages import get_package_share_directory


def graspnet_source_path(*parts: str) -> str:
    return os.path.join(get_package_share_directory("graspnet_source"), *parts)


def load_graspnet_modules(baseline_dir: str):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for path in (
        baseline_dir,
        os.path.join(baseline_dir, "models"),
        os.path.join(baseline_dir, "dataset"),
        os.path.join(baseline_dir, "utils"),
        os.path.join(baseline_dir, "graspnetAPI"),
        os.path.join(baseline_dir, "pointnet2"),
        os.path.join(baseline_dir, "knn"),
    ):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    import torch
    from collision_detector import ModelFreeCollisionDetector
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup

    return (
        torch,
        GraspNet,
        pred_decode,
        CameraInfo,
        create_point_cloud_from_depth_image,
        GraspGroup,
        ModelFreeCollisionDetector,
    )


def load_model(torch, graspnet_type, checkpoint_path: str, device):
    model = graspnet_type(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    model.to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"]
    )
    model.eval()
    return model


def release_model(torch, model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
