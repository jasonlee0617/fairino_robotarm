from dataclasses import dataclass
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive


@dataclass
class KeepoutConfig:
    object_id: str = "z_keepout"
    frame_id: str = "base_link"
    thickness: float = 0.06
    xy_size: float = 0.5


class KeepoutManager:
    """
    在规划场景中维护一个“低于某高度的禁入区域”（一个大盒子）。
    常用来防止机械臂末端或连杆下探碰撞桌面/地面。
    """

    def __init__(self, node, collision_obj_pub, planning_scene_pub, config: KeepoutConfig):
        self._node = node
        self._collision_obj_pub = collision_obj_pub
        self._planning_scene_pub = planning_scene_pub
        self._cfg = config
        self.enabled = False

    def _make_collision_object(self, z_min: float) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = self._cfg.frame_id
        co.id = self._cfg.object_id

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(self._cfg.xy_size), float(self._cfg.xy_size), float(self._cfg.thickness)]

        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x = 0.0
        pose.position.y = 0.0
        # 盒子“顶面”对齐到 z_min，因此中心在 z_min - thickness/2
        pose.position.z = float(z_min - self._cfg.thickness / 2.0)

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        return co

    def enable(self, z_min: float):
        co = self._make_collision_object(z_min)

        # 兼容你原来两种发布（CollisionObject + PlanningScene diff）
        self._collision_obj_pub.publish(co)

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self._planning_scene_pub.publish(ps)

        self.enabled = True
        self._node.get_logger().info(f"[KEEP_OUT] ENABLED top_z={z_min:.3f}")

    def disable(self):
        co = CollisionObject()
        co.header.frame_id = self._cfg.frame_id
        co.id = self._cfg.object_id
        co.operation = CollisionObject.REMOVE

        self._collision_obj_pub.publish(co)

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self._planning_scene_pub.publish(ps)

        self.enabled = False
        self._node.get_logger().info("[KEEP_OUT] DISABLED")
