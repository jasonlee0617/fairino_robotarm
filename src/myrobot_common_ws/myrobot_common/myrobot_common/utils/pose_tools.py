from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R


class PoseTools:
    def __init__(self, node, base_frame: str):
        self._node = node
        self.base_frame = base_frame

    def make_pose(self, x, y, z, roll_deg, pitch_deg, yaw_deg) -> Pose:
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        q = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        return pose

    def to_pose_stamped(self, pose: Pose, frame_id: str | None = None) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = frame_id if frame_id else self.base_frame
        ps.header.stamp = self._node.get_clock().now().to_msg()
        ps.pose = pose
        return ps


__all__ = ["PoseTools"]
