#!/usr/bin/env python3
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String


def trajectory_event_for_command(command: str):
    """Map motion-control commands to MoveIt's supported event values."""
    command = str(command).strip().lower()
    return "stop" if command in ("stop", "reset") else None


def motion_command_for_key(key: str) -> str:
    """Map the shared interactive safety keys to motion commands."""
    return {" ": "stop", "g": "g", "h": "reset", "r": "resume"}.get(
        str(key).lower(), ""
    )


class MotionControlNode(Node):
    def __init__(self):
        super().__init__("motion_control")
        self.abort_pub = self.create_publisher(Bool, "/manual_abort", 10)
        self.command_pub = self.create_publisher(String, "/motion_control/command", 10)
        self.event_pub = self.create_publisher(String, "/trajectory_execution_event", 1)
        self.command_burst_count = int(self.declare_parameter("command_burst_count", 3).value)
        self.command_burst_period_sec = float(self.declare_parameter("command_burst_period_sec", 0.01).value)
        self.keyboard_poll_period_sec = float(self.declare_parameter("keyboard_poll_period_sec", 0.01).value)
        self.command_sub = self.create_subscription(
            String, "/motion_control/command", self._relay_command, 10
        )

        self.input_stream = None
        self.fd = None
        self.old = None
        self._opened_tty = None
        self._configure_keyboard()

        if self.input_stream is None:
            self.timer = None
            self.get_logger().warn(
                "No TTY available; keyboard disabled. "
                "/motion_control/command relay is active."
            )
        else:
            self.timer = self.create_timer(max(0.005, self.keyboard_poll_period_sec), self.tick)
            self.get_logger().info(
                "Keys: g confirm grasp, SPACE stop, h reset, r resume."
            )

    def _configure_keyboard(self):
        stream = None
        opened_tty = None
        try:
            if sys.stdin.isatty():
                stream = sys.stdin
            else:
                opened_tty = open("/dev/tty", "r")
                stream = opened_tty

            fd = stream.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (OSError, termios.error):
            if opened_tty is not None:
                opened_tty.close()
            return

        self.input_stream = stream
        self.fd = fd
        self.old = old
        self._opened_tty = opened_tty

    def _publish_command(self, command: str):
        count = max(1, self.command_burst_count)
        period = max(0.0, self.command_burst_period_sec)
        for i in range(count):
            self.command_pub.publish(String(data=command))
            if i + 1 < count and period > 0.0:
                time.sleep(period)
        self.get_logger().info(f"motion command sent: {command}")

    def _relay_command(self, msg):
        command = str(msg.data).strip().lower()
        event = trajectory_event_for_command(command)
        if event is not None:
            self.event_pub.publish(String(data=event))
        if command == "stop":
            self.abort_pub.publish(Bool(data=True))

    def tick(self):
        if self.input_stream is None:
            return

        try:
            r, _, _ = select.select([self.input_stream], [], [], 0.0)
        except (OSError, ValueError):
            return
        if not r:
            return

        ch = self.input_stream.read(1)
        command = motion_command_for_key(ch)
        if command:
            self._publish_command(command)

    def destroy_node(self):
        if self.fd is not None and self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except termios.error:
                pass
        if self._opened_tty is not None:
            try:
                self._opened_tty.close()
            except OSError:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    n = MotionControlNode()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
