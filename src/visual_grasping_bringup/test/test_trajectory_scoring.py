# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from myrobot_common.planning.trajectory_scoring import (
    TrajectoryScoreConfig,
    rank_paths,
    score_trajectory,
    select_best_path,
)


def _traj(rows, names=None):
    points = [
        SimpleNamespace(
            positions=list(row),
            time_from_start=SimpleNamespace(sec=i, nanosec=0),
        )
        for i, row in enumerate(rows)
    ]
    joint_names = names or [f"j{i + 1}" for i in range(len(rows[0]))]
    return SimpleNamespace(points=points, joint_names=joint_names)


def test_empty_paths_return_none():
    assert select_best_path([]) is None


def test_single_point_score_is_finite():
    score = score_trajectory(_traj([[0.0, 0.0, 0.0]]))
    assert score.valid
    assert score.total_cost == 0.0
    assert score.num_points == 1


def test_shorter_path_is_selected():
    short = _traj([[0.0] * 6, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    long = _traj([[0.0] * 6, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert select_best_path([long, short], wrist_weight=0.0) is short


def test_wrist_weight_can_prefer_less_wrist_motion():
    less_wrist = _traj([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    more_wrist = _traj([[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 1.0, 0.0]])
    assert select_best_path(
        [more_wrist, less_wrist],
        wrist_weight=50.0,
        wrist_joint_indices=(2, 3),
    ) is less_wrist


def test_bad_dimensions_are_scored_invalid_without_crashing():
    bad = SimpleNamespace(
        points=[
            SimpleNamespace(positions=[0.0, 0.0]),
            SimpleNamespace(positions=[1.0]),
        ],
        joint_names=["j1", "j2"],
    )
    score = score_trajectory(bad)
    assert not score.valid
    assert score.total_cost >= 1.0e9


def test_wrist_index_out_of_range_is_invalid():
    score = score_trajectory(
        _traj([[0.0, 0.0], [1.0, 0.0]]),
        TrajectoryScoreConfig(wrist_joint_indices=(3,)),
    )
    assert not score.valid


def test_rank_paths_sorts_by_cost():
    short = _traj([[0.0] * 6, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    long = _traj([[0.0] * 6, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ranked = rank_paths(
        [long, short],
        TrajectoryScoreConfig(wrist_length_weight=0.0),
    )
    assert ranked[0][0] is short
