# Hand-eye calibration snapshots

See the [Fairino hand-eye calibration guide](../../../docs/手眼标定/手眼标定文档说明.md)
before creating, selecting, or publishing a snapshot.

`sim/` stores Gazebo calibration snapshots and `real/` stores real-robot snapshots.

Each successful automatic calibration writes a matched pair named
`<calibration_name>_YYYYMMDD_HHMMSS.{calib,samples}`. Launch files load the
latest timestamped `.calib` by default; pass the full timestamped name to load
an earlier snapshot.

Do not treat a generated `.calib` file as valid until its paired `.samples`
diagnostics and an independent pose check have passed.
