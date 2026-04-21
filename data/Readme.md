# Dataset: Outplaying Elite Table Tennis Players with an Autonomous Robot

## Description

This dataset contains the post-event ball states from matches between an autonomous robot and human players, including five elite players and two professional players.

## Dataset Format

| Field | Type | Description |
|-------|------|-------------|
| `player_id` | string | Unique identifier for each player |
| `rally_id` | string | Rally identifier in format `<game_id>/<rally_id>/<event_id>` |
| `type` | string | Event type `<event>_<player>`, where `<event>` is either `shot`, `bounce`, or `net` and `<player>` is either `p1` (robot) or `p2` (human). |
| `timestamp` | float | Timestamp of the event [s]|
| `ball_pos` | array | Ball position in 3D space [m] |
| `ball_vel_out` | array | Post-event ball velocity in 3D space [m/s] |
| `ball_spin_out` | array | Post-event ball angular velocity in 3D space [rad/s] |

We use right-handed conventions with the coordinate system's origin at the center of the table's playing surface, where the $x$-axis points toward the human player side of the table and the $z$-axis points upward.

