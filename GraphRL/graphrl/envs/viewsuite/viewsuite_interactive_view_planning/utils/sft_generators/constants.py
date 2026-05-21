"""Constants for SFT dataset generators."""

_LABELS = "ABCD"
_STEP_TRANSLATION = 0.5   # metres per translation step
_STEP_ROTATION = 30.0     # degrees per rotation step
_ACTION_LIST_STR = (
    "[move_forward, move_backward, move_right, move_left, "
    "move_up, move_down, turn_left, turn_right, look_up, look_down, "
    "rotate_ccw, rotate_cw]"
)

# ── action vocabulary & displacement helpers (for MCQ negatives) ─────────

_ACTION_VOCAB = [
    "move_forward", "move_backward", "move_left", "move_right",
    "move_up", "move_down", "turn_left", "turn_right",
    "look_up", "look_down", "rotate_ccw", "rotate_cw",
]

# Approximate net displacement per action (6-D: fwd, right, up, yaw, pitch, roll).
# Ignores rotation-translation coupling but catches most equivalences.
_ACTION_DISP = {
    "move_forward":  ( 1,  0,  0,  0,  0,  0),
    "move_backward": (-1,  0,  0,  0,  0,  0),
    "move_right":    ( 0,  1,  0,  0,  0,  0),
    "move_left":     ( 0, -1,  0,  0,  0,  0),
    "move_up":       ( 0,  0,  1,  0,  0,  0),
    "move_down":     ( 0,  0, -1,  0,  0,  0),
    "turn_right":    ( 0,  0,  0,  1,  0,  0),
    "turn_left":     ( 0,  0,  0, -1,  0,  0),
    "look_up":       ( 0,  0,  0,  0,  1,  0),
    "look_down":     ( 0,  0,  0,  0, -1,  0),
    "rotate_cw":     ( 0,  0,  0,  0,  0,  1),
    "rotate_ccw":    ( 0,  0,  0,  0,  0, -1),
}
