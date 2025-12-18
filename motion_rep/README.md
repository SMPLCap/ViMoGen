# 276-Dim Global Motion Representation (DART Style)

This project now uses a single, global, orientation-aligned motion representation with 276 features per frame. The layout follows `collect_motion_rep_DART` in `retarget_motion.py` and is derived from the [DART paper](https://arxiv.org/abs/2410.05260).

## Construction
Given SMPL params (`global_orient`, `body_pose`, `transl`) and global joints (`joints`), canonicalized to a common facing direction, we build velocities and convert rotations to 6D:

1. Body pose (21 non-root joints) in Rot6D.
2. Global joints positions.
3. Joint velocities (forward differences).
4. Root/global orientation in Rot6D.
5. Root orientation velocity (relative rotation between frames) in Rot6D.
6. Root translation.
7. Root translation velocity (forward differences).

Because velocities use frame `t+1`, the motion sequence has one fewer frame than the original SMPL sequence (`motion.shape[0] = smpl_seq_len - 1`).

## Layout (JOINT_NUM = 22)

| Index range | Length | Description |
| ----------- | ------ | ----------- |
| 0:126 | 126 | Body pose Rot6D for 21 non-root joints |
| 126:192 | 66 | Global joints XYZ (22 * 3) |
| 192:258 | 66 | Joint velocities XYZ |
| 258:264 | 6 | Root/global orientation (Rot6D) |
| 264:270 | 6 | Root orientation velocity (Rot6D) |
| 270:273 | 3 | Root translation |
| 273:276 | 3 | Root translation velocity |

## Notes
- All stored motions are **global/canonicalized** (pelvis-centered, facing alignment handled upstream in `canonicalize_motion` + `collect_motion_rep_DART`).
