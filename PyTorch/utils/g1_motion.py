"""
Minimal motion container for AIST++ G1 retargeted pickles (root + dof representation).

Each raw clip in `data/aistpp_g1/motions_g1/*_rtg.pkl` is a dict with keys:
- fps: float
- root_pos: (T, 3) float
- root_rot: (T, 4) float, quaternion in **xyzw** order
- dof_pos: (T, D) float (D=29 in provided data)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class G1Motion:
    """
    Root + DOF motion sequence.

    - dof_pos: (T, D=36)
    - fps: frames per second
    """

    dof_pos: np.ndarray
    fps: float = 30.0
    filepath: Optional[str] = None


    @property
    def frame_num(self) -> int:
        return int(self.dof_pos.shape[0])

    @property
    def dof_dim(self) -> int:
        return int(self.dof_pos.shape[-1])

    def copy(self) -> "G1Motion":
        return copy.deepcopy(self)

    def __getitem__(self, item) -> "G1Motion":
        # supports slicing by frames
        return G1Motion(
            dof_pos=self.dof_pos[item].copy(),
            fps=self.fps,
            filepath=self.filepath,
        )


