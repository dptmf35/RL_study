"""
Procedural terrain generator for MuJoCo heightfield.

Generates varied terrain (flat, slopes, stairs, rough) for locomotion training.
The heightfield data is injected into the MuJoCo model after loading.
"""

import os
import numpy as np

_MENAGERIE_GO2_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "mujoco_menagerie", "unitree_go2"
)

# Terrain heightfield parameters (must match scene_terrain.xml)
# NOTE: Low resolution (50x50) is critical for clean contact normals.
# With 200x200 (0.1m triangles), Go2 foot spheres (r=0.022m) generate
# multiple contacts per foot with bad normals, preventing locomotion.
# At 50x50 (0.4m triangles), each foot sits cleanly within one triangle.
TERRAIN_NROW = 50
TERRAIN_NCOL = 50
TERRAIN_X_HALF = 10.0   # half-extent in x → total 20m
TERRAIN_Y_HALF = 10.0   # half-extent in y → total 20m
TERRAIN_Z_MAX = 0.8      # max terrain height
TERRAIN_Z_MIN = 0.01     # collision margin at bottom

TERRAIN_SCENE_XML_PATH = os.path.join(_MENAGERIE_GO2_DIR, "scene_terrain.xml")

_SCENE_TERRAIN_XML = f"""\
<mujoco model="go2 terrain scene">
  <include file="go2.xml"/>

  <statistic center="0 0 0.5" extent="5"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="terrain_mat" rgba="0.5 0.55 0.45 0.8"/>
    <hfield name="terrain" nrow="{TERRAIN_NROW}" ncol="{TERRAIN_NCOL}"
      size="{TERRAIN_X_HALF} {TERRAIN_Y_HALF} {TERRAIN_Z_MAX} {TERRAIN_Z_MIN}"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <!-- Plane for reliable foot contacts (physics floor) -->
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>
    <!-- Heightfield for visual terrain display only (no collision) -->
    <geom name="terrain_visual" type="hfield" hfield="terrain" material="terrain_mat"
      contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def _smooth_2d(arr, size=3):
    """Simple box filter smoothing (no scipy dependency)."""
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    result = np.zeros_like(arr, dtype=np.float64)
    for di in range(size):
        for dj in range(size):
            result += padded[di : di + arr.shape[0], dj : dj + arr.shape[1]]
    return (result / (size * size)).astype(np.float32)


def ensure_terrain_scene_xml():
    """Write terrain scene XML to menagerie directory if not present."""
    if not os.path.exists(TERRAIN_SCENE_XML_PATH):
        with open(TERRAIN_SCENE_XML_PATH, "w") as f:
            f.write(_SCENE_TERRAIN_XML)
    return TERRAIN_SCENE_XML_PATH


class TerrainGenerator:
    """Procedural heightfield terrain generator.

    Generates a heightfield for MuJoCo with varied terrain patches.
    The robot spawns near x=0 on a flat section and walks into terrain (x+).

    Terrain layout:
        x < 1m  : flat (spawn area)
        x >= 1m : terrain patches (slope, stairs, rough)
    """

    def __init__(
        self,
        nrow=TERRAIN_NROW,
        ncol=TERRAIN_NCOL,
        x_half=TERRAIN_X_HALF,
        y_half=TERRAIN_Y_HALF,
        z_max=TERRAIN_Z_MAX,
    ):
        self.nrow = nrow
        self.ncol = ncol
        self.x_half = x_half
        self.y_half = y_half
        self.z_max = z_max

    def _x_to_col(self, x):
        return int((x + self.x_half) / (2 * self.x_half) * (self.ncol - 1))

    def _y_to_row(self, y):
        return int((self.y_half - y) / (2 * self.y_half) * (self.nrow - 1))

    def generate(self, difficulty=0.5, seed=None):
        """Generate heightfield data.

        Args:
            difficulty: 0.0 (flat) to 1.0 (hard terrain)
            seed: random seed for reproducibility

        Returns:
            np.ndarray of shape (nrow, ncol) with values in [0, 1]
        """
        rng = np.random.RandomState(seed)
        heights = np.zeros((self.nrow, self.ncol), dtype=np.float32)

        # Max feature height as fraction of z_max
        # difficulty=0.5 → max 10cm features, difficulty=1.0 → max 20cm
        max_feature = difficulty * 0.25

        # Flat area: x < 1m (robot spawns at x≈0, walks forward x+)
        flat_end_col = self._x_to_col(1.0)

        # Generate terrain patches in the rest
        patch_size = 5  # ~2m patches (at 50x50 res, 5 cells ≈ 2m)
        for r0 in range(0, self.nrow, patch_size):
            for c0 in range(flat_end_col, self.ncol, patch_size):
                r1 = min(r0 + patch_size, self.nrow)
                c1 = min(c0 + patch_size, self.ncol)
                rows, cols = r1 - r0, c1 - c0

                ttype = rng.choice(["flat", "slope", "stairs", "rough"])
                patch = self._make_patch(rows, cols, max_feature, ttype, rng)
                heights[r0:r1, c0:c1] = patch

        # Smooth transitions between patches
        heights = _smooth_2d(heights, size=3)

        # Ensure [0, 1] range
        heights = np.clip(heights, 0.0, 1.0)
        return heights

    def _make_patch(self, rows, cols, max_h, ttype, rng):
        if ttype == "flat":
            h = rng.uniform(0, max_h * 0.1)
            return np.full((rows, cols), h, dtype=np.float32)

        elif ttype == "slope":
            slope_h = max_h * rng.uniform(0.5, 1.0)
            slope = np.linspace(0, slope_h, cols, dtype=np.float32)
            if rng.random() > 0.5:
                slope = slope[::-1]
            return np.tile(slope, (rows, 1))

        elif ttype == "stairs":
            step_h = max_h * 0.3
            step_w = max(1, int(2 * (1 - max_h * 2)))
            stair = np.zeros((rows, cols), dtype=np.float32)
            for j in range(cols):
                stair[:, j] = step_h * (j // step_w)
            return np.clip(stair, 0, max_h)

        elif ttype == "rough":
            rough = rng.uniform(0, max_h, (rows, cols)).astype(np.float32)
            return _smooth_2d(rough, size=3)

        return np.zeros((rows, cols), dtype=np.float32)

    def sample_height(self, heights, x, y):
        """Get terrain height (meters) at world coordinates.

        Args:
            heights: heightfield data (nrow, ncol) in [0, 1]
            x, y: world coordinates

        Returns:
            Terrain height in meters
        """
        col_f = (x + self.x_half) / (2 * self.x_half) * (self.ncol - 1)
        row_f = (self.y_half - y) / (2 * self.y_half) * (self.nrow - 1)

        col = int(np.clip(col_f, 0, self.ncol - 1))
        row = int(np.clip(row_f, 0, self.nrow - 1))

        return float(heights[row, col]) * self.z_max

    def sample_heights_batch(self, heights, xs, ys):
        """Batch terrain height sampling."""
        cols = ((xs + self.x_half) / (2 * self.x_half) * (self.ncol - 1)).astype(int)
        rows = ((self.y_half - ys) / (2 * self.y_half) * (self.nrow - 1)).astype(int)

        cols = np.clip(cols, 0, self.ncol - 1)
        rows = np.clip(rows, 0, self.nrow - 1)

        return heights[rows, cols].astype(np.float64) * self.z_max
