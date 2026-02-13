"""
Box-based terrain generator for MuJoCo.

Generates terrain using box primitives (not heightfields) for reliable
physics collision with Go2's small foot sphere colliders (r=0.022m).

Heightfields have bad contact normals with small spheres, but box-sphere
collision is well-supported and gives accurate normals.

Terrain: stepped-pyramid mounds scattered across the field.
Curriculum: runtime z-scaling of terrain geom positions/sizes.
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

_MENAGERIE_GO2_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "mujoco_menagerie", "unitree_go2"
)
TERRAIN_SCENE_XML_PATH = os.path.join(_MENAGERIE_GO2_DIR, "scene_terrain.xml")

# Field dimensions (meters)
FIELD_X_RANGE = (-5.0, 5.0)   # 10m wide
FIELD_Y_RANGE = (-5.0, 5.0)   # 10m deep

# Heightmap resolution for queries
HEIGHTMAP_RESOLUTION = 0.1  # 10cm per cell


class MoundSpec:
    """Specification for a single terrain mound."""

    __slots__ = ("cx", "cy", "radius", "height", "n_layers")

    def __init__(self, cx, cy, radius, height, n_layers=8):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.height = height
        self.n_layers = n_layers


class TerrainGenerator:
    """Generates box-based terrain for Go2 locomotion training.

    Creates stepped-pyramid mounds using box geoms. Each mound is a stack
    of concentric boxes (large at bottom, small at top), forming a stepped
    hill that the robot can walk over.

    Supports:
    - Difficulty-based generation (controls mound count and height)
    - Runtime difficulty scaling for curriculum learning
    - Fast terrain height queries via precomputed heightmap
    """

    def __init__(
        self,
        x_range=FIELD_X_RANGE,
        y_range=FIELD_Y_RANGE,
        spawn_clear_radius=1.5,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.spawn_clear_radius = spawn_clear_radius
        self.mounds: list[MoundSpec] = []

        # Heightmap grid
        self._hm_x0 = x_range[0]
        self._hm_y0 = y_range[0]
        self._hm_res = HEIGHTMAP_RESOLUTION
        self._hm_nx = int((x_range[1] - x_range[0]) / HEIGHTMAP_RESOLUTION) + 1
        self._hm_ny = int((y_range[1] - y_range[0]) / HEIGHTMAP_RESOLUTION) + 1
        self._heightmap: np.ndarray | None = None

    def generate(self, difficulty=0.5, seed=None):
        """Generate terrain mound layout.

        Args:
            difficulty: 0.0 (nearly flat) to 1.0 (tall hills)
            seed: random seed for reproducibility

        Returns:
            List of MoundSpec objects
        """
        rng = np.random.RandomState(seed)
        self.mounds = []

        n_mounds = 6 + int(difficulty * 14)  # 6-20 mounds
        max_height = 0.05 + difficulty * 0.20  # 5cm - 25cm

        attempts = 0
        while len(self.mounds) < n_mounds and attempts < n_mounds * 3:
            attempts += 1
            cx = rng.uniform(self.x_range[0] + 1.5, self.x_range[1] - 1.5)
            cy = rng.uniform(self.y_range[0] + 1.5, self.y_range[1] - 1.5)

            # Keep spawn area clear
            if abs(cx) < self.spawn_clear_radius and abs(cy) < self.spawn_clear_radius:
                continue

            radius = rng.uniform(0.6, 2.0)
            height = rng.uniform(max(0.04, max_height * 0.3), max_height)
            n_layers = max(4, int(height / 0.02))  # ~2cm per step

            self.mounds.append(MoundSpec(cx, cy, radius, height, n_layers))

        self._build_heightmap()
        return self.mounds

    def _build_heightmap(self):
        """Build 2D heightmap grid for fast height queries (vectorized)."""
        self._heightmap = np.zeros((self._hm_ny, self._hm_nx), dtype=np.float32)

        xs = self._hm_x0 + np.arange(self._hm_nx) * self._hm_res
        ys = self._hm_y0 + np.arange(self._hm_ny) * self._hm_res
        xg, yg = np.meshgrid(xs, ys)  # (ny, nx)

        for mound in self.mounds:
            dx = np.abs(xg - mound.cx)
            dy = np.abs(yg - mound.cy)
            dist = np.maximum(dx, dy)  # Chebyshev distance (square mounds)

            mask = dist < mound.radius
            norm_dist = np.where(mask, dist / mound.radius, 1.0)
            layer = np.floor((1.0 - norm_dist) * mound.n_layers).astype(np.int32)
            layer = np.clip(layer, 0, mound.n_layers - 1)
            h = ((layer + 1) * mound.height / mound.n_layers).astype(np.float32)
            h = np.where(mask, h, 0.0)

            self._heightmap = np.maximum(self._heightmap, h)

    def sample_height(self, x, y, difficulty_scale=1.0):
        """Get terrain height at world coordinates.

        Args:
            x, y: world position
            difficulty_scale: multiplier for curriculum (0=flat, 1=full)

        Returns:
            Terrain height in meters
        """
        if self._heightmap is None:
            return 0.0

        ix = int((x - self._hm_x0) / self._hm_res)
        iy = int((y - self._hm_y0) / self._hm_res)
        ix = max(0, min(ix, self._hm_nx - 1))
        iy = max(0, min(iy, self._hm_ny - 1))

        return float(self._heightmap[iy, ix]) * difficulty_scale

    def generate_scene_xml(self, output_path=None):
        """Generate MuJoCo scene XML with terrain box geoms.

        Returns:
            Path to generated XML file
        """
        output_path = output_path or TERRAIN_SCENE_XML_PATH

        root = ET.Element("mujoco", model="go2 terrain scene")
        ET.SubElement(root, "include", file="go2.xml")

        stat = ET.SubElement(root, "statistic")
        stat.set("center", "0 0 0.5")
        stat.set("extent", "5")

        visual = ET.SubElement(root, "visual")
        hl = ET.SubElement(visual, "headlight")
        hl.set("diffuse", "0.6 0.6 0.6")
        hl.set("ambient", "0.3 0.3 0.3")
        hl.set("specular", "0 0 0")
        rgba = ET.SubElement(visual, "rgba")
        rgba.set("haze", "0.15 0.25 0.35 1")
        gl = ET.SubElement(visual, "global")
        gl.set("azimuth", "-130")
        gl.set("elevation", "-20")

        asset = ET.SubElement(root, "asset")
        # Skybox
        sky = ET.SubElement(asset, "texture")
        sky.set("type", "skybox")
        sky.set("builtin", "gradient")
        sky.set("rgb1", "0.3 0.5 0.7")
        sky.set("rgb2", "0 0 0")
        sky.set("width", "512")
        sky.set("height", "3072")
        # Ground texture/material
        gt = ET.SubElement(asset, "texture")
        gt.set("type", "2d")
        gt.set("name", "groundplane")
        gt.set("builtin", "checker")
        gt.set("mark", "edge")
        gt.set("rgb1", "0.2 0.3 0.4")
        gt.set("rgb2", "0.1 0.2 0.3")
        gt.set("markrgb", "0.8 0.8 0.8")
        gt.set("width", "300")
        gt.set("height", "300")
        gm = ET.SubElement(asset, "material")
        gm.set("name", "groundplane")
        gm.set("texture", "groundplane")
        gm.set("texuniform", "true")
        gm.set("texrepeat", "5 5")
        gm.set("reflectance", "0.2")
        # Terrain texture/material
        tt = ET.SubElement(asset, "texture")
        tt.set("type", "2d")
        tt.set("name", "terrain_tex")
        tt.set("builtin", "gradient")
        tt.set("mark", "random")
        tt.set("rgb1", "0.45 0.55 0.35")
        tt.set("rgb2", "0.35 0.42 0.30")
        tt.set("markrgb", "0.50 0.60 0.40")
        tt.set("width", "256")
        tt.set("height", "256")
        tm = ET.SubElement(asset, "material")
        tm.set("name", "terrain_mat")
        tm.set("texture", "terrain_tex")
        tm.set("texuniform", "true")
        tm.set("texrepeat", "3 3")
        tm.set("specular", "0.1")

        worldbody = ET.SubElement(root, "worldbody")
        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")
        light.set("dir", "0 0 -1")
        light.set("directional", "true")
        floor = ET.SubElement(worldbody, "geom")
        floor.set("name", "floor")
        floor.set("type", "plane")
        floor.set("size", "0 0 0.05")
        floor.set("material", "groundplane")

        # Add terrain box geoms (stepped-pyramid mounds)
        geom_idx = 0
        for mound in self.mounds:
            layer_h = mound.height / mound.n_layers
            for layer in range(mound.n_layers):
                layer_radius = mound.radius * (1.0 - layer / mound.n_layers)
                layer_z = (layer + 0.5) * layer_h

                g = ET.SubElement(worldbody, "geom")
                g.set("name", f"terrain_{geom_idx}")
                g.set("type", "box")
                g.set("pos", f"{mound.cx:.4f} {mound.cy:.4f} {layer_z:.4f}")
                g.set("size", f"{layer_radius:.4f} {layer_radius:.4f} {layer_h / 2:.4f}")
                g.set("material", "terrain_mat")
                g.set("friction", "0.8 0.005 0.001")
                geom_idx += 1

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, xml_declaration=False, encoding="unicode")

        return output_path

    def get_terrain_geom_count(self):
        """Total number of terrain box geoms."""
        return sum(m.n_layers for m in self.mounds)

    def get_terrain_geom_ref_data(self):
        """Get reference z-positions and z-half-sizes for curriculum scaling.

        Returns:
            (ref_pos_z, ref_size_z): arrays of shape (n_terrain_geoms,)
        """
        ref_pos_z = []
        ref_size_z = []
        for mound in self.mounds:
            layer_h = mound.height / mound.n_layers
            for layer in range(mound.n_layers):
                ref_pos_z.append((layer + 0.5) * layer_h)
                ref_size_z.append(layer_h / 2.0)
        return np.array(ref_pos_z, dtype=np.float64), np.array(ref_size_z, dtype=np.float64)
