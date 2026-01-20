# # run_cube_viewer.py
# import time
# import mujoco
# from mujoco import viewer

# # Load cube.xml in an otherwise empty world
# m = mujoco.MjModel.from_xml_path("cube.xml")
# d = mujoco.MjData(m)

# duration = 10.0  # seconds
# framerate = 60
# frames = int(duration * framerate)

# with viewer.launch_passive(m, d) as v:
#     for i in range(frames):
#         mujoco.mj_step(m, d)
#         v.sync()  # render
#         time.sleep(1.0 / framerate)

import mujoco
spec = mujoco.MjSpec()
spec.modelname = "my model"
body = spec.worldbody.add_body(
    pos=[1, 2, 3],
    quat=[0, 1, 0, 0],
)
geom = body.add_geom(
    name='my_geom',
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[1, 0, 0],
    rgba=[1, 0, 0, 1],
)
...
model = spec.compile()