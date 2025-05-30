# %%
import mitsuba as mi
import drjit as dr
import numpy as np

try:
    import luisarender
    gt_args = ["/home/lizengyu/miniconda3/envs/luisa/lib/python3.10/site-packages/luisarender/dylibs","-b","cuda", "/home/lizengyu/dpm/exp/data/luisa_scene/scenes/cbox_caustic.luisa"]
    luisarender.load_scene(gt_args)
except:
    print("import luisa fail")

mi.set_variant('cuda_ad_rgb')

# %%
scene_path = './scene.xml'
# scene_path = './living-room/test.xml'
scene = mi.load_file(scene_path)

integrator = mi.load_dict({
    'type': "dpm",
    'max_depth': 8
})

# %%
params = mi.traverse(scene)
# key = 'OBJMesh.vertex_positions'
# V = dr.unravel(mi.Point3f, params[key])
# dr.min(V.z), dr.max(V.z)

# %%
img = mi.render(scene, spp=128, integrator = integrator)
mi.util.write_bitmap("gt.exr", img)


