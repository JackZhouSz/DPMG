import mitsuba as mi
import drjit as dr

exp_name="torus"
method="dpm"

it=1000
gt_spp=64
spp=16
resolution=512
thres = 375
max_depth = 5
match_res = 128

optim_view = True
optim_photon = False
loss_type = "RGB"
photon_per_iter = 512*512*16
photon_backward_iter = spp
backward_query_radius = 0.01
forward_query_radius = 0.01
init_query_radius = 0.01

scene_file = 'torus/torus.xml'
scene = mi.load_file(scene_file)
params = mi.traverse(scene)
# key = 'donut.vertex_positions'
# init_p = dr.unravel(mi.Point3f, params[key])
# params[key] = dr.ravel(init_p + mi.Vector3f(4.0, -4.0, 0.0))
# params.update()

# params = mi.traverse(scene)

def optim_settings():
    objects = ['butterfly', 'sunflower', 'leaf']
    init_trans = [
        (7.0, 0, 0.0),
        (-7.0, 0, 0),
        (-3.0, 0, 1.5)
    ]

    delta_colors = {
        'butterfly' : [2,0,2],
        'leaf': [0.5, 1, 0.5]
    }
    init_colors = {}
    opt = mi.ad.Adam(lr=0.2)
    opt2 = mi.ad.Adam(lr=0.01)
    init_positions = {}
    shapes = {}
    keys = {}
    keys['butterfly'] = 'ButterflyBSDFBlue.reflectance.data'
    keys['leaf'] = 'LeafBSDF.brdf_0.reflectance.data'
    for obj, trans in zip(objects, init_trans):
        init_positions[obj] = dr.unravel(mi.Point3f, params[f"{obj}.vertex_positions"])
        opt[f"trans_{obj}_x"] = mi.Float(trans[0])
        opt[f"trans_{obj}_y"] = mi.Float(trans[1])
        opt[f"trans_{obj}_z"] = mi.Float(trans[2])
        if not obj == 'sunflower':
            opt2[f'color_trans_{obj}'] = mi.Color3f(delta_colors[obj])
            init_colors[obj] = dr.unravel(mi.Color3f, params[keys[obj]])
            shapes[obj] = params[keys[obj]].shape

    def apply_transformation(params, opt):
        for obj in objects:
            opt[f"trans_{obj}_x"] = dr.clamp(opt[f'trans_{obj}_x'],-10,10)
            opt[f"trans_{obj}_y"] = dr.clamp(opt[f'trans_{obj}_y'],-0,0)
            opt[f"trans_{obj}_z"] = dr.clamp(opt[f'trans_{obj}_z'],-10,10)
            trafo = mi.Transform4f.translate([opt[f'trans_{obj}_x'], opt[f'trans_{obj}_y'], opt[f'trans_{obj}_z']])
            params[f'{obj}.vertex_positions'] = dr.ravel(trafo @ init_positions[obj])
            if not obj == 'sunflower':
                opt2[f'color_trans_{obj}'] = dr.clamp(opt2[f'color_trans_{obj}'],-10,10)
                params[keys[obj]] = mi.TensorXf(dr.clamp(dr.ravel(init_colors[obj] + opt2[f'color_trans_{obj}']/10.0), 0.0, 1.0), shape=shapes[obj])
        params.update()
    
    def output(opt):
        l2 = 0
        l2_c = 0
        for obj in objects:
            x = float(opt[f"trans_{obj}_x"][0])
            y = float(opt[f"trans_{obj}_y"][0])
            z = float(opt[f"trans_{obj}_z"][0])
            l2 += x * x + y * y + z * z
            if not obj == 'sunflower':
                c = dr.detach(opt2[f'color_trans_{obj}'])
                l2_c += float(dr.sum(c * c)[0])
        return f"vertex_L2={l2}, color_L2={l2_c}"

    return opt, opt2, apply_transformation, output, params
