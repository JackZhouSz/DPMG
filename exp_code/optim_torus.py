
from shutil import copyfile
copyfile("/home/lizengyu/dpm/integrators/dpm.py","/home/lizengyu/miniconda3/envs/mi/lib/python3.9/site-packages/mitsuba/python/ad/integrators/dpm.py")
copyfile("/home/lizengyu/dpm/integrators/dpm.py","/home/lizengyu/EPSM_Mitsuba3/build_py310/python/mitsuba/python/ad/integrators/dpm.py")

import torch
import drjit as dr
import mitsuba as mi

import cv2,math
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

import sys,os,json
import importlib
sys.path.append("./")
sys.path.append("../")
from utils.logger import Logger
from utils.matcher import Matcher
from mitsuba.scalar_rgb import Transform4f as T
from tqdm.std import tqdm
import torchvision
rez = torchvision.transforms.Resize((512,512))
try:
    import luisarender
    luisa_args = ["/home/lizengyu/miniconda3/envs/luisa/lib/python3.10/site-packages/luisarender/dylibs","-b","cuda", "/home/lizengyu/dpm/exp/data/luisa_scene/scenes/cbox_caustic.luisa"]
    luisarender.load_scene(luisa_args)
except:
    pass

import argparse

mi.set_variant('cuda_ad_rgb')

Pooler = torch.nn.AvgPool2d(kernel_size=2)
@dr.wrap_ad(source='drjit', target='torch')
def down_res_loss(st, img, img_ref):
    img = img[None,...].permute(0,3,1,2)
    img_ref = img_ref[None,...].permute(0,3,1,2)
    while st>0:
        img = Pooler(img)
        img_ref = Pooler(img_ref)
        st = st-1
    if log_level>0:
        Logger.save_img("down_res.png",img.permute(0,2,3,1)[0])
    return torch.mean((img-img_ref)**2)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", "--config", type=str)
    
    args = parser.parse_known_args()
    
    config_file = args[0].config
    config = config_file[:-3].replace("/",".")
    tasks = importlib.import_module(f'{config}') # import specific task
    
    parser.add_argument("-m", "--method",type=str, default=tasks.method)
    parser.add_argument("-o", "--optim_type",type=int, default=0)
    parser.add_argument("-l", "--log_level",type=int, default=1)
    parser.add_argument("-t", "--time",type=bool, default=False)
    parser.add_argument("-s", "--spp",type=int, default=tasks.spp)
    parser.add_argument("-g", "--gt_spp",type=int, default=tasks.gt_spp)
    parser.add_argument("-d", "--max_depth",type=int, default=tasks.max_depth)
    parser.add_argument("-e", "--exp_name",type=str, default=tasks.exp_name)
    
    args = parser.parse_args()
    method = args.method
    optim_type = args.optim_type
    log_level = args.log_level
    exp_name = args.exp_name
    spp = args.spp # spp
    gt_spp = args.gt_spp
    max_depth = args.max_depth
    
    # must in config file
    loss_type = tasks.loss_type
    resolution = tasks.resolution #resolution
    thres = tasks.thres # for hybrid scheme
    match_res = tasks.match_res
    scene = tasks.scene # scene
    optim_view = tasks.optim_view
    optim_photon = tasks.optim_photon
    photon_per_iter = tasks.photon_per_iter
    photon_backward_iter = tasks.photon_backward_iter
    backward_query_radius = tasks.backward_query_radius
    forward_query_radius = tasks.forward_query_radius
    init_query_radius = tasks.init_query_radius
    
    Logger.init(exp_name="", show=False, debug=False, path=f"results/{exp_name}_{method}/",add_time=args.time)
    Logger.save_file(config_file)
    Logger.save_file("./optim.py")
    Logger.save_file("/home/lizengyu/dpm/integrators/dpm.py")
    Logger.save_config(args)
    
    integrator = mi.load_dict({
        'type': method,
        'max_depth': max_depth
    })
    
    gt_path = f"results/{exp_name}_{method}/gt_{method}_{gt_spp}.pt"
    if os.path.exists(gt_path) and optim_type==0:
        gt_img_torch = torch.load(gt_path)
        gt_img_hdr = mi.TensorXf(gt_img_torch)[...,:3]
    else:
        if hasattr(tasks,"gt_scene")==True:
            gt_img_hdr = mi.render(tasks.gt_scene, seed=0, spp=gt_spp, sensor=0, integrator=integrator)
        else:
            gt_img_hdr = mi.render(scene, seed=0, spp=gt_spp, sensor=0, integrator=integrator)
        
        gt_img_hdr = gt_img_hdr[...,:3]
        torch.save(gt_img_hdr.torch(), gt_path)
        mi.util.write_bitmap(gt_path[:-2]+"png", gt_img_hdr[...,:3])
    
    img_np = np.array(mi.util.convert_to_bitmap(gt_img_hdr))
    gt_img_ldr = torch.from_numpy(img_np).to(device)/255.0
    # # get target image
    # if hasattr(tasks,"gt_img")==True:
    #     gt_img_ldr = torch.from_numpy(cv2.cvtColor(cv2.imread(tasks.gt_img),cv2.COLOR_BGR2RGB)).to(device)/255.0
    #     gt_img_hdr = mi.TensorXf(gt_img_ldr.reshape(-1,3)) # fake hdr
    # else:
    #     if hasattr(tasks,"gt_scene")==True:
    #         gt_img_hdr = mi.render(tasks.gt_scene, seed=0, spp=tasks.gt_spp, sensor=0, integrator=integrator)
    #     else:
    #         gt_img_hdr = mi.render(scene, seed=0, spp=tasks.gt_spp, sensor=0, integrator=integrator)

    #     gt_img_hdr = gt_img_hdr[...,:3]
    #     mi.util.write_bitmap(f"{exp_name}/gt_{method}_{tasks.gt_spp}.png", gt_img_hdr)
    #     img_np = np.array(mi.util.convert_to_bitmap(gt_img_hdr))
    #     gt_img_ldr = torch.from_numpy(img_np).to(device)/255.0

    gt_img_ldr_low= torch.from_numpy(cv2.resize(np.array(mi.util.convert_to_bitmap(gt_img_hdr)),(match_res,match_res),interpolation=cv2.INTER_CUBIC)).to(device)/255.0
    
    if log_level>0:
        Logger.save_img("gt_img.png",gt_img_ldr)
    # pixel matcher using optimal transport(Sinkhorn)
    
    matcher = Matcher(match_res, device)

    # get optimized parameter and transformation
    opt, opt2, apply_transformation, output, params = tasks.optim_settings()
    apply_transformation(params, opt)
    for key in opt.keys():
        dr.enable_grad(opt[key])
    params = mi.traverse(scene)

    init_path = f"results/{exp_name}_{method}/init_{method}_{gt_spp}.pt"
    if os.path.exists(init_path) and optim_type==0:
        init_img_torch = torch.load(init_path)
        init_img_hdr = mi.TensorXf(init_img_torch)[...,:3]
    else:
        init_img_hdr = mi.render(scene, seed=0, spp=gt_spp, sensor=0, integrator=integrator)[...,:3]
        torch.save(init_img_hdr.torch(), init_path)
        mi.util.write_bitmap(init_path[:-2]+"png", init_img_hdr[...,:3])
    
    img_np = np.array(mi.util.convert_to_bitmap(init_img_hdr))
    init_img_ldr = torch.from_numpy(img_np).to(device)/255.0
    
    # # get init image
    # init_img_hdr = mi.render(scene, params, seed=0, integrator=integrator, spp=tasks.gt_spp, sensor=0)
    # init_img_hdr = init_img_hdr[...,:3]
    # mi.util.write_bitmap(f"{exp_name}/init_{method}_{tasks.gt_spp}.png", init_img_hdr)
    # init_img_ldr = torch.from_numpy(np.array( mi.util.convert_to_bitmap(init_img_hdr[...,:3]))).to(device)/255.0
    
    if log_level>0:
        Logger.save_img("init_img.png",init_img_ldr)
    
    if optim_type==1:
        Logger.exit()
        exit()
    
    loop = tqdm(range(tasks.it))    
    for it in loop:
        apply_transformation(params, opt)
        img_hdr = mi.render(scene, params, seed=0, spp=spp, integrator=integrator, sensor=0)
        img_ldr = np.array(mi.util.convert_to_bitmap(img_hdr[...,:3]))
        #mi.util.write_bitmap(f"{exp_name}/optim_{method}_{loss_type}_{spp}.png", img_hdr[...,:3])
        if log_level>0:
            Logger.save_img(f"optim_{method}.png",img_ldr/255.0,flip=False)
            Logger.add_image(f"optim_{method}",img_ldr/255.0,flip=False)
        if log_level>1:
            Logger.save_img_2(f"optim_{method}_{it}.png",img_ldr/255.0,flip=False)

        if loss_type=="RGBXY" and method=="dpm":
            img_ldr_low = torch.from_numpy(cv2.resize(img_ldr,(match_res,match_res),interpolation=cv2.INTER_CUBIC)).to(device)/255.0
            
            grad_ = matcher.match_Sinkhorn(img_ldr_low[...,:3].reshape(-1,3), gt_img_ldr_low[...,:3].reshape(-1,3))#, weight=1.0/(math.exp((it+1)/50)-1))
            grad_ = grad_.clone().reshape(match_res,match_res,5)
            
            cv2.imwrite("test.png",abs((grad_[...,3].cpu().numpy()*255).astype(np.uint8)))
            
            if isinstance(resolution, tuple):
                grad_ = grad_.repeat(resolution[0]//match_res,resolution[1]//match_res,1)
            else:
                grad_ = grad_.repeat(resolution//match_res,resolution//match_res,1)
                
            grad_rgb = 2*(img_hdr[...,:3].torch() - gt_img_hdr[...,:3].torch())
            grad_ori_ = rez(grad_[:match_res,:match_res,:3].permute(2,0,1)).permute(1,2,0)
            if it<100:
                grad_[...,:3] = grad_ori_
            elif it>100 and it <200:
                grad_[...,:3]=grad_[...,:3]*(200-it)/100+grad_rgb*((it-100)/100)
            else:
                grad_[...,:3] = grad_rgb
            grad = mi.TensorXf(grad_)
            #grad_[...,:3]=0
            dr.backward(img_hdr*grad)
        else:
            loss = down_res_loss(6-((7*it)//tasks.it),img_hdr,gt_img_hdr[...,:3])
            #loss = dr.mean(dr.sqr(img_hdr - gt_img_hdr))
            dr.backward(loss)
        #try:
        for key in opt.keys():
            x = dr.grad(opt[key])
            print(key,opt[key],x)
            x[dr.isnan(x)] = 0
            dr.set_grad(opt[key],x)
            if log_level>1:
                Logger.save_param(f"param_{key}_{it}.npy",opt[key].torch().cpu().numpy())
            
        for key in opt2.keys():
            x = dr.grad(opt2[key])
            print(key,opt2[key],x)
            x[dr.isnan(x)] = 0
            dr.set_grad(opt2[key],x)
            if log_level>1:
                Logger.save_param(f"param_{key}_{it}.npy",opt2[key].torch().cpu().numpy())
        
        opt.step()
        opt2.step()
        loop.set_description(f"Iteration {it:02d}: error={output(opt)}")
    
    img_final_hdr = mi.render(scene, params, seed=0, spp=1024, sensor=0)
    img_final_ldr = torch.from_numpy(np.array(mi.util.convert_to_bitmap(img_final_hdr[...,:3]))).to(device)/255.0
    mi.util.write_bitmap(f"final_{exp_name}_{method}.png", img_final_hdr[...,:3])
    if log_level>0:
        Logger.save_img(f"final_{exp_name}_{method}.png",img_final_ldr)
    
    Logger.exit()
    print("finish optim")
