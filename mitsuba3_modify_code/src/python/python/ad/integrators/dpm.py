from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import gc
import torch
from .common import ADIntegrator

import luisarender
import importlib
import sys
sys.path.append("./")
sys.path.append("../")

if len(sys.argv) > 2:
    config_file = sys.argv[2]
    config = config_file[:-3].replace("/",".")
    tasks = importlib.import_module(f'{config}') # import specific task

    optim_view = tasks.optim_view
    optim_photon = tasks.optim_photon
    loss_type = tasks.loss_type
    photon_per_iter = tasks.photon_per_iter
    photon_backward_iter = tasks.photon_backward_iter

    backward_query_radius = tasks.backward_query_radius
    forward_query_radius = tasks.forward_query_radius
    init_query_radius = tasks.init_query_radius
else:
    optim_view = True
    optim_photon = True
    loss_type = "RGB"
    photon_per_iter = 512*512*16
    photon_backward_iter = 16

    backward_query_radius = 0.01
    forward_query_radius = 0.01
    init_query_radius = 0.01

def mis_weight(pdf_a, pdf_b):
    a2 = dr.sqr(pdf_a)
    b2 = dr.sqr(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))

class DPMIntegrator(ADIntegrator):
        
    def prepare_view_ray(self, scene, sensor, seed, spp):
        with dr.suspend_grad(): 
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )
            try:
                ray, weight, pos = self.sample_rays(scene, sensor, sampler)
            except:
                ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
            
        return sampler, spp, ray, weight, pos 
      
    def prepare_photon(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):

        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        wavefront_size = photon_per_iter
        
        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)
    
        sampler.seed(seed, wavefront_size)
        return sampler, spp

    def sample_photon_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample  = sampler.next_1d()
        direction_sample = sampler.next_2d()
        position_sample  = sampler.next_2d()

        ray, ray_weight, emitter = scene.sample_emitter_ray(time, wavelength_sample, direction_sample, position_sample, mi.Bool(True))
    
        return ray, ray_weight, emitter
    
    def render_direct(self, mode, scene, sensor, seed, spp):
        sampler, spp, ray, weight, pos = self.prepare_view_ray(scene, sensor, seed, spp)
        if mode is dr.ADMode.Primal:
            L, valid, _ = self.sample_view_loop(
                mode=dr.ADMode.Primal,#useless
                scene=scene,
                sampler=sampler,
                ray=ray,
                δL=None,
                state_in=None,
                active=mi.Bool(True),
                weight=weight
            )
        else:
            L, valid, _ = self.sample_view(
                mode=dr.ADMode.Primal,#useless
                scene=scene,
                sampler=sampler,
                ray=ray,
                δL=None,
                state_in=None,
                active=mi.Bool(True),
                weight=weight
            )
        
        film = sensor.film()
        block = film.create_block()
        block.set_coalesce(block.coalesce() and spp >= 4)
        block.put(
            pos=pos,
            wavelengths=ray.wavelengths,
            value=L,
            weight=mi.Float(1),
            alpha=mi.Float(1)
        )

        film.put_block(block)
        image = film.develop()
        del sampler, spp, ray, weight, pos, L,  valid 
        gc.collect() # Clean up
        return image
    
    def render_indirect(self, mode, scene, sensor, seed, spp, grad_in=None):
        film = sensor.film()
        pixel_nums = film.crop_size()[0]*film.crop_size()[1]
        viewpoint_pos_torch = self.viewpoint_pos.torch().reshape((pixel_nums,spp,3)).permute(1,0,2).contiguous()
        viewpoint_wo_torch = self.viewpoint_wo.torch().reshape((pixel_nums,spp,3)).permute(1,0,2).contiguous()
        viewpoint_beta_torch = self.viewpoint_beta.torch().reshape((pixel_nums,spp,3)).permute(1,0,2).contiguous()
        torch.cuda.synchronize()
        iterations = spp
        if mode is dr.ADMode.Primal:
            luisarender.init_viewpointmap(pixel_nums, init_query_radius, 128*128*8)
        else:
            if mode is dr.ADMode.Forward:
                luisarender.init_viewpointmap(pixel_nums, forward_query_radius, 128*128*8)
            else:
                luisarender.init_viewpointmap(pixel_nums, backward_query_radius, 128*128*8)
                iterations = photon_backward_iter
                assert(iterations<=spp)
                iterations = min(iterations, spp)
            if grad_in is not None:
                grad_in_color = grad_in.torch()[...,:3].clone()
                grad_torch = grad_in_color.reshape(-1,3).contiguous()/spp
                torch.cuda.synchronize()
                luisarender.update_grad(grad_torch.data_ptr(), grad_torch.shape[0],False)
                if loss_type=="RGBXY":
                    luisarender.add_point_grad(self.diffuse_pos_torch.data_ptr(), self.diffuse_grad_torch.data_ptr(), self.diffuse_pos_torch.data_ptr(), self.diffuse_pos_torch.shape[0], False)
        
        for iter in range(iterations):
            #iter)
            luisarender.add_viewpoint(viewpoint_pos_torch[iter].data_ptr(), viewpoint_wo_torch[iter].data_ptr(), viewpoint_beta_torch[iter].data_ptr(), viewpoint_pos_torch.shape[1], False)
            #with dr.suspend_grad():
            sampler_photon, spp_photon = self.prepare_photon(
                seed=seed+iter,
                sensor=sensor, 
                spp=photon_per_iter,
                aovs=self.aov_names()
            )

            ray_photon, weight_photon, emitter_photon = self.sample_photon_rays(scene, sensor, sampler_photon)
            self.photon_pass(
                mode=mode,
                scene=scene,
                sampler=sampler_photon,
                ray=ray_photon,
                active=mi.Bool(True),
                weight = weight_photon,
                pos_grad = (loss_type=="RGBXY")
            )
            if mode is dr.ADMode.Primal:
                luisarender.indirect_update()
            
            del sampler_photon, spp_photon, ray_photon, weight_photon, emitter_photon
    
    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        self.pixels = film.crop_size()[0]*film.crop_size()[1]
        self.indirect_buffer = torch.zeros(self.pixels, 3).cuda()
        self.grad_vis_buffer = torch.zeros(self.pixels).cuda().contiguous()
        
        with dr.suspend_grad():
            image = self.render_direct(dr.ADMode.Primal, scene, sensor, seed, spp)
            self.render_indirect(dr.ADMode.Primal, scene, sensor, seed, spp)
            self.indirect_buffer *= 0
            luisarender.get_indirect(self.indirect_buffer.data_ptr(), self.indirect_buffer.shape[0], photon_per_iter, False)
            torch.cuda.synchronize()
            indirect = mi.TensorXf(self.indirect_buffer.reshape(film.crop_size()[1],film.crop_size()[0],3)/spp)
            self.indirect = indirect
        
        if loss_type=="RGBXY":
            image_torch = (image+indirect).torch()
            position = torch.zeros((image_torch.shape[0],image_torch.shape[1],2)).cuda()
            image_torch = torch.cat([image_torch,position],axis=-1)
            return mi.TensorXf(image_torch)
        else:
            return image+self.indirect
            

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        
        #if sys.argv[2]=="RGBXY":
        #    raise Exception("Not implemented forward for RGBXY derivatives")
        
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        
        film = sensor.film()
        with dr.resume_grad():
            image = self.render_direct(dr.ADMode.Forward, scene, sensor, seed, spp)
            grad_direct = dr.forward_to(image, flags = dr.ADFlag.ClearInterior)
            self.indirect_buffer *= 0
            self.render_indirect(dr.ADMode.Forward, scene, sensor, seed, spp)
            luisarender.get_indirect_origin(self.indirect_buffer.data_ptr(), self.indirect_buffer.shape[0], photon_per_iter, False)
            torch.cuda.synchronize()
            grad_indirect = mi.TensorXf(self.indirect_buffer.reshape(film.crop_size()[1],film.crop_size()[0],3)/spp)
            del image
            gc.collect()
        
        return grad_indirect+grad_direct
        
    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        
        with dr.resume_grad(): 
            if loss_type=="RGB":
                if optim_view:
                    image = self.render_direct(dr.ADMode.Backward, scene, sensor, seed, spp)
                    grad_c = mi.Spectrum(grad_in.torch()[...,None,:3].repeat(1,1,spp,1).reshape(-1,3))
                    dr.backward(grad_c*self.viewpoint_beta, flags=dr.ADFlag.ClearInterior)
                    dr.backward(image*grad_in, flags=dr.ADFlag.ClearInterior)
                if optim_photon:
                    self.render_indirect(dr.ADMode.Backward, scene, sensor, seed, spp, grad_in)
                
            elif loss_type=="RGBXY":
                if optim_view:
                    image = self.render_direct(dr.ADMode.Backward, scene, sensor, seed, spp)
                    grad_c = mi.Spectrum(grad_in.torch()[...,None,:3].repeat(1,1,spp,1).reshape(-1,3))
                    dr.backward(grad_c*self.viewpoint_beta)
                self.handle_epsm(grad_in, scene, sensor, seed, spp)
                if optim_photon:
                    self.render_indirect(dr.ADMode.Backward, scene, sensor, seed, spp, grad_in)


    def sample_view(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               weight: mi.Float
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], Any]:

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = False
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        L = mi.Spectrum(state_in if state_in is not None else 0)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)*weight                     # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        
        self.viewpoint_pos = mi.Point3f(-1000)                            # Diffusepos
        self.viewpoint_wo = mi.Vector3f(0)                            # Diffusepos
        self.viewpoint_beta = mi.Spectrum(0)                          # DiffuseBeta
        
        iteration = 0
        max_depth = self.max_depth
        
        #if tasks.exp_name=="torus":
        #    max_depth=2
        
        while iteration<max_depth and dr.any(active):
            iteration+=1
            active_next = mi.Bool(active)

            si = scene.ray_intersect(ray,
                                        ray_flags=mi.RayFlags.All,
                                        coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )
            Le = β * mis * ds.emitter.eval(si, active_next)

            # ---------------------- Emitter sampling ----------------------
            
            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
                    
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
            Lr_dir = β * mis_em * bsdf_value_em * em_weight

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                sampler.next_1d(),
                                                sampler.next_2d(),
                                                active_next)
            
            ray2 = dr.detach(ray)
            si_det = scene.ray_intersect(ray2,
                                        ray_flags=mi.RayFlags.All | mi.RayFlags.DetachShape,
                                        coherent=dr.eq(depth, 0))
            
            
            bsdf_det = si_det.bsdf(ray2)
            si_det.p = dr.detach(si_det.p)
            si_det.n = dr.detach(si_det.n)
            si_det.sh_frame.n = dr.detach(si_det.sh_frame.n)
            si_det.wi = dr.detach(si_det.wi)
            _, bsdf_weight2 = bsdf_det.sample(bsdf_ctx, si_det,
                                                sampler.next_1d(),
                                                sampler.next_2d(),
                                                active_next)

            is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse) & si.is_valid()
            self.viewpoint_pos = mi.Point3f(dr.select(is_diffuse,dr.detach(si.p),self.viewpoint_pos))
            self.viewpoint_beta = mi.Spectrum(dr.select(is_diffuse, β * bsdf_weight2, self.viewpoint_beta))
            self.viewpoint_wo = mi.Vector3f(dr.select(is_diffuse,dr.detach(-si.wi),self.viewpoint_wo))
            
            L = L + Le + Lr_dir
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            si_nxt = scene.ray_intersect(ray,
                                        ray_flags=mi.RayFlags.All,
                                        coherent=dr.eq(depth, 0))

            ds_nxt = mi.DirectionSample3f(scene, si=si_nxt, ref=si)

            mis_nxt = mis_weight(
                bsdf_sample.pdf,
                scene.pdf_emitter_direction(si, ds_nxt, ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta))
            )
            
            # with dr.resume_grad(when=not primal):
            Lr_end_dir = β * mis_nxt * ds_nxt.emitter.eval(si_nxt, active_next & is_diffuse) 
            L = L + Lr_end_dir

            if loss_type=="RGBXY" and mode is dr.ADMode.Backward:
                β *= bsdf_weight2
            else:
                β *= bsdf_weight
            
            η *= bsdf_sample.eta

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            depth[si.is_valid()] += 1                
            active = active_next & ~is_diffuse

        return (
            L ,                 # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )
    
    

    def photon_pass(self,
                    mode: dr.ADMode,
                    scene: mi.Scene,
                    sampler: mi.Sampler,
                    ray: mi.Ray3f,
                    active: mi.Bool,
                    weight: mi.Float,
                    pos_grad = False) -> Any:

        bsdf_ctx = mi.BSDFContext()
        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        β = weight                                    # Path throughput weight
        active = mi.Bool(active)                      # Active SIMD lanes

        iteration = 0
        max_depth = self.max_depth
        if mode==dr.ADMode.Backward:
            max_depth = 3
        
        backward_sum = mi.Point3f(0)
        while iteration<max_depth and dr.any(active):
            iteration+=1
            active_next = mi.Bool(active)
            
            pi = scene.ray_intersect_preliminary(ray,coherent=True,active=active)
            si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All)
            active_next &= (depth + 1 < max_depth) & si.is_valid()

            # Get the BSDF. Potentially computes texture-space differentials.
            bsdf = si.bsdf(ray)

            # ------------------ BSDF sampling -------------------
            #with dr.suspend_grad():
            
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                            sampler.next_1d(),
                                            sampler.next_2d(),
                                            active_next)
            
            if iteration>1:
                p_torch_ori = dr.select(active&si.is_valid(), si.p, mi.Point3f(-1000.,-1000.,-1000.)).torch().clone()
                p_torch = p_torch_ori.clone().contiguous()
                p_torch_ptr = p_torch.data_ptr()
                
                wi_torch = dr.select(active&si.is_valid(), -si.wi, mi.Vector3f(0)).torch().clone().contiguous()
                wi_torch_ptr = wi_torch.data_ptr()
                
                beta_torch = dr.select(active&si.is_valid(),β, mi.Spectrum(0)).torch().clone().contiguous()
                beta_torch_ptr = beta_torch.data_ptr()
                torch.cuda.synchronize()
                if mode == dr.ADMode.Primal:
                    luisarender.accum_ind(p_torch_ptr, wi_torch_ptr, beta_torch_ptr, p_torch.shape[0], False)
                elif mode == dr.ADMode.Forward:
                    dr.forward_to(si.p)
                    p_torch_fd_grad = dr.grad(si.p).torch().clone().contiguous()
                    dr.forward_to(β)
                    beta_torch_fd_grad = dr.grad(β).torch().clone().contiguous()
                    torch.cuda.synchronize()
                    luisarender.scatter_grad(p_torch_ptr, wi_torch_ptr,beta_torch_ptr, p_torch_fd_grad.data_ptr(),beta_torch_fd_grad.data_ptr(), photon_per_iter, p_torch.shape[0], False)
                else:
                    luisarender.compute_ind_grad(p_torch_ptr, wi_torch_ptr, beta_torch_ptr, p_torch.shape[0], photon_per_iter, self.grad_vis_buffer.data_ptr(), self.pixels, backward_query_radius, False)
                    p_torch = torch.clamp(torch.nan_to_num(p_torch,0),-1000,1000)#*backward_query_radius
                    beta_torch = torch.clamp(torch.nan_to_num(beta_torch,0),-10,10) 
                    wi_torch = torch.clamp(torch.nan_to_num(wi_torch,0),-1000,1000) 
                    grad_p = mi.Point3f(p_torch)
                    grad_beta = mi.Vector3f(beta_torch)
                    if pos_grad==True:
                        p_torch_pos = p_torch_ori.clone().contiguous()
                        torch.cuda.synchronize()
                        luisarender.accum_grad(p_torch_pos.data_ptr(), p_torch.shape[0], False)
                        p_torch_pos = torch.clamp(torch.nan_to_num(p_torch_pos,0),-1000,1000)
                        grad_p=mi.Point3f(p_torch_pos)
                        grad_beta = 0 # handle with epsm based method
                    
                    is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse)
                    grad_p = dr.select(active&is_diffuse&si.is_valid(), grad_p, mi.Point3f(0))
                    grad_beta = dr.select(active&is_diffuse&si.is_valid(), grad_beta, mi.Spectrum(0))
                    backward_sum += (si.p*grad_p)#+β*grad_beta)
                    active_next  = active_next & ~is_diffuse
            
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            
            #wo = si.to_local(ray.d)
            #bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_next)
            # bsdf_val_det = bsdf_weight * bsdf_sample.pdf
            # inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0),
            #                                 dr.rcp(bsdf_val_det), 0)
            
            β *= bsdf_weight 
            active_next &= dr.any(dr.neq(β, 0))
            depth[si.is_valid()] += 1
            active = active_next
            
        if mode == dr.ADMode.Backward:
            try:
                dr.backward(backward_sum)
            except:
                pass
        
    def sample_view_loop(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               weight: mi.Float
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], Any]:

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal
        bsdf_ctx = mi.BSDFContext()

            # --------------------- Configure loop state ----------------------

            # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex

            #self.rr_depth = 10

        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)*weight                     # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        self.viewpoint_pos = mi.Point3f(-1000)                            # Diffusepos
        self.viewpoint_wo = mi.Vector3f(0)                            # Diffusepos
        self.viewpoint_beta = mi.Spectrum(0)                          # DiffuseBeta
        
       
        loop = mi.Loop(name="PM ViewPath (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, δL, β, η, active,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                                      self.viewpoint_beta, self.viewpoint_pos, self.viewpoint_wo))
        
        while loop(active):
            
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            #if self.hide_emitters:
            #    active_next &= ~(dr.eq(depth, 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si, active_next)

            # ---------------------- Emitter sampling ----------------------
            
            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
            
            with dr.resume_grad(when=not primal):
                if not primal:
                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
                    em_val = scene.eval_emitter_direction(si, ds, active_em)
                    em_weight = dr.replace_grad(em_weight, dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0))
                    dr.disable_grad(ds.d)
                    
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                sampler.next_1d(),
                                                sampler.next_2d(),
                                                active_next)

            
            is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse) & si.is_valid()
            self.viewpoint_pos = mi.Point3f(dr.select(is_diffuse,dr.detach(si.p),self.viewpoint_pos))
            self.viewpoint_beta = mi.Spectrum(dr.select(is_diffuse,dr.detach(β * bsdf_weight), self.viewpoint_beta))
            self.viewpoint_wo = mi.Vector3f(dr.select(is_diffuse,dr.detach(-si.wi),self.viewpoint_wo))
            
            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # for diffuse end direct lighting
            with dr.resume_grad(when=not primal):
                si_nxt = scene.ray_intersect(ray,
                                            ray_flags=mi.RayFlags.All,
                                            coherent=dr.eq(depth, 0))

            ds_nxt = mi.DirectionSample3f(scene, si=si_nxt, ref=si)

            mis_nxt = mis_weight(
                bsdf_sample.pdf,
                scene.pdf_emitter_direction(si, ds_nxt, ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta))
            )
            
            with dr.resume_grad(when=not primal):
                Lr_end_dir = β * mis_nxt * ds_nxt.emitter.eval(si_nxt, active_next & is_diffuse) 

            L = (L + Lr_end_dir) if primal else (L - Lr_end_dir)

            η *= bsdf_sample.eta
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            depth[si.is_valid()] += 1                
            active = active_next & ~is_diffuse

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )
    
    def handle_epsm(self, grad_in, scene, sensor, seed, spp):
        
        sensor = scene.sensors()[1]
        film = sensor.film()
        aovs = self.aovs()
        assert(film.crop_size()[0]==128)
        spp = 8
        
        with dr.suspend_grad():
            
            sampler, spp = self.prepare(sensor, seed, 8, aovs)
            reparam = None
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                    sampler, reparam)
            L, valid, state_out, path_info = self.sample_epsm(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                active=mi.Bool(True),
                log_path=True
            )
        
            block = film.create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)
            
            with dr.resume_grad():
                dr.enable_grad(L)
                if (dr.all(mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special))):
                    aovs = sensor.film().prepare_sample(L * weight * 1, ray.wavelengths,
                                                        block.channel_count(),
                                                        weight=1,
                                                        alpha=dr.select(valid, mi.Float(1), mi.Float(0)))
                    block.put(pos, aovs)
                    del aovs
                else:
                    block.put(
                        pos=pos,
                        wavelengths=ray.wavelengths,
                        value=L * weight * det,
                        weight=det,
                        alpha=dr.select(valid, mi.Float(1), mi.Float(0))
                    )
                film.put_block(block)
                dr.schedule(state_out, block.tensor())
                image = film.develop()
                
                #mi.util.write_bitmap("epsm_render.png", image)
                #exit()
                
                tmp = ray.d.torch().reshape(-1,spp,3)
                height = film.crop_size()[0]
                width = film.crop_size()[1]
                
                grad_in_part = grad_in[:height,:width,:]
                grad_color_in = grad_in_part[...,:3]
                dr.set_grad(image, grad_color_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)
                δL = dr.grad(L)
                tmp = tmp.reshape(height, width, spp, 3)
                grad_d = tmp.clone()
                tmp_x = ray.d_x.torch().reshape(height, width,-1,3)
                tmp_y = ray.d_y.torch().reshape(height, width,-1,3)
                self.grad_pos = grad_in_part[...,3:].torch()[...,None,:].reshape(-1,2)
                grad_pos = grad_in_part[...,3:].torch()[...,None,:]
                grad_d = (tmp_x-tmp)*grad_pos[...,0:1]+(tmp_y-tmp)*grad_pos[...,1:2]
                dlduv = torch.zeros((path_info[1]["uv"][0].shape[0],1,len(path_info*2))).cuda()
                grad_d = mi.Vector3f(grad_d.reshape(-1,3))

                #camera grad enable
                if dr.grad_enabled(ray.o):
                    dr.backward(ray.o*-grad_d, flags = dr.ADFlag.ClearInterior)

                dr.enable_grad(ray.d)
                dr.set_grad(ray.d, grad_d)
                pi  = scene.ray_intersect_preliminary(ray, coherent=True)
                si = pi.compute_surface_interaction(ray, mi.RayFlags.All)
                dr.forward_to(si.p, flags=dr.ADFlag.ClearEdges)
                dlduv[:,0,0] = dr.grad(si.b0).torch()
                dlduv[:,0,1] = dr.grad(si.b1).torch()
                dldp1 = dr.grad(si.p).torch()
                dr.disable_grad(ray.d)
                dr.set_grad(ray.d, 0)
                Lt = L.torch()
                Lt = torch.sum(Lt,dim=-1)
                img = image.torch()
                img = torch.sum(img,dim=-1)
                path_grad, light_grad, diffuse_grad = self.calc_grad(path_info=path_info,dlduv=dlduv,dldp=dldp1,Lt=Lt)

        # Launch Monte Carlo sampling in backward AD mode (2)
        self.sample_epsm(
            mode=dr.ADMode.Backward,
            scene=scene,
            sampler=sampler,
            ray=ray,
            depth=mi.UInt32(0),
            δL=δL,
            state_in=state_out,
            active=mi.Bool(True),
            final_grad = path_grad,
            light_grad = light_grad,
            diffuse_grad = diffuse_grad,
            Lt=Lt
        )
        self.diffuse_pos_torch = self.diffusepoint_pos.torch().contiguous()
        self.diffuse_grad_torch = self.diffusepoint_grad.torch().contiguous()
        
        torch.cuda.synchronize()
    
    def sample_epsm(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               log_path=False,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        path_grad = kwargs.get("final_grad",mi.Point3f(0.0))
        light_grad = kwargs.get("light_grad",mi.Point3f(0.0))
        diffuse_grad = kwargs.get("diffuse_grad",mi.Point3f(0.0))
        
        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        iteration = 0
        logs=[{"cam":ray.o.torch()}]
        debug=False
        max_depth = min(self.max_depth, 6)
        
        #if tasks.exp_name=="torus":
        #    max_depth=2
        
        self.diffusepoint_pos = mi.Point3f(-1000)                            # Diffusepos
        self.diffusepoint_grad = mi.Point3f(-1000)                            # Diffusegrad
        
        while iteration<max_depth and dr.any(active):
            
            with dr.resume_grad(when=not primal):
                pi = scene.ray_intersect_preliminary(ray,coherent=True,active=active)
                si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All)
                si_follow = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape)
                if not primal and iteration*5+4<len(path_grad) and dr.grad_enabled(si.p):
                   dr.backward(si.p0*path_grad[iteration*5]+si.p1*path_grad[iteration*5+1]+si.p2*path_grad[iteration*5+2], flags = dr.ADFlag.ClearInterior)
                if not primal and iteration<len(diffuse_grad):
                    if dr.grad_enabled(si_follow.p):
                        dr.backward(si_follow.p*diffuse_grad[iteration], flags = dr.ADFlag.ClearInterior)
                        
            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)
            
            # ---------------------- Direct emission ----------------------

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid() 

            #print("dsf")
            if not primal and iteration<len(diffuse_grad):
                is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse) & si.is_valid()
                self.diffusepoint_pos = mi.Point3f(dr.select(is_diffuse,dr.detach(si.p),self.diffusepoint_pos))
                self.diffusepoint_grad = mi.Point3f(dr.select(is_diffuse,diffuse_grad[iteration],self.diffusepoint_grad))
                active_next = active_next & ~is_diffuse
            
            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    ds.d = dr.normalize(ds.p - si.p)
                    em_val = scene.eval_emitter_direction(si, ds, active_em)
                    em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)
                    dr.disable_grad(ds.d)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_em * bsdf_value_em * em_weight

                # if not primal:
                #     ray_direct = si.spawn_ray(ds.d)
                #     si_direct = scene.ray_intersect(ray_direct,
                #                         ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                #                         coherent=dr.eq(depth, 0),active=active_em)
                #     if iteration<len(light_grad) and dr.grad_enabled(si_direct.p):
                #         dr.backward(si_direct.p*mi.Vector3f(light_grad[iteration].torch()* torch.sum(Lr_dir.torch(),dim=-1,keepdim=True)), flags = dr.ADFlag.ClearInterior)
                #         dr.backward(si_direct.p*light_grad[iteration], flags = dr.ADFlag.ClearInterior)
                    
                    

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            
            # ------------------ Attached BSDF sampling -------------------
            with dr.resume_grad(when=not primal):
                bsdf = si.bsdf(ray)
                bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                    sampler.next_1d(),
                                                    sampler.next_2d(),
                                                    active_next)
                
                if not primal and iteration*5+4<len(path_grad) and dr.grad_enabled(si_follow.sh_frame.n):
                    dr.backward(bsdf_sample.hf*path_grad[iteration*5+4]+si_follow.sh_frame.n*path_grad[iteration*5+3], flags = dr.ADFlag.ClearInterior)
                #dr.disable_grad(bsdf_sample)        
                #dr.disable_grad(bsdf_weight)   
            
            if log_path and iteration<5:
                logs.append({"it":iteration,"active":(active&si.is_valid()).torch(),"bsdf":bsdf.flags(),"ismesh":si.ismesh.torch(),
                            "light":ds.p.torch(),"active_em":active_em.torch(),
                            "points":[si.p0.torch(),si.p1.torch(),si.p2.torch(),si.p.torch()],"uv":[si.b0.torch(),si.b1.torch()],"normal":si.sh_frame.n.torch(),\
                            "normals":[si.n0.torch(),si.n1.torch(),si.n2.torch()],\
                            "eta":bsdf_sample.eta.torch(),"hf":bsdf_sample.hf.torch()
                            })
            
            # ---- Update loop variables based on current interaction -----

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            depth[si.is_valid()] += 1
            active = active_next
            iteration+=1
        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L,                    # State for the differential phase，
            logs
        )  

    def calc_grad(self, path_info, dlduv, dldp, Lt):
        def find_orthogonal_vector(normal):
            v = torch.cat([torch.zeros_like(normal[:,0:1]), -normal[:,2:], normal[:,1:2]],dim=-1)
            v[normal[:,0]>0.9] = torch.cat([-normal[:,2:], torch.zeros_like(normal[:,1:2]), normal[:,0:1]],dim=-1)[normal[:,0]>0.9]
            v[normal[:,0]<-0.9] = torch.cat([-normal[:,2:], torch.zeros_like(normal[:,1:2]), normal[:,0:1]],dim=-1)[normal[:,0]<-0.9]
            return v / torch.norm(v,dim=-1,keepdim=True)

        def create_local_frame(normal):
            normal_normalized = normal / torch.norm(normal,dim=-1,keepdim=True)
            tangent = find_orthogonal_vector(normal_normalized)
            bitangent = torch.cross(normal_normalized, tangent, dim=-1)

            local_frame = torch.column_stack((tangent[:,None,:], bitangent[:,None,:], normal_normalized[:,None,:]))
            return local_frame

        constraint = torch.zeros((path_info[0]["cam"].shape[0],len(path_info)*2,len(path_info)*2)).cuda()
        diffuse_pos = torch.zeros((path_info[0]["cam"].shape[0])).cuda()
        hasdiffuse = torch.zeros((path_info[0]["cam"].shape[0])).cuda()
        param_grad_list = []
        final_param_grad = []
        diffuse_grad = []
        light_grad = []
        param_list = []

        def get_point(pointinfo):
            return pointinfo['points'][0]*pointinfo['uv'][0][...,None]+pointinfo['points'][1]*pointinfo['uv'][1][...,None]+pointinfo['points'][2]*(1-pointinfo['uv'][0]-pointinfo['uv'][1])[...,None]
        def get_normal(pointinfo):
            return pointinfo['normals'][0]*pointinfo['uv'][0][...,None]+pointinfo['normals'][1]*pointinfo['uv'][1][...,None]+pointinfo['normals'][2]*(1-pointinfo['uv'][0]-pointinfo['uv'][1])[...,None]

        def add(v):
            v.requires_grad_(True)
            v.retain_grad()
            param_list.append(v)
            param_grad_list.append(torch.zeros((v.shape[0],len(path_info)*2,v.shape[1])).cuda())
            final_param_grad.append(torch.zeros((v.shape[0],v.shape[1])).cuda())

        for id in range(1,len(path_info)):
            if id==1:
                point_prev = path_info[id-1]["cam"]
                path_info[id]["uv"][0].requires_grad_(True)
                path_info[id]["uv"][1].requires_grad_(True)
                path_info[id]["uv"][0].retain_grad()
                path_info[id]["uv"][1].retain_grad()
                add(path_info[id]["points"][0])
                add(path_info[id]["points"][1])
                add(path_info[id]["points"][2])
                isdiffuse = (mi.has_flag(path_info[id]["bsdf"],mi.BSDFFlags.Diffuse).torch()>0)
                dldp[isdiffuse==0] = 0
                diffuse_grad.append(dldp)
                ismesh = (path_info[id]["ismesh"]>0)
            else:
                point_prev = get_point(path_info[id-1])
                path_info[id]['uv'][0].requires_grad_(True)
                path_info[id]['uv'][1].requires_grad_(True)
                path_info[id]['uv'][0].retain_grad()
                path_info[id]['uv'][1].retain_grad()
                add(path_info[id]["points"][0])
                add(path_info[id]["points"][1])
                add(path_info[id]["points"][2])
                ismesh &= (path_info[id]["ismesh"]>0)
            
            isdiffuse = (mi.has_flag(path_info[id]["bsdf"],mi.BSDFFlags.Diffuse).torch()>0)
            hasdiffuse+=isdiffuse
            ismesh&=(hasdiffuse<2)
            diffuse_pos[isdiffuse] = id 
            nolight = (path_info[id]['active_em']==0)
            point_cur = get_point(path_info[id])
            ## light sampling 
            path_info[id]['light'].requires_grad_(True)
            path_info[id]['light'].retain_grad()
            point_next = path_info[id]['light']
            param_light_grad = torch.zeros((path_info[id]['light'].shape[0],len(path_info)*2,path_info[id]['light'].shape[1])).cuda()
            wi = point_prev-point_cur
            wo = point_next-point_cur
            wi = wi/torch.norm(wi,dim=-1,keepdim=True)#.detach()
            wo = wo/torch.norm(wo,dim=-1,keepdim=True)#.detach()
            n = get_normal(path_info[id])
            m = path_info[id]["hf"]
            add(n)
            add(m)
            transmat = create_local_frame(n)
            wi2 = torch.bmm(transmat,wi[...,None])[...,0]
            wo2 = torch.bmm(transmat,wo[...,None])[...,0]
            res = (wi2+wo2*path_info[id]["eta"][...,None])
            res = res/torch.norm(res,dim=-1,keepdim=True)#.detach()
            for i in range(2):
                loss = torch.sum(res[:,i])
                loss.backward(retain_graph=True)
                if id>1:
                    constraint[:,2*id+i-2,2*id-2] = path_info[id-1]['uv'][0].grad
                    constraint[:,2*id+i-2,2*id-1] = path_info[id-1]['uv'][1].grad
                    path_info[id-1]['uv'][0].grad = None
                    path_info[id-1]['uv'][1].grad = None
                constraint[:,2*id+i-2,2*id+0] = path_info[id+0]['uv'][0].grad
                constraint[:,2*id+i-2,2*id+1] = path_info[id-0]['uv'][1].grad
                path_info[id+0]['uv'][0].grad = None
                path_info[id-0]['uv'][1].grad = None

                for (idx,para) in enumerate(param_list):
                    if para.grad!=None:
                        param_grad_list[idx][:,2*id-2+i,:] = para.grad
                        para.grad = None
                    else:
                        param_grad_list[idx][:,2*id-2+i,:] = 0
                param_light_grad[:,2*id-2+i,:] = point_next.grad
                point_next.grad = None

            cur = constraint[:,:2*id,2:2*id+2].clone()
            cur[ismesh==0] = torch.eye(2*id).cuda()
            cur[path_info[id]['active']==0] = torch.eye(2*id).cuda()
            cur[nolight] = torch.eye(2*id).cuda()
            grad_uv_inv = torch.linalg.inv(cur)
            for (idx,param_grad) in enumerate(param_grad_list):
                duvdp = -torch.bmm(grad_uv_inv,param_grad[:,:2*id,:]) #(N,C,3)
                dldp = torch.bmm(dlduv[...,:2*id],duvdp)[:,0,:]
                dldp[ismesh==0] = 0
                dldp[path_info[id]['active']==0] = 0
                dldp[nolight] = 0
                dldp[hasdiffuse>0] = 0
                dldp = torch.nan_to_num(dldp)
                final_param_grad[idx]+=dldp
            
            duvdlp = -torch.bmm(grad_uv_inv,param_light_grad[:,:2*id,:]) #(N,C,3)
            dldlp = torch.bmm(dlduv[...,:2*id],duvdlp)[:,0,:] #(N,3)
            dldlp[ismesh==0] = 0
            dldlp[hasdiffuse>0] = 0
            dldlp[path_info[id]['active']==0] = 0
            dldlp[nolight] = 0
            dldlp = torch.nan_to_num(dldlp)
            light_grad.append(dldlp)
            
            # exit()
            if id<len(path_info)-1:
                path_info[id+1]['uv'][0].requires_grad_(True)
                path_info[id+1]['uv'][1].requires_grad_(True)
                path_info[id+1]['uv'][0].retain_grad()
                path_info[id+1]['uv'][1].retain_grad()
                point_next = get_point(path_info[id+1])
                point_next.retain_grad()
                wi = point_prev-point_cur
                wo = point_next-point_cur
                wi = wi/torch.norm(wi,dim=-1,keepdim=True)
                wo = wo/torch.norm(wo,dim=-1,keepdim=True)
                transmat = create_local_frame(n)
                wi2 = torch.bmm(transmat,wi[...,None])[...,0]
                wo2 = torch.bmm(transmat,wo[...,None])[...,0]
                res = (wi2+wo2*path_info[id]["eta"][...,None])
                res = res/torch.norm(res,dim=-1,keepdim=True)-m
                param_diffuse_grad = torch.zeros((path_info[id+1]['points'][3].shape[0],len(path_info)*2,path_info[id+1]['points'][3].shape[1])).cuda()
                for i in range(2):
                    loss = torch.sum(res[:,i])
                    loss.backward(retain_graph=True)
                    if id>1: 
                        constraint[:,2*id+i-2,2*id-2] = path_info[id-1]['uv'][0].grad
                        constraint[:,2*id+i-2,2*id-1] = path_info[id-1]['uv'][1].grad
                        path_info[id-1]['uv'][0].grad = None
                        path_info[id-1]['uv'][1].grad = None
                    constraint[:,2*id+i-2,2*id+0] = path_info[id+0]['uv'][0].grad
                    constraint[:,2*id+i-2,2*id+1] = path_info[id-0]['uv'][1].grad
                    constraint[:,2*id+i-2,2*id+2] = path_info[id+1]['uv'][0].grad
                    constraint[:,2*id+i-2,2*id+3] = path_info[id+1]['uv'][1].grad
                    path_info[id+0]['uv'][0].grad = None
                    path_info[id-0]['uv'][1].grad = None
                    path_info[id+1]['uv'][0].grad = None
                    path_info[id+1]['uv'][1].grad = None
                    for (idx,para) in enumerate(param_list):
                        if para.grad!=None:
                            param_grad_list[idx][:,2*id-2+i,:] = para.grad
                            para.grad = None
                    
                    param_diffuse_grad[:,2*id-2+i,:] = point_next.grad
                    point_next.grad = None
                
                cur = constraint[:,:2*id,2:2*id+2].clone()
                cur[ismesh==0] = torch.eye(2*id).cuda()
                cur[path_info[id+1]['active']==0] = torch.eye(2*id).cuda()
                grad_uv_inv = torch.linalg.inv(cur)
                for (idx,param_grad) in enumerate(param_grad_list):
                    duvdp = -torch.bmm(grad_uv_inv,param_grad[:,:2*id,:]) #(N,C,3)
                    dldp = torch.bmm(dlduv[...,:2*id],duvdp)[:,0,:]
                    dldp[ismesh==0]=0
                    dldp[path_info[id+1]['active']==0]=0
                    dldp[mi.has_flag(path_info[id+1]['bsdf'],mi.BSDFFlags.Diffuse).torch()==0] = 0
                    dldp = torch.nan_to_num(dldp)
                    dldp[hasdiffuse>0] = 0
                    final_param_grad[idx]+=dldp

                duvddp = -torch.bmm(grad_uv_inv,param_diffuse_grad[:,:2*id,:])
                dlddp = torch.bmm(dlduv[...,:2*id],duvddp)[:,0,:] #(N,3)
                dlddp[ismesh==0]=0
                dlddp[path_info[id+1]['active']==0]=0
                dlddp[mi.has_flag(path_info[id+1]['bsdf'],mi.BSDFFlags.Diffuse).torch()==0] = 0
                dlddp[hasdiffuse>0] = 0
                dlddp = torch.nan_to_num(dlddp)
                diffuse_grad.append(dlddp)
        #exit()
        for idx in range(len(final_param_grad)):
            final_param_grad[idx][final_param_grad[idx]>0.1] = 0
            final_param_grad[idx][final_param_grad[idx]<-0.1] = 0
            final_param_grad[idx][final_param_grad[idx]>0.01] = 0.01
            final_param_grad[idx][final_param_grad[idx]<-0.01] = -0.01
            final_param_grad[idx] = mi.Point3f(final_param_grad[idx])
        for idx in range(len(light_grad)):
            light_grad[idx][light_grad[idx]>1]  = 0
            light_grad[idx][light_grad[idx]<-1] = 0
            light_grad[idx] = mi.Point3f(light_grad[idx])
        for idx in range(len(diffuse_grad)):
            diffuse_grad[idx][diffuse_grad[idx]>0.01] = 0
            diffuse_grad[idx][diffuse_grad[idx]<-0.01] = 0
            diffuse_grad[idx] = mi.Point3f(diffuse_grad[idx])
        return final_param_grad,light_grad,diffuse_grad    

    #
    # graddis_avg = self.grad_vis_buffer.cpu().numpy()/self.spp
    # cmap = cm.get_cmap('viridis')
    # grad_reldis_vis = cmap(np.maximum(-1.0,np.minimum(graddis_avg,1.0)))
    # grad_reldis_vis = grad_reldis_vis.reshape((film.size()[1],film.size()[0],4)).astype(np.float32)
    # imageio.imwrite(f"outputs/grad_vis.png",(grad_reldis_vis * 255).astype(np.uint8))
    # exit()
    # Explicitly delete any remaining unused variables
    # del self.ray, self.weight, self.pos
    # del self.primal_image
    # gc.collect()
    # def sample_view2(self,
    #            mode: dr.ADMode,
    #            scene: mi.Scene,
    #            sampler: mi.Sampler,
    #            ray: mi.Ray3f,
    #            δL: Optional[mi.Spectrum],
    #            state_in: Any,
    #            active: mi.Bool,
    #            weight: mi.Float
    # ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], Any]:
    #     """
    #     See ``ADIntegrator.sample()`` for a description of this interface and
    #     the role of the various parameters and return values.
    #     """

    #     # Rendering a primal image? (vs performing forward/reverse-mode AD)
    #     primal = mode == dr.ADMode.Primal
    #     bsdf_ctx = mi.BSDFContext()

    #         # --------------------- Configure loop state ----------------------

    #         # Copy input arguments to avoid mutating the caller's state
    #     ray = mi.Ray3f(dr.detach(ray))
    #     depth = mi.UInt32(0)                          # Depth of current vertex

    #         #self.rr_depth = 10

    #     L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
    #     δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
    #     β = mi.Spectrum(1)*weight                     # Path throughput weight
    #     η = mi.Float(1)                               # Index of refraction
    #     active = mi.Bool(active)                      # Active SIMD lanes

    #     # Variables caching information from the previous bounce
    #     prev_si         = dr.zeros(mi.SurfaceInteraction3f)
    #     prev_bsdf_pdf   = mi.Float(1.0)
    #     prev_bsdf_delta = mi.Bool(True)
        
    #     iteration = 0
    #     max_depth = self.max_depth
        
    #     while dr.any(active) and iteration<max_depth:
    #         iteration+=1
    #         active_next = mi.Bool(active)
    #         # Compute a surface interaction that tracks derivatives arising
    #         # from differentiable shape parameters (position, normals, etc.)
    #         # In primal mode, this is just an ordinary ray tracing operation.
    #         si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

    #         # Get the BSDF, potentially computes texture-space differentials
    #         bsdf = si.bsdf(ray)

    #         # ---------------------- Direct emission ----------------------

    #         # Hide the environment emitter if necessary
    #         #if self.hide_emitters:
    #         #    active_next &= ~(dr.eq(depth, 0) & ~si.is_valid())

    #         # Compute MIS weight for emitter sample from previous bounce
    #         ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

    #         mis = mis_weight(
    #             prev_bsdf_pdf,
    #             scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
    #         )

    #         Le = β * mis * ds.emitter.eval(si, active_next)

    #         # ---------------------- Emitter sampling ----------------------
            
    #         # Should we continue tracing to reach one more vertex?
    #         active_next &= (depth + 1 < self.max_depth) & si.is_valid()

    #         # Is emitter sampling even possible on the current vertex?
    #         active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

    #         # If so, randomly sample an emitter without derivative tracking.
    #         ds, em_weight = scene.sample_emitter_direction(
    #             si, sampler.next_2d(), True, active_em)
    #         active_em &= dr.neq(ds.pdf, 0.0)
            
                
    #         wo = si.to_local(ds.d)
    #         bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
    #         mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
    #         Lr_dir = β * mis_em * bsdf_value_em * em_weight

    #         # ------------------ Detached BSDF sampling -------------------

    #         bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
    #                                             sampler.next_1d(),
    #                                             sampler.next_2d(),
    #                                             active_next)



    #         is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse) & si.is_valid()
            
    #         # ---- Update loop variables based on current interaction -----
    #         L = (L + Le + Lr_dir)
    #         ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

    #         # for diffuse end direct lighting
    #         si_nxt = scene.ray_intersect(ray,
    #                                     ray_flags=mi.RayFlags.All,
    #                                     coherent=dr.eq(depth, 0))

    #         ds_nxt = mi.DirectionSample3f(scene, si=si_nxt, ref=si)

    #         mis_nxt = mis_weight(
    #             bsdf_sample.pdf,
    #             scene.pdf_emitter_direction(si, ds_nxt, ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta))
    #         )
            
    #         Lr_end_dir = β * mis_nxt * ds_nxt.emitter.eval(si_nxt, active_next & is_diffuse) 

    #         L = (L + Lr_end_dir)
            
    #         # print(torch.sum(L[active&si.is_valid()].torch()))
    #         # viewmap_active[is_diffuse] = True
    #         # viewmap_direction[is_diffuse] = si.wi[is_diffuse]
    #         # viewmap_point[is_diffuse] = si.p[is_diffuse]
    #         # viewmap_beta[is_diffuse] = β[is_diffuse]

    #         η *= bsdf_sample.eta
    #         β *= bsdf_weight

    #         # Information about the current vertex needed by the next iteration

    #         prev_si = dr.detach(si, True)
    #         prev_bsdf_pdf = bsdf_sample.pdf
    #         prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

    #         # -------------------- Stopping criterion ---------------------

    #         # Don't run another iteration if the throughput has reached zero
    #         β_max = dr.max(β)
    #         active_next &= dr.neq(β_max, 0)
            
    #         # Russian roulette stopping probability (must cancel out ior^2
    #         # to obtain unitless throughput, enforces a minimum probability)
    #         rr_prob = dr.minimum(β_max * η**2, .95)

    #         # Apply only further along the path since, this introduces variance
    #         rr_active = depth >= self.rr_depth
    #         β[rr_active] *= dr.rcp(rr_prob)
    #         rr_continue = sampler.next_1d() < rr_prob
    #         active_next &= ~rr_active | rr_continue

    #         depth[si.is_valid()] += 1                
    #         active = active_next & ~is_diffuse
        
    #     return L
    #     #dr.backward(L*δL)
    # with dr.resume_grad():
    #         dr.enable_grad(self.Ld)
    #         block = film.create_block()
    #         block.set_coalesce(block.coalesce() and self.spp >= 4)
            
    #         #block.put(self.pos, self.ray.wavelengths, self.Ld, self.valid)
    #         #film.put_block(block)
            
    #         if (dr.all(mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special))):
    #             aovs = sensor.film().prepare_sample(self.Ld * self.weight, self.ray.wavelengths,
    #                                                 block.channel_count(),
    #                                                 weight=mi.Float(1.0),
    #                                                 alpha=dr.select(self.valid, mi.Float(1), mi.Float(0)))
    #             block.put(self.pos, aovs)
    #             del aovs
    #         else:
    #             block = film.create_block()
    #             block.set_coalesce(block.coalesce() and self.spp >= 4)
    #             block.put(
    #                 pos=self.pos,
    #                 wavelengths=self.ray.wavelengths,
    #                 value=self.Ld * self.weight,
    #                 weight=mi.Float(1.0),
    #                 alpha=dr.select(self.valid, mi.Float(1), mi.Float(0))
    #             )

    #         sensor.film().put_block(block)

    #         # Probably a little overkill, but why not.. If there are any
    #         # DrJit arrays to be collected by Python's cyclic GC, then
    #         # freeing them may enable loop simplifications in dr.eval().
    #         del self.valid
    #         gc.collect()

    #         # This step launches a kernel
    #         dr.schedule(self.state_out, block.tensor())
    #         image = sensor.film().develop()

    #         # Differentiate sample splatting and weight division steps to
    #         # retrieve the adjoint radiance
    #         dr.set_grad(image, grad_in)
    #         dr.enqueue(dr.ADMode.Backward, image)
    #         dr.traverse(mi.Float, dr.ADMode.Backward)
    #         δL = dr.grad(self.Ld)
    #         del self.Ld
        
mi.register_integrator("dpm", lambda props: DPMIntegrator(props))