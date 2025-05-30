from __future__ import annotations # Delayed parsing of type annotations

import luisarender

import numpy as np

import drjit as dr
import mitsuba as mi
import gc
import torch
import math
from .common import RBIntegrator, ADIntegrator


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import imageio
from tqdm.std import tqdm

def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    b2 = dr.sqr(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))

class PTracer2Integrator(ADIntegrator):
    def prepare_photon(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        wavefront_size = spp

        sampler.seed(seed, wavefront_size)
        return sampler, spp

    def sample_photon_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray
        """

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        
        wavelength_sample  = sampler.next_1d()
        direction_sample = sampler.next_2d()
        position_sample  = sampler.next_2d()

        ray, ray_weight, emitter = scene.sample_emitter_ray(time, wavelength_sample, direction_sample, position_sample, mi.Bool(True))
    
        return ray, ray_weight, emitter
    
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

        self.pixels = film.size()[0]*film.size()[1]
        self.photon_per_iter = 1048576*4
        
        self.seed = seed
        self.Ld = None
        self.spp = spp
        self.indirect_buffer = torch.zeros(self.pixels, 3).cuda()
        
        with dr.suspend_grad():
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=self.seed,
                spp=self.spp,
                aovs=self.aov_names()
            )

            # Generate a set of rays starting at the sensor
            self.ray, self.weight, self.pos, self.det = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, self.valid, aovs, self.state_out = self.sample_view(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=self.ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True),
                weight=self.weight
            )
            
            if self.Ld is None:
                self.Ld = L
            else:
                self.Ld += L
        
        block = film.create_block()
        block.set_coalesce(block.coalesce() and spp >= 4)
        block.put(self.pos, self.ray.wavelengths, L, mi.Float(1))
        
        film.put_block(block)
        #print("viewpath finish")
        
        del aovs, L, spp # Clean up
        #return film.develop()
        
        #print(self.pixels*spp, self.viewpoint_pos.torch().shape)
        #print(self.viewpoint_beta)
        #print(torch.max(self.viewpoint_beta.torch()), torch.min(self.viewpoint_beta.torch()))
        
        # get indirect
        viewpoint_pos_torch = self.viewpoint_pos.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        viewpoint_wo_torch = self.viewpoint_wo.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        viewpoint_beta_torch = self.viewpoint_beta.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        torch.cuda.synchronize()
        
        luisarender.init_viewpointmap(self.pixels, 0.01, 128*128*8)
        for iter in range(0):
            luisarender.add_viewpoint(viewpoint_pos_torch[iter].data_ptr(), viewpoint_wo_torch[iter].data_ptr(), viewpoint_beta_torch[iter].data_ptr(), viewpoint_pos_torch.shape[1], False)
            sampler_photon, spp_photon = self.prepare_photon(
                seed=self.seed+iter,
                sensor=sensor, 
                spp=self.photon_per_iter,
                aovs=self.aov_names()
            )
                # Generate a set of rays starting at the sensor
            ray_photon, weight_photon, emitter_photon = self.sample_photon_rays(scene, sensor, sampler_photon)
                #print(ray_photon.o.torch()[:3],ray_photon.d.torch()[:3], weight_photon.torch()[:3])
            self.photon_pass(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler_photon,
                ray=ray_photon,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True),
                weight = weight_photon
            )
            luisarender.indirect_update()
            del sampler_photon, spp_photon, ray_photon, weight_photon, emitter_photon
            #print(f"photon{iter}")
            #luisarender.get_indirect(self.indirect_buffer.data_ptr(), self.indirect_buffer.shape[0], self.photon_per_iter, False)
        
        #print("photon finish")
        self.indirect_buffer *= 0
        luisarender.get_indirect(self.indirect_buffer.data_ptr(), self.indirect_buffer.shape[0], self.photon_per_iter, False)
        torch.cuda.synchronize()
        self.indirect = mi.TensorXf(self.indirect_buffer.reshape(film.size()[1],film.size()[0],3)/self.spp)
        gc.collect()
        self.primal_image = film.develop()+self.indirect
        del self.indirect
        return self.primal_image

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        
        self.grad_vis_buffer = torch.zeros(self.pixels).cuda().contiguous()
        self.grad_query_radius = 0.01
        luisarender.init_viewpointmap(self.pixels, self.grad_query_radius, 128*128*8)

        film = sensor.film()
        aovs = self.aovs()
        #print(torch.max(grad_torch),torch.min(grad_torch))
        #exit()
        # grad_torch_vis = torch.sum(grad_torch, axis=-1).cpu().numpy().reshape((film.size()[0],film.size()[1]))
        # normalize = mcolors.Normalize(vmin=-1, vmax=1)
        # s_map = cm.ScalarMappable(norm=normalize, cmap=cm.viridis)
        # grad_vis=s_map.to_rgba(grad_torch_vis)
        # plt.imsave("gradvis.png",grad_vis)
        # print(np.min(grad_torch_vis), np.max(grad_torch_vis), grad_torch_vis)
        # print(grad_vis)
        # exit()
        
        # view path backward
        with dr.resume_grad():
            dr.enable_grad(self.Ld)

                # Accumulate into the image block.
                # After reparameterizing the camera ray, we need to evaluate
                #   Σ (fi Li det)
                #  ---------------
                #   Σ (fi det)
            block = film.create_block()
            block.set_coalesce(block.coalesce() and self.spp >= 4)
            
            #block.put(self.pos, self.ray.wavelengths, self.Ld, self.valid)
            #film.put_block(block)
            
            if (dr.all(mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special))):
                aovs = sensor.film().prepare_sample(self.Ld * self.weight * self.det, self.ray.wavelengths,
                                                    block.channel_count(),
                                                    weight=self.det,
                                                    alpha=dr.select(self.valid, mi.Float(1), mi.Float(0)))
                block.put(self.pos, aovs)
                del aovs
            else:
                block.put(
                    pos=self.pos,
                    wavelengths=self.ray.wavelengths,
                    value=self.Ld * self.weight * self.det,
                    weight=self.det,
                    alpha=dr.select(self.valid, mi.Float(1), mi.Float(0))
                )

            sensor.film().put_block(block)

            # Probably a little overkill, but why not.. If there are any
            # DrJit arrays to be collected by Python's cyclic GC, then
            # freeing them may enable loop simplifications in dr.eval().
            del self.valid
            gc.collect()

            # This step launches a kernel
            dr.schedule(self.state_out, block.tensor())
            image = sensor.film().develop()

            # Differentiate sample splatting and weight division steps to
            # retrieve the adjoint radiance
            dr.set_grad(image, grad_in)
            dr.enqueue(dr.ADMode.Backward, image)
            dr.traverse(mi.Float, dr.ADMode.Backward)
            δL = dr.grad(self.Ld)
            del self.Ld
          
        with dr.suspend_grad(): 
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=self.seed,
                spp=self.spp,
                aovs=self.aov_names()
            )

            # # Generate a set of rays starting at the sensor
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
                # Launch the Monte Carlo sampling process in primal mode
            
        with dr.resume_grad(): 
            self.sample_view2(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=δL,
                δaovs=None,
                state_in=self.state_out,
                active=mi.Bool(True),
                weight=weight
            )
        
        gc.collect()
        
        return
        # indirect phase
        #print(grad_in.torch().shape)
        #exit()
        #print(self.viewpoint_pos.torch(),self.viewpoint_pos.torch().shape)
        viewpoint_pos_torch = self.viewpoint_pos.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        viewpoint_wo_torch = self.viewpoint_wo.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        viewpoint_beta_torch = self.viewpoint_beta.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        #exit()
        grad_in_color = grad_in.torch()[...,:3].clone()
        grad_torch = grad_in_color.reshape(-1,3).contiguous()/self.spp
        
        torch.cuda.synchronize()
        luisarender.update_grad(grad_torch.data_ptr(), grad_torch.shape[0],False)
        
        with dr.resume_grad():
            for iter in range(0):
                luisarender.add_viewpoint(viewpoint_pos_torch[iter].data_ptr(), viewpoint_wo_torch[iter].data_ptr(), viewpoint_beta_torch[iter].data_ptr(), viewpoint_pos_torch.shape[1], False)
                sampler_photon, spp_photon = self.prepare_photon(
                    seed=self.seed+iter,
                    sensor=sensor, 
                    spp=self.photon_per_iter,
                    aovs=self.aov_names()
                )
                # Generate a set of rays starting at the sensor
                ray_photon, weight_photon, emitter_photon = self.sample_photon_rays(scene, sensor, sampler_photon)
                #print(ray_photon.o.torch()[:3],ray_photon.d.torch()[:3], weight_photon.torch()[:3])
                self.photon_pass(
                    mode=dr.ADMode.Backward,
                    scene=scene,
                    sampler=sampler_photon,
                    ray=ray_photon,
                    depth=mi.UInt32(0),
                    δL=None,
                    δaovs=None,
                    state_in=None,
                    active=mi.Bool(True),
                    weight = weight_photon,
                    pos_grad = False,
                )
                del sampler_photon, spp_photon, ray_photon, weight_photon, emitter_photon
        
        torch.cuda.synchronize()
        # graddis_avg = self.grad_vis_buffer.cpu().numpy()/self.spp
        # cmap = cm.get_cmap('viridis')
        # grad_reldis_vis = cmap(np.maximum(-1.0,np.minimum(graddis_avg,1.0)))
        # grad_reldis_vis = grad_reldis_vis.reshape((film.size()[0],film.size()[1],4)).astype(np.float32)
        # imageio.imwrite(f"outputs/grad_vis.png",(grad_reldis_vis * 255).astype(np.uint8))
        # exit()
        # Explicitly delete any remaining unused variables
        # del self.ray, self.weight, self.pos
        del self.primal_image
        gc.collect()
    
    def sample_view(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               δaovs: Optional[mi.Spectrum],
               state_in: Any,
               active: mi.Bool,
               weight: mi.Float
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], Any]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

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

        # # # Specify the max. number of loop iterations (this can help avoid
        # # # costly synchronization when wavefront-style loops are generated)
        # loop.set_max_iterations(self.max_depth)
        
        #iteration = 0
        #max_depth = min(self.max_depth, 4)
        
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
            
            # iteration = iteration+1
            # self.point_logs[iteration][active_next] = si.p[active_next]
            # self.normal_logs[iteration][active_next] = si.n[active_next]
            # self.bary_logs[iteration][active_next] = si.uv[active_next]
            # self.eta_logs[iteration][active_next] = bsdf_sample.eta[active_next]
            # self.flag_logs[iteration][active_next] = bsdf.flags()[active_next]
            
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
            
            # viewmap_active[is_diffuse] = True
            # viewmap_direction[is_diffuse] = si.wi[is_diffuse]
            # viewmap_point[is_diffuse] = si.p[is_diffuse]
            # viewmap_beta[is_diffuse] = β[is_diffuse]

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
            [],                 # AOVs (not supported in this integrator)
            L                    # State for the differential phase
        )

    def photon_pass(self,
                    mode: dr.ADMode,
                    scene: mi.Scene,
                    sampler: mi.Sampler,
                    ray: mi.Ray3f,
                    depth: mi.UInt32,
                    δL: Optional[mi.Spectrum],
                    δaovs: Optional[mi.Spectrum],
                    state_in: Any,
                    active: mi.Bool,
                    weight: mi.Float,
                    pos_grad = False) -> Any:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """


        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        β = weight                                    # Path throughput weight
        active = mi.Bool(active)                      # Active SIMD lanes

        # Record the following loop in its entirety
        #loop = mi.Loop(name="Photon path in mode (%s)" % mode.name,
        #               state=lambda: (sampler, ray, depth, L, δL, β, active))


        iteration = 0
        max_depth = self.max_depth
        
        while iteration<max_depth and dr.any(active):
            iteration+=1
            active_next = mi.Bool(active)

            # ---------------------- Direct emission ----------------------
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            
            with dr.resume_grad():
                pi = scene.ray_intersect_preliminary(ray,coherent=True,active=active)
                #si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All|mi.RayFlags.FollowShape)
                si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All)
                # Hide the environment emitter if necessary
                # if self.hide_emitters:
                #     active_next &= ~(dr.eq(depth, 0) & ~si.is_valid())

                # Differentiable evaluation of intersected emitter / envmap
                # with dr.resume_grad(when=not primal):
                #     Le = β * si.emitter(scene).eval(si, active_next)

                # Should we continue tracing to reach one more vertex?
                active_next &= (depth + 1 < self.max_depth) & si.is_valid()

                # Get the BSDF. Potentially computes texture-space differentials.
                bsdf = si.bsdf(ray)

                # ------------------ BSDF sampling -------------------

                bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                    sampler.next_1d(),
                                                    sampler.next_2d(),
                                                    active_next)
                
                if iteration>1:
                    p_torch_ori = dr.select(active&si.is_valid(), si.p, mi.Point3f(-1000.,-1000.,-1000.)).torch().clone()
                    p_torch = p_torch_ori.clone().contiguous()
                    
                    p_torch_pos = p_torch_ori.clone().contiguous()
                    p_torch_ptr = p_torch.data_ptr()
                    
                    wi_torch = dr.select(active&si.is_valid(), -si.wi, mi.Vector3f(0)).torch().clone().contiguous()
                    wi_torch_ptr = wi_torch.data_ptr()
                    
                    beta_torch = β.torch().clone().contiguous()
                    beta_torch_ptr = beta_torch.data_ptr()
                    torch.cuda.synchronize()
                    if mode == dr.ADMode.Primal:
                        luisarender.accum_ind(p_torch_ptr, wi_torch_ptr, beta_torch_ptr, p_torch.shape[0], False)
                    else:
                        luisarender.compute_ind_grad(p_torch_ptr, wi_torch_ptr, beta_torch_ptr, p_torch.shape[0], self.photon_per_iter, self.grad_vis_buffer.data_ptr(), self.pixels, self.grad_query_radius, False)
                        p_torch = torch.clamp(torch.nan_to_num(p_torch,0),-1000,1000)#*self.grad_query_radius
                        beta_torch = torch.clamp(torch.nan_to_num(beta_torch,0),-10,10) 
                        wi_torch = torch.clamp(torch.nan_to_num(wi_torch,0),-1000,1000) 
                        grad_p = mi.Point3f(p_torch)
                        grad_beta = mi.Vector3f(beta_torch)
                        if pos_grad==True:
                            luisarender.accum_grad(p_torch_pos.data_ptr(), p_torch.shape[0], False)
                            p_torch_pos = torch.clamp(torch.nan_to_num(p_torch_pos,0),-1000,1000)
                            grad_p+=mi.Point3f(p_torch_pos)
                        sum = si.p*grad_p+β*grad_beta
                        if iteration<max_depth:
                            dr.backward(sum,flags = dr.ADFlag.ClearInterior)
                        else:
                            dr.backward(sum)
                
                ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
                β *= bsdf_weight 
                active_next &= dr.any(dr.neq(β, 0))
                depth[si.is_valid()] += 1
                active = active_next

    def sample_view2(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               δaovs: Optional[mi.Spectrum],
               state_in: Any,
               active: mi.Bool,
               weight: mi.Float
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], Any]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal
        bsdf_ctx = mi.BSDFContext()

            # --------------------- Configure loop state ----------------------

            # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex

            #self.rr_depth = 10

        L = mi.Spectrum(0)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)*weight                     # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        
        iteration = 0
        max_depth = self.max_depth
        
        while dr.any(active) and iteration<max_depth:
            iteration+=1
            active_next = mi.Bool(active)
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

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

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample1, bsdf_weight1 = bsdf.sample(bsdf_ctx, si,
                                                sampler.next_1d(),
                                                sampler.next_2d(),
                                                active_next)
            #rev_wo = si.to_local(dr.detach(si.to_world(bsdf_sample1.wo)))
            #print("abcd", si.wi.torch()[:], bsdf_sample.wo.torch()[:])
            #si.wi[mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta)] = rev_wo[mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta)]
            def find_orthogonal_vector(normal):
                v = torch.cat([torch.zeros_like(normal[:,0:1]), -normal[:,2:], normal[:,1:2]],dim=-1)
                return v / torch.norm(v,dim=-1,keepdim=True)

            def create_local_frame(normal):
                normal_normalized = normal / torch.norm(normal,dim=-1,keepdim=True)
                tangent = find_orthogonal_vector(normal_normalized)
                bitangent = torch.cross(normal_normalized, tangent)

                local_frame = torch.column_stack((tangent[:,None,:], bitangent[:,None,:], normal_normalized[:,None,:]))
                return local_frame
            
            ray_nxt = si.spawn_ray(si.to_world(bsdf_sample1.wo))
            active = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta)
            si_pre = dr.detach(ray.o).torch()[:]
            si_nxt = dr.detach(si.p+si.to_world(bsdf_sample1.wo)).torch()[:]
            
            u = si.b0.torch()[:].requires_grad_()
            v = si.b1.torch()[:].requires_grad_()
            p0 = si.p0.torch()[:]
            p1 = si.p1.torch()[:]
            p2 = si.p2.torch()[:]
            si_cur_old = dr.detach(si.p).torch()[:]
            si_cur = u[...,None]*p0+v[...,None]*p1+(1-u[...,None]-v[...,None])*p2
            #print("check si_cur", si_cur-si_cur_old)
            #exit()
            si_cur_n = dr.detach(si.n).torch()[:].requires_grad_()
            #print(si_pre.shape)
            wi = (si_pre-si_cur)/torch.norm(si_pre-si_cur,dim=-1,keepdim=True)
            wo = (si_nxt-si_cur)/torch.norm(si_nxt-si_cur,dim=-1,keepdim=True)
            curn = (si_cur_n)/torch.norm(si_cur_n,dim=-1,keepdim=True)
            transmat = create_local_frame(curn)
            wi2 = torch.bmm(transmat,wi[...,None])[...,0]
            wo2 = torch.bmm(transmat,wo[...,None])[...,0]
            res = (wi2+wo2)
            #print("check_res is zero", res)
            mat_0 = torch.zeros((res.shape[0],2,2)).cuda()
            mat_1 = torch.zeros((res.shape[0],2,3)).cuda()
            torch.sum(res[...,0]).backward(retain_graph=True)
            mat_0[:,0,0] = u.grad
            mat_0[:,0,1] = v.grad
            mat_1[:,0,:] = si_cur_n.grad
            #print("check uv grad",u.grad, v.grad,si_cur_n.grad)
            u.grad = None
            v.grad = None
            si_cur_n.grad  = None
            ##286*512*16:286*512*16+256
            torch.sum(res[...,1]).backward(retain_graph=True)
            mat_0[:,1,0] = u.grad
            mat_0[:,1,1] = v.grad
            mat_1[:,1,:] = si_cur_n.grad
            u.grad = None
            v.grad = None
            si_cur_n.grad  = None
            grad_uv_inv = torch.linalg.inv(mat_0)
            grad_uv_curn = -torch.bmm(grad_uv_inv,mat_1[:,:2,:]) #(N,C,3)
            beta = (1-wo2[...,2])
            torch.sum(beta).backward(retain_graph=True)
            dbetadn1 = si_cur_n.grad
            #print("dbetadn1 ", dbetadn1)
            grad_uv = torch.cat([u.grad[...,None],v.grad[...,None]],axis=-1)[:,None,:]
            #print(grad_uv.shape, grad_uv_curn.shape)
            dbetadn2 = torch.bmm(grad_uv,grad_uv_curn)[:,0,:]
            #print("dbetadn2 ", dbetadn2)
            final_grad_n = dbetadn1+dbetadn2
            final_grad_n = torch.nan_to_num(final_grad_n)
            grad_n = mi.Vector3f(final_grad_n)
            dr.backward(si.sh_frame.n[active_next]*grad_n[active_next])
            return
            #exit()
            #print("dcba", si.wi.torch()[:], bsdf_sample.wo.torch()[:])
            #exit()
            # ---- Update loop variables based on current interaction -----
            L = (L + Le + Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample1.wo))

            # for diffuse end direct lighting
            
            # print(torch.sum(L[active&si.is_valid()].torch()))
            # viewmap_active[is_diffuse] = True
            # viewmap_direction[is_diffuse] = si.wi[is_diffuse]
            # viewmap_point[is_diffuse] = si.p[is_diffuse]
            # viewmap_beta[is_diffuse] = β[is_diffuse]

            η *= bsdf_sample2.eta
            β *= bsdf_weight2

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample2.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample2.sampled_type, mi.BSDFFlags.Delta)

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
        
        dr.backward(L*δL)
        
mi.register_integrator("ptracer2", lambda props: PTracer2Integrator(props))