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

class DPMFIntegrator(ADIntegrator):
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
            self.ray, self.weight, self.pos = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, aovs, _ = self.sample_view(
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
        
        del self.ray, self.weight, self.pos, sampler, valid, aovs, L, spp # Clean up
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
        with dr.resume_grad():
            import sys
            length = len(sys.argv)
            if length < 2:
                print('Please input command paramter to select scene:') 
                print('like: python foo.py mirror')
                exit()
            method = sys.argv[1]
            if method == 'water':
                print('forward scene is:', method)
                params = mi.traverse(scene)
                key_p = 'OBJMesh.vertex_positions'
                self.opt = mi.ad.Adam(lr=0.00001)
                self.opt['angle'] = mi.Float(0)

                positions_initial = dr.unravel(mi.Vector3f, params[key_p])
                
                dr.enable_grad(self.opt['angle'])
                dr.set_grad(self.opt['angle'], 0.02)
            
                def apply():
                    trafo = mi.Transform4f.rotate([0, 1, 0], self.opt['angle'])
                    params[key_p] = dr.ravel(trafo @ positions_initial)
                    params.update()
            
            elif method == 'mirror':
                print('forward scene is:', method)
                params = mi.traverse(scene)
                key_p = 'lighter.vertex_positions'
                
                self.opt = mi.ad.Adam(lr=0.00001)
                self.opt['angle'] = mi.Float(0)

                positions_initial = dr.unravel(mi.Vector3f, params[key_p])
                
                dr.enable_grad(self.opt['angle'])
                dr.set_grad(self.opt['angle'], 0.02)
            
                def apply():
                    # trafo = mi.Transform4f.translate(mi.Point3f([self.opt['angle'], 0, 0]))
                    # new_p = positions_initial
                    # new_p.x += self.opt['angle']
                    # print(positions_initial)
                    # print(new_p)
                    params[key_p] = dr.ravel(positions_initial + [self.opt['angle'], 0.0, 0.0])
                    params.update()
            else:
                print('please set your scene name')
            
            apply()
            
            for iter in range(self.spp):
                
                dr.enable_grad(self.opt['angle'])
                dr.set_grad(self.opt['angle'], 1.0)
                apply()
                
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
                    mode=dr.ADMode.Forward,
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
                del sampler_photon, spp_photon, ray_photon, weight_photon, emitter_photon
        
        print("photon finish")
        self.indirect_buffer *= 0
        luisarender.get_indirect_origin(self.indirect_buffer.data_ptr(), self.indirect_buffer.shape[0], self.photon_per_iter, False)
        torch.cuda.synchronize()
        self.indirect = mi.TensorXf(self.indirect_buffer.reshape(512,512,3)/self.spp)
        gc.collect()
        self.primal_image = self.indirect
        del self.Ld, self.indirect
        return self.primal_image
    
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
        #@Todo:change it to EPSM formulation
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
        
        self.point_logs = [mi.Point3f(-1000),mi.Point3f(-1000),mi.Point3f(-1000),mi.Point3f(-1000)]
        self.normal_logs = [mi.Vector3f(-1000),mi.Vector3f(-1000),mi.Vector3f(-1000),mi.Vector3f(-1000)]
        self.bary_logs = [mi.Point2f(-1000),mi.Point2f(-1000),mi.Point2f(-1000),mi.Point2f(-1000)]
        self.eta_logs = [mi.Float(-1000),mi.Float(-1000),mi.Float(-1000),mi.Float(-1000)]
        self.flag_logs = [mi.Int(-1000),mi.Int(-1000),mi.Int(-1000),mi.Int(-1000)]
        #iter = mi.Int(0)
        # Record the following loop in its entirety
        loop = mi.Loop(name="PM ViewPath Backpropagation (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, δL, β, η, active,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                                      self.viewpoint_beta, self.viewpoint_pos, self.viewpoint_wo,
                                      self.point_logs, self.normal_logs, self.bary_logs, self.eta_logs, self.flag_logs))

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

            # ---- Update loop variables based on current interaction -----
            
            is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse) & si.is_valid()
            self.viewpoint_pos = mi.Point3f(dr.select(is_diffuse,dr.detach(si.p),self.viewpoint_pos))
            self.viewpoint_beta = mi.Spectrum(dr.select(is_diffuse,dr.detach(β * bsdf_weight),self.viewpoint_beta))
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
                L = (L + β * mis_nxt * ds_nxt.emitter.eval(si_nxt, active_next & is_diffuse)) 

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
        #print(β.torch()[:3])
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
                    
                    elif mode == dr.ADMode.Forward:
                        dr.forward_to(si.p, flags = dr.ADFlag.ClearInterior)
                        #dr.forward_to(β, flags = dr.ADFlag.ClearInterior)
                        #dr.forward_to(si.wi)
                        p_torch_fd_grad = dr.grad(si.p).torch().clone().contiguous()
                        beta_torch_fd_grad = dr.grad(β).torch().clone().contiguous()
                        torch.cuda.synchronize()
                        # print(p_torch[1024:1024+100],p_torch_fd_grad[1024:1024+100])
                        # print(torch.max(p_torch_fd_grad),torch.min(p_torch_fd_grad),torch.sum(p_torch_fd_grad))
                        #torch.cuda.synchronize()
                        luisarender.scatter_grad(p_torch_ptr, wi_torch_ptr, beta_torch_ptr, p_torch_fd_grad.data_ptr(),beta_torch_fd_grad.data_ptr(), self.photon_per_iter, p_torch.shape[0], False)
                        #exit()
                    
                ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
                β *= bsdf_weight 
                active_next &= dr.any(dr.neq(β, 0))
                depth[si.is_valid()] += 1
                active = active_next
    
mi.register_integrator("dpm_forward", lambda props: DPMFIntegrator(props))