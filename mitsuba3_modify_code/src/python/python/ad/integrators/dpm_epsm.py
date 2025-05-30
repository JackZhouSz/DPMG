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

class DPMEPSMIntegrator(ADIntegrator):
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
            self.ray, self.weight, self.pos, _ = self.sample_rays(scene, sensor, sampler)

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
        for iter in range(self.spp):
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
        del self.Ld, self.indirect
        
        image = self.primal_image.torch()
        position = torch.zeros((image.shape[0],image.shape[1],2)).cuda()
        image = torch.cat([image,position],axis=-1)
        self.primal_image = mi.TensorXf(image)
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

        sensor = scene.sensors()[2]
        film = sensor.film()
        aovs = self.aovs()
        spp = 8
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, 8, aovs)
            reparam = None
            
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                    sampler, reparam)
            
            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, state_out, path_info = self.sample2(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True),
                log_path=True
            )
            
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
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

                sensor.film().put_block(block)

                # Probably a little overkill, but why not.. If there are any
                # DrJit arrays to be collected by Python's cyclic GC, then
                # freeing them may enable loop simplifications in dr.eval().
                del valid
                gc.collect()

                # This step launches a kernel
                dr.schedule(state_out, block.tensor())
                image = sensor.film().develop()

                tmp = ray.d.torch().reshape(-1,spp,3)
                height = film.size()[0]
                width = film.size()[1]
                grad_in_part = grad_in[:height,:width,:]
                #@Todo
                grad_color_in = grad_in_part[...,:3]
                dr.set_grad(image, grad_color_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)
                δL = dr.grad(L)
                tmp = tmp.reshape(height, width, spp, 3)
                grad_d = tmp.clone()
                tmp_x = ray.d_x.torch().reshape(height, width,-1,3)
                tmp_y = ray.d_y.torch().reshape(height, width,-1,3)
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
                #final_grad, light_grad, diffuse_grad = calc_grad(path_info=path_info,dlduv=dlduv,dldp=dldp1,Lt=L.torch())
                path_grad, light_grad, diffuse_grad = self.calc_grad(path_info=path_info,dlduv=dlduv,dldp=dldp1,Lt=Lt)
                #final_grad, light_grad, diffuse_grad = calc_grad_caustic(path_info=path_info,dlduv=dlduv,dldp=dldp1,Lt=Lt)
                #print(diffuse_grad[0].torch()[Lt==0])
                #exit()
        
        # Launch Monte Carlo sampling in backward AD mode (2)
        self.sample2(
            mode=dr.ADMode.Backward,
            scene=scene,
            sampler=sampler,
            ray=ray,
            depth=mi.UInt32(0),
            δL=δL,
            state_in=state_out,
            reparam=reparam,
            active=mi.Bool(True),
            final_grad = path_grad,
            light_grad = light_grad,
            diffuse_grad = diffuse_grad,
            Lt=Lt
        )
        self.diffuse_pos_torch = self.diffusepoint_pos.torch().contiguous()
        self.diffuse_grad_torch = self.diffusepoint_grad.torch().contiguous()
        
        # torch.cuda.synchronize()
        # luisarender.add_point_grad(diffuse_pos_torch.data_ptr(), diffuse_grad_torch.data_ptr(), diffuse_pos_torch.data_ptr(), diffuse_pos_torch.shape[0], False)

        # viewpoint_pos_torch = self.viewpoint_pos.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        # viewpoint_wo_torch = self.viewpoint_wo.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        # viewpoint_beta_torch = self.viewpoint_beta.torch().reshape((self.pixels,self.spp,3)).permute(1,0,2).contiguous()
        # #exit()
        # grad_in_color = grad_in.torch()[...,:3].clone()
        # grad_torch = grad_in_color.reshape(-1,3).contiguous()/self.spp
        
        # sensor = scene.sensors()[0]
        # film = sensor.film()
        # aovs = self.aovs()
        
        # torch.cuda.synchronize()
        # luisarender.update_grad(grad_torch.data_ptr(), grad_torch.shape[0],False)
        
        # with dr.resume_grad():
        #     for iter in range(self.spp):
        #         luisarender.add_viewpoint(viewpoint_pos_torch[iter].data_ptr(), viewpoint_wo_torch[iter].data_ptr(), viewpoint_beta_torch[iter].data_ptr(), viewpoint_pos_torch.shape[1], False)
        #         sampler_photon, spp_photon = self.prepare_photon(
        #             seed=self.seed+iter,
        #             sensor=sensor, 
        #             spp=self.photon_per_iter,
        #             aovs=self.aov_names()
        #         )
        #         # Generate a set of rays starting at the sensor
        #         ray_photon, weight_photon, emitter_photon = self.sample_photon_rays(scene, sensor, sampler_photon)
        #         #print(ray_photon.o.torch()[:3],ray_photon.d.torch()[:3], weight_photon.torch()[:3])
        #         self.photon_pass(
        #             mode=dr.ADMode.Backward,
        #             scene=scene,
        #             sampler=sampler_photon,
        #             ray=ray_photon,
        #             depth=mi.UInt32(0),
        #             δL=None,
        #             δaovs=None,
        #             state_in=None,
        #             active=mi.Bool(True),
        #             weight = weight_photon,
        #             pos_grad = True,
        #         )
        #         del sampler_photon, spp_photon, ray_photon, weight_photon, emitter_photon
        
        # torch.cuda.synchronize()
        # graddis_avg = self.grad_vis_buffer.cpu().numpy()/self.spp
        # cmap = cm.get_cmap('viridis')
        # grad_reldis_vis = cmap(np.maximum(-1.0,np.minimum(graddis_avg,1.0)))
        # grad_reldis_vis = grad_reldis_vis.reshape((512,512,4)).astype(np.float32)
        # imageio.imwrite(f"outputs/grad_vis.png",(grad_reldis_vis * 255).astype(np.uint8))
        # exit()
        # Explicitly delete any remaining unused variables
        # del self.ray, self.weight, self.pos
        #print(torch.max(grad_torch),torch.min(grad_torch))
        #exit()
        # grad_torch_vis = torch.sum(grad_torch, axis=-1).cpu().numpy().reshape((512,512))
        # normalize = mcolors.Normalize(vmin=-1, vmax=1)
        # s_map = cm.ScalarMappable(norm=normalize, cmap=cm.viridis)
        # grad_vis=s_map.to_rgba(grad_torch_vis)
        # plt.imsave("gradvis.png",grad_vis)
        # print(np.min(grad_torch_vis), np.max(grad_torch_vis), grad_torch_vis)
        # print(grad_vis)
        # exit()
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
                        #@Todo:fix it
                        grad_p = mi.Point3f(p_torch)*0
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
    
    def sample2(self,
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
        Lt = kwargs.get("Lt")
        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        iteration = 0
        logs=[{"cam":ray.o.torch()}]
        debug=False
        max_depth = min(self.max_depth, 6)
        
        
        self.diffusepoint_pos = mi.Point3f(-1000)                            # Diffusepos
        self.diffusepoint_grad = mi.Point3f(-1000)                            # Diffusepos
        while iteration<max_depth and dr.any(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
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
            if not primal and iteration<len(diffuse_grad):
                is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse) & si.is_valid()
                self.diffusepoint_pos = mi.Point3f(dr.select(is_diffuse,dr.detach(si.p),self.diffusepoint_pos))
                self.diffusepoint_grad = mi.Point3f(dr.select(is_diffuse,diffuse_grad[iteration],self.diffusepoint_grad))
                
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
                    # if iteration<len(light_grad) and dr.grad_enabled(si_direct.p):
                    #     dr.backward(si_direct.p*mi.Vector3f(light_grad[iteration].torch()* torch.sum(Lr_dir.torch(),dim=-1,keepdim=True)), flags = dr.ADFlag.ClearInterior)
                        #dr.backward(si_direct.p*light_grad[iteration], flags = dr.ADFlag.ClearInterior)
                    
                    

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

            # ------------------ Differential phase only ------------------

            if not primal:
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0),
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind

                    # if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                    #     raise Exception(
                    #         "The contribution computed by the differential "
                    #         "rendering phase is not attached to the AD graph! "
                    #         "Raising an exception since this is usually "
                    #         "indicative of a bug (for example, you may have "
                    #         "forgotten to call dr.enable_grad(..) on one of "
                    #         "the scene parameters, or you may be trying to "
                    #         "optimize a parameter that does not generate "
                    #         "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    # if mode == dr.ADMode.Backward:
                    #     dr.backward_from(δL * Lo)
                    # else:
                    #     δL += dr.forward_to(Lo)

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
            return v / torch.norm(v,dim=-1,keepdim=True)

        def create_local_frame(normal):
            normal_normalized = normal / torch.norm(normal,dim=-1,keepdim=True)
            tangent = find_orthogonal_vector(normal_normalized)
            bitangent = torch.cross(normal_normalized, tangent)

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
        for idx in range(len(final_param_grad)):
            final_param_grad[idx][final_param_grad[idx]>0.1] = 0
            final_param_grad[idx][final_param_grad[idx]<-0.1] = 0
            final_param_grad[idx] = mi.Point3f(final_param_grad[idx])
        for idx in range(len(light_grad)):
            light_grad[idx][light_grad[idx]>0.1] = 0
            light_grad[idx][light_grad[idx]<-0.1] = 0
            light_grad[idx] = mi.Point3f(light_grad[idx])
        for idx in range(len(diffuse_grad)):
            diffuse_grad[idx][diffuse_grad[idx]>0.1] = 0
            diffuse_grad[idx][diffuse_grad[idx]<-0.1] = 0
            diffuse_grad[idx] = mi.Point3f(diffuse_grad[idx])
        return final_param_grad,light_grad,diffuse_grad
    
mi.register_integrator("dpm_epsm", lambda props: DPMEPSMIntegrator(props))