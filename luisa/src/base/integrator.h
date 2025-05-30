//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/command_buffer.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/spectrum.h>
#include <base/light_sampler.h>
#include <base/camera.h>
#include "loss.h"
#include "optimizer.h"

namespace luisa::render {

class Pipeline;
class Display;

class Integrator : public SceneNode {

public:
    class Instance {

    private:
        Pipeline &_pipeline;
        const Integrator *_integrator;
        luisa::unique_ptr<Sampler::Instance> _sampler;
        luisa::unique_ptr<LightSampler::Instance> _light_sampler;

    public:
        explicit Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Integrator *integrator) noexcept;
        virtual ~Instance() noexcept = default;

        template<typename T = Integrator>
            requires std::is_base_of_v<Integrator, T>
        [[nodiscard]] auto
        node() const noexcept { return static_cast<const T *>(_integrator); }
        [[nodiscard]] auto &pipeline() noexcept { return _pipeline; }
        [[nodiscard]] const auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto sampler() noexcept { return _sampler.get(); }
        [[nodiscard]] auto sampler() const noexcept { return _sampler.get(); }
        [[nodiscard]] auto light_sampler() noexcept { return _light_sampler.get(); }
        [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler.get(); }
        virtual void render(Stream &stream) noexcept = 0;
        virtual luisa::vector<void *> render_backward(Stream &stream, luisa::vector<Buffer<float>> &grad_in) {
            LUISA_INFO("Not implemented!");
            return luisa::vector<void *>{};
        };
        virtual luisa::vector<void*> render_with_return(Stream &stream) {
            LUISA_INFO("Not implemented!");
            return luisa::vector<void *>{};
        }
    };

private:
    const Sampler *_sampler;
    const LightSampler *_light_sampler;

public:
    Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler; }
    [[nodiscard]] virtual bool is_differentiable() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

class ProgressiveIntegrator : public Integrator {

public:
    class Instance : public Integrator::Instance {

    protected:
        [[nodiscard]] virtual Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                                        Expr<uint2> pixel_id, Expr<float> time) const noexcept;
        virtual void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

    public:
        Instance(Pipeline &pipeline,
                 CommandBuffer &command_buffer,
                 const ProgressiveIntegrator *node) noexcept;
        ~Instance() noexcept override;
        void render(Stream &stream) noexcept override;
        luisa::vector<void*> render_with_return(Stream &stream) noexcept override;
    };

public:
    ProgressiveIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
};

class DifferentiableIntegrator : public ProgressiveIntegrator {

public:
    class Instance : public ProgressiveIntegrator::Instance {

    private:
        // luisa::vector<float4> _pixels;//store the rendered results for backward
        // luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float>>
        //     _render_shaders;
        // luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float, Image<float>>> _bp_shaders, _render_1spp_shaders;
        // luisa::unordered_map<const Camera::Instance *, Image<float>> _Li;

        // protected:
        // virtual void _render_one_camera(CommandBuffer &command_buffer, uint iteration,
        //                                 Camera::Instance *camera) noexcept;
        // virtual void _integrate_one_camera(CommandBuffer &command_buffer, uint iteration,
        //                                    const Camera::Instance *camera) noexcept;
        // [[nodiscard]] virtual Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
        //                                 Expr<uint2> pixel_id, Expr<float> time) const noexcept;
    protected:
        luisa::unique_ptr<Loss::Instance> _loss;
        luisa::unique_ptr<Optimizer::Instance> _optimizer;

        luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float>> _render_shaders;
        luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float, Image<float>>> _render_1spp_shaders;
        luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float, Image<float>, Buffer<float>>> _bp_shaders;
        luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float>> _render_grad_map_shaders;
        luisa::unordered_map<const Camera::Instance *, Image<float>> replay_Li;
        luisa::unordered_map<const Camera::Instance *, Buffer<float4>> last_time_rendered;
        virtual void _render_one_camera_backward(CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera, Buffer<float> & grad_in) noexcept;
        
    public:
        luisa::vector<void *> render_backward(Stream &stream, luisa::vector<Buffer<float>> &grad_in) noexcept override;
        Instance(Pipeline &pipeline, CommandBuffer &command_buffer,
                          const DifferentiableIntegrator *integrator) noexcept;
        ~Instance() noexcept override;
        [[nodiscard]] auto loss() const noexcept { return _loss.get(); }
        [[nodiscard]] auto optimizer() const noexcept { return _optimizer.get(); }
        // void render(Stream &stream) noexcept override;
    };
    
private:
    const Loss *_loss;
    const Optimizer *_optimizer;
    uint _iterations;
    int _display_camera_index;
    bool _save_process;
    bool _save_grad_map;
    bool _save_finite_diff;

public:
    DifferentiableIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] bool is_differentiable() const noexcept override { return true; }
    [[nodiscard]] auto loss() const noexcept { return _loss; }
    [[nodiscard]] auto optimizer() const noexcept { return _optimizer; }
    [[nodiscard]] auto iterations() const noexcept { return _iterations; }
    [[nodiscard]] int display_camera_index() const noexcept { return _display_camera_index; }
    [[nodiscard]] bool save_process() const noexcept { return _save_process; }
    [[nodiscard]] bool save_grad_map() const noexcept { return _save_grad_map; }
    [[nodiscard]] bool save_finite_diff() const noexcept { return _save_finite_diff; }
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Integrator::Instance)
