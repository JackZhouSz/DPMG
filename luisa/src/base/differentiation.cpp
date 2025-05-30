//
// Created by Mike Smith on 2022/3/9.
//

#include "dsl/builtin.h"
#include <tinyexr.h>

#include <dsl/syntax.h>
#include <base/pipeline.h>
#include <base/differentiation.h>
#include <base/geometry.h>
#include <util/rng.h>
#include <util/sampling.h>

namespace luisa::render {

// #define LUISA_RENDER_DIFFERENTIATION_DEBUG
#define LUISA_RENDER_DIFFERENTIATION_DEBUG_2
#define LUISA_RENDER_USE_BP_TIMES_NORMALIZATION

Differentiation::Differentiation(Pipeline &pipeline, Stream &stream) noexcept
    : _pipeline{pipeline},
      _gradient_buffer_size{constant_parameter_gradient_buffer_size},
      _param_buffer_size{constant_parameter_buffer_capacity * 4u},
      _counter_size{constant_parameter_counter_size},_is_dirty(false)
    {

    instance2offset.emplace(_pipeline.create<Buffer<uint>>(256u)->view());

    _constant_params.reserve(constant_parameter_buffer_capacity);
    _constant_ranges.reserve(constant_parameter_buffer_capacity);

    using namespace compute;
    Kernel1D clear_uint_buffer = [](BufferUInt gradients) noexcept {
        gradients.write(dispatch_x(), 0u);
    };
    Kernel1D clear_float_buffer = [](BufferFloat gradients) noexcept {
        gradients.write(dispatch_x(), 0.f);
    };
    _clear_uint_buffer = _pipeline.device().compile(clear_uint_buffer);
    _clear_float_buffer = _pipeline.device().compile(clear_float_buffer);

    Kernel1D accumulate_grad_const_kernel = [](BufferFloat gradients, BufferFloat param_gradients, BufferUInt counter) noexcept {
        auto thread = dispatch_x();
        auto counter_offset = thread * gradiant_collision_avoidance_block_size;
        auto grad_offset = 4u * counter_offset;
        auto grad = def(make_float4());
        auto count = def(0u);
        for (auto i = 0u; i < gradiant_collision_avoidance_block_size; i++) {
            auto x = gradients.read(grad_offset + i * 4u + 0u);
            auto y = gradients.read(grad_offset + i * 4u + 1u);
            auto z = gradients.read(grad_offset + i * 4u + 2u);
            auto w = gradients.read(grad_offset + i * 4u + 3u);
            grad += make_float4(x, y, z, w);
            count += counter.read(counter_offset + i);
        }
#ifdef LUISA_RENDER_USE_BP_TIMES_NORMALIZATION
        grad /= Float(max(count, 1u));
#endif
        // device_log("grad accumulated: ({}, {}, {})", grad[0u], grad[1u], grad[2u]);

        auto param_offset = thread * 4u;
        param_gradients.write(param_offset + 0u, grad.x);
        param_gradients.write(param_offset + 1u, grad.y);
        param_gradients.write(param_offset + 2u, grad.z);
        param_gradients.write(param_offset + 3u, grad.w);
    };
    _accumulate_grad_const = _pipeline.device().compile(accumulate_grad_const_kernel);

    Kernel1D accumulate_grad_tex_kernel = [](BufferFloat gradients, UInt grad_offset,
                                             BufferUInt counter, UInt counter_offset,
                                             BufferFloat param_gradients, UInt param_offset,
                                             UInt channels) noexcept {
        auto index = dispatch_x();
        auto grad = gradients.read(grad_offset + index);
        auto count = counter.read(counter_offset + index / channels);
#ifdef LUISA_RENDER_USE_BP_TIMES_NORMALIZATION
        grad /= Float(max(count, 1u));
#endif
        param_gradients.write(param_offset + index, grad);
    };
    _accumulate_grad_tex = _pipeline.device().compile(accumulate_grad_tex_kernel);

    Kernel1D accumulate_grad_geom_kernel = [](BufferFloat gradients, UInt grad_offset,
                                             BufferUInt counter, UInt counter_offset,
                                             BufferFloat param_gradients, UInt param_offset) noexcept {
        auto index = dispatch_x();
        auto grad = gradients.read(grad_offset + index);
        auto count = counter.read(counter_offset + index);
        param_gradients.write(param_offset + index, grad);
    };
    _accumulate_grad_geom = _pipeline.device().compile(accumulate_grad_geom_kernel);

    stream << _clear_uint_buffer(*instance2offset).dispatch(256);

}

Differentiation::ConstantParameter Differentiation::parameter(float4 x, uint channels, float2 range) noexcept {
    auto index = static_cast<uint>(_constant_params.size());
    _constant_params.emplace_back(x);
    _constant_ranges.emplace_back(range);
    return {index, channels};
}

Differentiation::ConstantParameter Differentiation::parameter(float x, float2 range) noexcept {
    return parameter(make_float4(x, 0.f, 0.f, 0.f), 1u, range);
}

Differentiation::ConstantParameter Differentiation::parameter(float2 x, float2 range) noexcept {
    return parameter(make_float4(x, 0.f, 0.f), 2u, range);
}

Differentiation::ConstantParameter Differentiation::parameter(float3 x, float2 range) noexcept {
    return parameter(make_float4(x, 0.f), 3u, range);
}
Differentiation::ConstantParameter Differentiation::parameter(float4 x, float2 range) noexcept {
    return parameter(x, 4u, range);
}

Differentiation::TexturedParameter Differentiation::parameter(const Image<float> &image, TextureSampler s, float2 range) noexcept {
    auto nc = compute::pixel_format_channel_count(image.format());
    auto pixel_count = image.size().x * image.size().y;
    auto param_count = pixel_count * nc;
    auto grad_offset = _gradient_buffer_size;
    auto param_offset = _param_buffer_size;
    auto counter_offset = _counter_size;
    auto param_index = static_cast<uint>(_textured_params.size());
    _counter_size = (_counter_size + pixel_count + 3u) & ~0b11u;
    _param_buffer_size = (_param_buffer_size + param_count + 3u) & ~0b11u;
    _gradient_buffer_size = (_gradient_buffer_size + param_count + 3u) & ~0b11u;
    return _textured_params.emplace_back(param_index, image, s, grad_offset, param_offset, counter_offset, range);
}

void Differentiation::materialize(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(!_grad_buffer, "Differentiation already materialized.");
    _param_buffer_size = _gradient_buffer_size - constant_parameter_gradient_buffer_size + constant_parameter_buffer_capacity * 4u;
    _param_buffer.emplace(pipeline().create<Buffer<float>>(std::max(_param_buffer_size, 1u))->view());
    _param_range_buffer.emplace(pipeline().create<Buffer<float2>>(std::max(_param_buffer_size, 1u))->view());
    _param_grad_buffer.emplace(pipeline().create<Buffer<float>>(std::max(_param_buffer_size, 1u))->view());
    _grad_buffer.emplace(pipeline().create<Buffer<float>>(std::max(_gradient_buffer_size, 1u))->view());
    _counter.emplace(pipeline().create<Buffer<uint>>(std::max(_counter_size, 1u))->view());
    clear_gradients(command_buffer);


    if (auto n = _constant_params.size()) {
        Kernel1D constant_params_range_kernel = [](BufferFloat2 params_range_buffer, BufferFloat2 ranges) noexcept {
            auto index = dispatch_x();
            auto range = ranges.read(index);
            index *= 4u;
            params_range_buffer.write(index + 0u, range);
            params_range_buffer.write(index + 1u, range);
            params_range_buffer.write(index + 2u, range);
            params_range_buffer.write(index + 3u, range);
        };
        auto constant_params_range_shader = pipeline().device().compile(constant_params_range_kernel);
        Buffer<float2> ranges = pipeline().device().create<Buffer<float2>>(n);

        command_buffer << _param_buffer->subview(0u, 4u * n).copy_from(_constant_params.data())
                       << ranges.view(0u, n).copy_from(_constant_ranges.data())
                       << constant_params_range_shader(*_param_range_buffer, ranges).dispatch(n)
                       << synchronize();
    }

    Kernel1D textured_params_range_kernel = [](BufferFloat2 params_range_buffer, Float2 range, UInt offset) noexcept {
        params_range_buffer.write(offset + dispatch_x(), range);
    };
    auto textured_params_range_shader = pipeline().device().compile(textured_params_range_kernel);

    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto param_offset = p.param_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        command_buffer << image.copy_to(_param_buffer->subview(param_offset, length))
                       << textured_params_range_shader(*_param_range_buffer, p.range(), param_offset).dispatch(length);
    }

    //Todo: add buffer parameters(mesh vertex or sth) to _param_buffer
    for (auto &&p : _geometry_params) {
        auto buffer = p.buffer();
        auto param_offset = p.param_offset();
        auto length = buffer.size();
        command_buffer << _param_buffer->subview(param_offset, length*8).copy_from(buffer.as<float>());
    }
    command_buffer << synchronize();
    _optimizer->initialize(command_buffer, _param_buffer_size, *_param_buffer, *_param_grad_buffer, *_param_range_buffer);
}

void Differentiation::clear_gradients(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    if (auto n = _gradient_buffer_size) {
        command_buffer << _clear_float_buffer(*_grad_buffer).dispatch(_grad_buffer->size());
    }
    if (auto n = _counter_size) {
        command_buffer << _clear_uint_buffer(*_counter).dispatch(_counter->size());
    }
}

void Differentiation::accum_gradients(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");

    // accumulate constant parameters
    if (auto n = _constant_params.size()) {
        command_buffer << _accumulate_grad_const(*_grad_buffer, *_param_grad_buffer, *_counter).dispatch(n);

#ifdef LUISA_RENDER_DIFFERENTIATION_DEBUG
        luisa::vector<float4> params_before(n);
        luisa::vector<uint> counter(n * gradiant_collision_avoidance_block_size);
        luisa::vector<float4> params_gradients(n * gradiant_collision_avoidance_block_size);
        luisa::vector<float4> params_gradients_calculated(n);
        luisa::vector<float2> params_ranges(n * 4u);
        command_buffer << _param_buffer->subview(0u, n * 4u).copy_to(params_before.data())
                       << _counter->subview(0u, n * gradiant_collision_avoidance_block_size).copy_to(counter.data())
                       << _grad_buffer->subview(0u, n * 4u * gradiant_collision_avoidance_block_size)
                              .copy_to(params_gradients.data())
                       << _param_grad_buffer->subview(0u, n * 4u).copy_to(params_gradients_calculated.data())
                       << _param_range_buffer->subview(0u, n * 4u).copy_to(params_ranges.data())
                       << synchronize();

        // DEBUG: print constant parameters
        for (auto i = 0u; i < n; i++) {
            auto p0 = params_before[i];
            auto grad = make_float4();
            auto count_uint = 0u;
            for (auto g = 0u; g < gradiant_collision_avoidance_block_size; g++) {
                auto index = i * gradiant_collision_avoidance_block_size + g;
                grad += params_gradients[index];
                count_uint += counter[index];
            }
            auto count = float(std::max(count_uint, 1u));
            grad /= count;
            auto grad_calculated = params_gradients_calculated[i];
            auto range_x = params_ranges[i * 4u + 0u];
            auto range_y = params_ranges[i * 4u + 1u];
            auto range_z = params_ranges[i * 4u + 2u];
            auto range_w = params_ranges[i * 4u + 3u];
            LUISA_INFO(
                "\nParam #{}: \n"
                "({}, {}, {}, {}) - {} * ({}, {}, {}, {})\n"
                "?= ({}, {}, {}, {}) - {} * ({}, {}, {}, {})\n"
                "count = {}, range = ([{}, {}], [{}, {}], [{}, {}], [{}, {}])",
                i, p0.x, p0.y, p0.z, p0.w,
                "alpha", grad.x, grad.y, grad.z, grad.w,
                p0.x, p0.y, p0.z, p0.w,
                "alpha", grad_calculated.x, grad_calculated.y, grad_calculated.z, grad_calculated.w,
                count, range_x.x, range_x.y, range_y.x, range_y.y, range_z.x, range_z.y, range_w.x, range_w.y);
        }
#endif
    }

    // accumulate textured parameters
    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto param_offset = p.param_offset();
        auto counter_offset = p.counter_offset();
        auto grad_offset = p.gradient_buffer_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        command_buffer << _accumulate_grad_tex(
                              *_grad_buffer, grad_offset,
                              *_counter, counter_offset,
                              *_param_grad_buffer, param_offset,
                              channels)
                              .dispatch(length);
    }

    // accumulate geometry parameters
    for (auto &&p : _geometry_params) {
        auto buffer = p.buffer();
        auto param_offset = p.param_offset();
        auto counter_offset = p.counter_offset();
        auto grad_offset = p.gradient_buffer_offset();
        auto length = buffer.size_bytes();
        command_buffer << _accumulate_grad_geom(
                              *_grad_buffer, grad_offset,
                              *_counter, counter_offset,
                              *_param_grad_buffer, param_offset
                              ).dispatch(length);
    }
}



void Differentiation::apply_gradients(CommandBuffer &command_buffer) noexcept {
    _optimizer->step(command_buffer);

#ifdef LUISA_RENDER_DIFFERENTIATION_DEBUG
    if (auto n = _constant_params.size()) {
        luisa::vector<float4> params_after(n);
        command_buffer << _param_buffer->subview(0u, n * 4u).copy_to(params_after.data())
                       << synchronize();

        // DEBUG: print constant parameters
        for (auto i = 0u; i < n; i++) {
            auto p0 = params_after[i];
            LUISA_INFO(
                "\nParam #{}: ({}, {}, {}, {})",
                i, p0.x, p0.y, p0.z, p0.w);
        }
    }
#endif

    // apply textured parameters
    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto param_offset = p.param_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        command_buffer << image.copy_from(_param_buffer->subview(param_offset, length));
    }

    // apply geometry parameters
    for (auto &&p : _geometry_params) {
        auto param_offset = p.param_offset();
        auto buffer_id = p.buffer_id();
        auto [buffer_view, bindlessbuffer_id] = _pipeline.bindless_arena_buffer<Vertex>(buffer_id);
        auto length = buffer_view.size();
        command_buffer << buffer_view.copy_from(_param_buffer->subview(param_offset, length).as<Vertex>());
    }
    command_buffer << synchronize();
    _is_dirty = true;
}

Float4 Differentiation::decode(const Differentiation::ConstantParameter &param) const noexcept {
    auto x = _param_buffer.value()->read(param.index() * 4u + 0u);
    auto y = _param_buffer.value()->read(param.index() * 4u + 1u);
    auto z = _param_buffer.value()->read(param.index() * 4u + 2u);
    auto w = _param_buffer.value()->read(param.index() * 4u + 3u);
    return make_float4(x, y, z, w);
}

void Differentiation::accumulate(const Differentiation::ConstantParameter &param, Expr<float4> grad,
                                 Expr<uint> slot_seed) const noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    auto slots = (slot_seed ^ pcg4d(as<uint4>(grad))) & gradiant_collision_avoidance_bit_and;
    for (auto i = 0u; i < param.channels(); i++) {
        auto grad_offset = (param.index() * gradiant_collision_avoidance_block_size + slots[i]) * 4u + i;
        _grad_buffer.value()->atomic(grad_offset).fetch_add(grad[i]);
    }
    auto counter_offset = param.index() * gradiant_collision_avoidance_block_size + slots[0];
    _counter.value()->atomic(counter_offset).fetch_add(1u);
}

void Differentiation::accumulate(const Differentiation::TexturedParameter &param, Expr<float2> p,
                                 Expr<float4> grad) const noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    using namespace compute;
    auto map_uv = [s = param.sampler()](Expr<float2> uv) noexcept {
        switch (s.address()) {
            case TextureSampler::Address::EDGE:
                return clamp(uv, 0.f, one_minus_epsilon);
            case TextureSampler::Address::REPEAT:
                return fract(uv);
            case TextureSampler::Address::MIRROR: {
                auto t = floor(uv);
                auto frac = uv - t;
                return ite(make_int2(t) % 2 == 0, frac, 1.f - frac);
            }
            case TextureSampler::Address::ZERO:
                return def(uv);
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture address mode.");
    };
    auto write_grad = [&param, this](Expr<float2> uv, Expr<float4> grad) noexcept {
        $if(all(uv >= 0.f && uv < 1.f)) {
            auto size = param.image().size();
            auto coord = clamp(make_uint2(uv * make_float2(size)), 0u, size - 1u);
            auto pixel_id = coord.y * size.x + coord.x;
            auto nc = pixel_format_channel_count(param.image().format());
            auto grad_offset = param.gradient_buffer_offset() + pixel_id * nc;
            auto counter_offset = param.counter_offset() + pixel_id;
            for (auto i = 0u; i < nc; i++) {
                _grad_buffer.value()->atomic(grad_offset + i).fetch_add(grad[i]);
            }
            _counter.value()->atomic(counter_offset).fetch_add(1u);
        };
    };
    write_grad(map_uv(p), grad);
}

void Differentiation::add_geom_gradients(Float3 grad_v, Float3 grad_n, Float3 weight, UInt inst_id, UInt triangle_id) noexcept {
    
    UInt gradient_offset = instance2offset.value()->read(inst_id);
    $if(gradient_offset == 0u) { 
        return; 
    }; 
    UInt real_grad_offset = gradient_offset-1;
    //device_log("instance_id is {} grad {}",inst_id, real_grad_offset);

    auto instance = pipeline().geometry()->instance(inst_id);
    auto triangle = pipeline().geometry()->triangle(instance, triangle_id);
    //device_log("triangle {} {} {}",triangle.i0, triangle.i1,triangle.i2);

    _grad_buffer.value()->atomic(real_grad_offset + triangle.i0 * 8 + 0).fetch_add(grad_v[0] * weight[0]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i0 * 8 + 1).fetch_add(grad_v[1] * weight[0]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i0 * 8 + 2).fetch_add(grad_v[2] * weight[0]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i0 * 8 + 3).fetch_add(grad_n[0] * weight[0]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i0 * 8 + 4).fetch_add(grad_n[1] * weight[0]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i0 * 8 + 5).fetch_add(grad_n[2] * weight[0]);
    
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i1 * 8 + 0).fetch_add(grad_v[0] * weight[1]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i1 * 8 + 1).fetch_add(grad_v[1] * weight[1]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i1 * 8 + 2).fetch_add(grad_v[2] * weight[1]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i1 * 8 + 3).fetch_add(grad_n[0] * weight[1]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i1 * 8 + 4).fetch_add(grad_n[1] * weight[1]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i1 * 8 + 5).fetch_add(grad_n[2] * weight[1]);
    
    //device_log("weight2 {}, grad_v {}, grad_offset {} triangle i2 {}",weight[2], grad_v, real_grad_offset, triangle.i2);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i2 * 8 + 0).fetch_add(grad_v[0] * weight[2]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i2 * 8 + 1).fetch_add(grad_v[1] * weight[2]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i2 * 8 + 2).fetch_add(grad_v[2] * weight[2]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i2 * 8 + 3).fetch_add(grad_n[0] * weight[2]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i2 * 8 + 4).fetch_add(grad_n[1] * weight[2]);
    _grad_buffer.value()->atomic(real_grad_offset + triangle.i2 * 8 + 5).fetch_add(grad_n[2] * weight[2]);
    
    // for(uint i=0;i<6;i++)
    // {
    //     _counter.value()->atomic(real_grad_offset + triangle.i0 * 8 + i).fetch_add(UInt(1));
    //     _counter.value()->atomic(real_grad_offset + triangle.i1 * 8 + i).fetch_add(UInt(1));
    //     _counter.value()->atomic(real_grad_offset + triangle.i2 * 8 + i).fetch_add(UInt(1));
    // }
}

void Differentiation::step(CommandBuffer &command_buffer) noexcept {
    //accum_gradients(command_buffer);
    apply_gradients(command_buffer);
    clear_gradients(command_buffer);
}

void Differentiation::dump(CommandBuffer &command_buffer, const std::filesystem::path &folder) const noexcept {
    // FIXME : several channels will be 0 when grads explode
    for (auto i = 0u; i < _textured_params.size(); i++) {
        auto param = _textured_params[i];
        auto image = param.image().view();
        auto size = image.size();
        auto channels = compute::pixel_storage_channel_count(image.storage());
        luisa::vector<float> pixels(size.x * size.y * channels);
        command_buffer << image.copy_to(pixels.data()) << compute::synchronize();
        auto file_name = folder / luisa::format("dump-{:05}.exr", i);
        SaveEXR(pixels.data(), static_cast<int>(size.x), static_cast<int>(size.y),
                static_cast<int>(channels), false, file_name.string().c_str(), nullptr);
    }
#ifdef LUISA_RENDER_DIFFERENTIATION_DEBUG_2
    if (auto n = _constant_params.size()) {
        luisa::vector<float4> params_after(n);
        command_buffer << _param_buffer->subview(0u, n * 4u).copy_to(params_after.data())
                       << synchronize();

        // DEBUG: print constant parameters
        for (auto i = 0u; i < n; i++) {
            auto p0 = params_after[i];
            LUISA_INFO(
                "\nParam #{}: ({}, {}, {}, {})",
                i, p0.x, p0.y, p0.z, p0.w);
        }
    }
#endif
}

void Differentiation::register_optimizer(Optimizer::Instance *optimizer) noexcept {
    _optimizer = optimizer;
}

void Differentiation::register_geometry_parameter(CommandBuffer &command_buffer, Geometry::MeshData& mesh, Accel& accel, uint instance_id) noexcept {
    
    //LUISA_INFO("register geometry parameter start");
    auto param_offset = _param_buffer_size;
    auto counter_offset = _counter_size;
    auto grad_offset = _gradient_buffer_size;
    auto param_index = static_cast<uint>(_geometry_params.size());
    //auto channels = 3u;
    //uint buffer_id = mesh.geometry_buffer_id_base;
    auto buffer_view = mesh.vertices;
    
    auto tri_buffer_view = mesh.triangles;
    auto triangle_buffer_new = _pipeline.create<Buffer<int>>(tri_buffer_view.size() * 3);
    auto triangle_buffer_new_view = triangle_buffer_new->view();

    Kernel1D write_tri = [&]() noexcept {
        auto x = dispatch_x();
        auto tri = tri_buffer_view->read(x);
        triangle_buffer_new_view->write(x * 3, (Int)(tri.i0));
        triangle_buffer_new_view->write(x * 3 + 1, (Int)(tri.i1));
        triangle_buffer_new_view->write(x * 3 + 2, (Int)(tri.i2));
    };
    auto write_tri_shader = pipeline().device().compile(write_tri);
    command_buffer << write_tri_shader().dispatch(tri_buffer_view.size()) << synchronize();

    auto length = buffer_view.size() * 8;
    _counter_size = (_counter_size + length + 8u) & ~0b11u;
    _param_buffer_size = (_param_buffer_size + length + 8u) & ~0b11u;
    _gradient_buffer_size = (_gradient_buffer_size + length + 8u) & ~0b11u;
    _geometry_params.emplace_back(param_index, instance_id, grad_offset, param_offset, counter_offset, buffer_view, length, 0u, mesh.resource, triangle_buffer_new_view);
    Kernel1D write_i2o = [&](UInt instance_id, UInt grad_offset) noexcept {
        instance2offset.value()->write(instance_id, grad_offset+1u);
    };
    auto write_i2o_shader = pipeline().device().compile(write_i2o);
    command_buffer << write_i2o_shader(instance_id, grad_offset).dispatch(1u) << synchronize();
    //LUISA_INFO("register geometry parameter finished");
}

void Differentiation::update_parameter_from_external(Stream &stream, luisa::vector<uint> &constants_id, luisa::vector<float4> &constants, luisa::vector<uint> &textures_id, 
    luisa::vector<Buffer<float4>> &textures, luisa::vector<uint> &geoms_id, luisa::vector<Buffer<float>> &geoms) noexcept {

    // apply texture parameters
    for (auto i = 0u; i < textures_id.size(); i++) {
        auto image = _textured_params[textures_id[i]].image().view();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        stream << image.copy_from(textures[textures_id[i]]);
    }

    //LUISA_INFO("working on update paramaters {}",geoms_id.size());
    // apply geometry parameters
    for (auto i=0;i<geoms_id.size();i++) {
        auto &&p = _geometry_params[geoms_id[i]];
        auto param_offset = p.param_offset();
        auto buffer_view = p.buffer();
        //auto [buffer_view, bindlessbuffer_id] = _pipeline.bindless_arena_buffer<Vertex>(buffer_id);
        auto length = buffer_view.size();
        //LUISA_INFO("here length is {}, size is {}, as<vertex> size is {}",length,geoms[geoms_id[i]].view().size(),geoms[geoms_id[i]].view().as<Vertex>().size());
        stream << buffer_view.copy_from(geoms[geoms_id[i]].view().as<Vertex>()) << synchronize() << p.mesh()->build() << synchronize();
        _is_dirty = true;
    }
}

std::tuple<luisa::vector<void *>, luisa::vector<uint>, luisa::vector<void *>, luisa::vector<uint>> Differentiation::get_parameter_from_external(Stream &stream, luisa::vector<uint> &constants_id, luisa::vector<uint> &textures_id, luisa::vector<uint> &geoms_id) noexcept {
    luisa::vector<void*> geom_param{};
    luisa::vector<uint> geom_size{};
    luisa::vector<void*> tri_param{};
    luisa::vector<uint> tri_size{};
    // apply geometry parameters
    for (auto i: geoms_id) {
        auto p = _geometry_params[i];
        auto param_offset = p.param_offset();
        auto buffer_view = p.buffer();
        auto tri_buffer_view = p.tri_buffer();
        auto length = buffer_view.size();
        auto tri_length = tri_buffer_view.size();
        geom_param.push_back(buffer_view.as<float>().native_handle());
        geom_size.push_back(length*8u);
        tri_param.push_back(tri_buffer_view.native_handle());
        tri_size.push_back(tri_length);
    }
    return std::make_tuple(geom_param, geom_size, tri_param, tri_size);
}
 
std::tuple<luisa::vector<void *>, luisa::vector<void *>> Differentiation::get_gradients(Stream &stream) {
    luisa::vector<void*> texture_res{};
    luisa::vector<void*> geom_res{};
    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto param_offset = p.param_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        LUISA_INFO("{} {} {}", image.size().x , image.size().y , channels);
        auto tex_grad_buf_view = _param_grad_buffer->subview(param_offset, length);
        texture_res.push_back(reinterpret_cast<void*>(reinterpret_cast<uint64_t>(tex_grad_buf_view.native_handle())+tex_grad_buf_view.offset_bytes()));
        // auto tex_buf_view = _param_buffer->subview(param_offset, length);
        // geom_res.push_back(reinterpret_cast<void*>(reinterpret_cast<uint64_t>(tex_buf_view.native_handle())+tex_buf_view.offset_bytes()));
        // auto size = image.size();
        // luisa::vector<float> pixels(size.x * size.y * channels);
        // stream << image.copy_to(pixels.data()) << compute::synch ronize();
        // // string file_name = "D:/dump-1.exr";
        // //save_image(pixels.data(), static_cast<int>(size.x), static_cast<int>(size.y),
        //   //      static_cast<int>(channels), false, file_name.c_str(), nullptr);
          
        // auto film_path = _pipeline.camera(0)->node()->file();
        // save_image(film_path, reinterpret_cast<const float *>(pixels.data()), uint2(size.x, size.y));
    }
    // apply geometry parameters
    for (auto &&p : _geometry_params) {
        auto grad_offset = p.gradient_buffer_offset();
        auto buffer_view = p.buffer();
        auto length = buffer_view.size();
        //LUISA_INFO("here length is {}",length);
        auto geom_grad_buf_view = _grad_buffer->subview(grad_offset, length*8);
        geom_res.push_back(reinterpret_cast<void*>(reinterpret_cast<uint64_t>(geom_grad_buf_view.native_handle())+geom_grad_buf_view.offset_bytes()));
    }
    return std::make_tuple(texture_res, geom_res);
}

void Differentiation::set_parameter(CommandBuffer &command_buffer, const ConstantParameter &param, BufferView<float> value) noexcept {
    auto param_offset = param.index();
    auto length = value.size();
    command_buffer << _param_buffer->subview(param_offset, length).copy_from(value) << synchronize();
}
void Differentiation::set_parameter(CommandBuffer &command_buffer, const TexturedParameter &param, BufferView<float> value) noexcept {
    auto param_offset = param.param_offset();
    auto length = value.size();
    command_buffer << _param_buffer->subview(param_offset, length).copy_from(value) << synchronize();
}
void Differentiation::set_parameter(CommandBuffer &command_buffer, const GeometryParameter &param, BufferView<float> value) noexcept {
    auto param_offset = param.param_offset();
    auto length = value.size();
    command_buffer << _param_buffer->subview(param_offset, length).copy_from(value) << synchronize();
}

}// namespace luisa::render
