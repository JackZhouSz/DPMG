//
// Created by Mike Smith on 2022/3/9.
//

#pragma once

#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/shader.h>
#include <base/geometry.h>
#include <util/command_buffer.h>
#include <util/vertex.h>
#include <base/optimizer.h>

namespace luisa::render {

using compute::Buffer;
using compute::BufferView;
using compute::Expr;
using compute::Float4;
using compute::Image;
using compute::ImageView;
using compute::Shader1D;
using compute::Shader2D;
using TextureSampler = compute::Sampler;

class Pipeline;

class Differentiation {

private:
    friend class Optimizer;

    static constexpr uint gradiant_collision_avoidance_block_bits = 9u;
    static constexpr uint gradiant_collision_avoidance_block_size = 1u << gradiant_collision_avoidance_block_bits;// 512u
    static constexpr uint gradiant_collision_avoidance_bit_and = gradiant_collision_avoidance_block_size - 1u;    // 511u

    static constexpr uint constant_parameter_buffer_capacity = 4096u;
    static constexpr uint constant_parameter_counter_size =
        constant_parameter_buffer_capacity *
        gradiant_collision_avoidance_block_size;
    static constexpr uint constant_parameter_gradient_buffer_size =
        constant_parameter_buffer_capacity * 4u *
        gradiant_collision_avoidance_block_size;

public:
    class ConstantParameter {

    private:
        uint _index;
        uint _channels;

    public:
        ConstantParameter(uint index, uint channels) noexcept
            : _index{index}, _channels{channels} {}
        [[nodiscard]] auto index() const noexcept { return _index; }
        [[nodiscard]] auto channels() const noexcept { return _channels; }
        [[nodiscard]] auto identifier() const noexcept { return luisa::format("diffconst({})", _index); }
    };

    class GeometryParameter {
        uint _index;
        uint _instance_id;
        uint _grad_offset;
        uint _param_offset;
        uint _counter_offset;
        float2 _range;
        BufferView<Vertex> _buffer_view;
        BufferView<int> _tri_buffer_view;
        uint _length;
        uint _buffer_id;
        Mesh *_mesh;
    public:
        GeometryParameter(uint index, uint instance_id, uint grad_offset, uint param_offset,
                          uint counter_offset, BufferView<Vertex> buffer_view, uint length, uint buffer_id, Mesh *mesh, BufferView<int> tri_buffer_view) noexcept
            : _index(index), _instance_id{instance_id}, _grad_offset{grad_offset}, _param_offset{param_offset},
              _counter_offset{counter_offset}, _buffer_view{buffer_view}, _length(length), _buffer_id(buffer_id), _mesh(mesh), _tri_buffer_view(tri_buffer_view) {}
        [[nodiscard]] auto index() const noexcept { return _index; }
        [[nodiscard]] auto buffer() const noexcept { return _buffer_view; }
        [[nodiscard]] auto tri_buffer() const noexcept { return _tri_buffer_view; }
        [[nodiscard]] auto buffer_id() const noexcept { return _buffer_id; }
        [[nodiscard]] auto instance_id() const noexcept { return _instance_id; }
        [[nodiscard]] auto gradient_buffer_offset() const noexcept { return _grad_offset; }
        [[nodiscard]] auto param_offset() const noexcept { return _param_offset; }
        [[nodiscard]] auto counter_offset() const noexcept { return _counter_offset; }
        [[nodiscard]] auto mesh() const noexcept { return _mesh; }
        [[nodiscard]] auto identifier() const noexcept { return luisa::format("diff_geom({})", _index); }
    };

    class TexturedParameter {

    private:
        const Image<float> &_image;
        uint _index;
        TextureSampler _sampler;
        uint _grad_offset;
        uint _param_offset;
        uint _counter_offset;
        float2 _range;

    public:
        TexturedParameter(uint index,
                          const Image<float> &image, TextureSampler sampler,
                          uint grad_offset, uint param_offset,
                          uint counter_offset, float2 range) noexcept
            : _image{image}, _index{index}, _sampler{sampler},
              _grad_offset{grad_offset}, _param_offset{param_offset},
              _counter_offset{counter_offset}, _range{range} {}
        [[nodiscard]] auto &image() const noexcept { return _image; }
        [[nodiscard]] auto index() const noexcept { return _index; }
        [[nodiscard]] auto sampler() const noexcept { return _sampler; }
        [[nodiscard]] auto range() const noexcept { return _range; }
        [[nodiscard]] auto gradient_buffer_offset() const noexcept { return _grad_offset; }
        [[nodiscard]] auto param_offset() const noexcept { return _param_offset; }
        [[nodiscard]] auto counter_offset() const noexcept { return _counter_offset; }
        [[nodiscard]] auto identifier() const noexcept { return luisa::format("difftex({})", _index); }
    };

private:
    Pipeline &_pipeline;

    Optimizer::Instance *_optimizer;

    luisa::vector<float4> _constant_params;
    luisa::vector<float2> _constant_ranges;
    luisa::vector<TexturedParameter> _textured_params;
    
    luisa::vector<GeometryParameter> _geometry_params;

    uint _gradient_buffer_size;
    luisa::optional<BufferView<float>> _grad_buffer;

    uint _param_buffer_size;
    luisa::optional<BufferView<float>> _param_buffer;
    luisa::optional<BufferView<float2>> _param_range_buffer;
    luisa::optional<BufferView<float>> _param_grad_buffer;

    uint _counter_size;
    luisa::optional<BufferView<uint>> _counter;

    Shader1D<Buffer<uint>> _clear_uint_buffer;
    Shader1D<Buffer<float>> _clear_float_buffer;
    Shader1D<Buffer<float>, Buffer<float>, Buffer<uint>> _accumulate_grad_const;
    Shader1D<Buffer<float>, uint, Buffer<uint>, uint, Buffer<float>, uint, uint> _accumulate_grad_tex;
    Shader1D<Buffer<float>, uint, Buffer<uint>, uint, Buffer<float>, uint> _accumulate_grad_geom;

    luisa::optional<BufferView<uint>> instance2offset;


private:
    auto &pipeline() noexcept { return _pipeline; }

public:
    explicit Differentiation(Pipeline &pipeline, Stream &stream) noexcept;
    void register_optimizer(Optimizer::Instance *optimizer) noexcept;
    void register_geometry_parameter(CommandBuffer &command_buffer, Geometry::MeshData &mesh, Accel &accel, uint instance_id) noexcept;
    [[nodiscard]] ConstantParameter parameter(float x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float2 x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float3 x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float4 x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float4 x, uint channels, float2 range) noexcept;
    [[nodiscard]] TexturedParameter parameter(const Image<float> &image, TextureSampler s, float2 range) noexcept;
    // geom
    [[nodiscard]] luisa::vector<GeometryParameter> geometry_parameters(){return _geometry_params;}
    void materialize(CommandBuffer &command_buffer) noexcept;
    void clear_gradients(CommandBuffer &command_buffer) noexcept;
    void apply_gradients(CommandBuffer &command_buffer) noexcept;
    void accum_gradients(CommandBuffer &command_buffer) noexcept;

    void add_geom_gradients(Float3 grad_v, Float3 grad_n, Float3 weight, UInt inst_id, UInt triangle_id) noexcept;
    /// Apply then clear the gradients
    void step(CommandBuffer &command_buffer) noexcept;
    void dump(CommandBuffer &command_buffer, const std::filesystem::path &folder) const noexcept;
    // check dirty
    bool _is_dirty;
    bool is_dirty(){return _is_dirty;}
    bool clear_dirty(){return _is_dirty = false;}
    void update_parameter_from_external(Stream &stream, luisa::vector<uint> &constants_id, luisa::vector<float4> &constants, luisa::vector<uint> &textures_id, 
    luisa::vector<Buffer<float4>> &textures, luisa::vector<uint> &geoms_id, luisa::vector<Buffer<float>> &geoms) noexcept;

    std::tuple<luisa::vector<void *>, luisa::vector<uint>,luisa::vector<void *>, luisa::vector<uint>> get_parameter_from_external
    (Stream &stream, luisa::vector<uint> &constants_id, luisa::vector<uint> &textures_id, luisa::vector<uint> &geoms_id) noexcept;


public:
    [[nodiscard]] Float4 decode(const ConstantParameter &param) const noexcept;
    void accumulate(const ConstantParameter &param, Expr<float4> grad, Expr<uint> slot_seed) const noexcept;
    void accumulate(const TexturedParameter &param, Expr<float2> p, Expr<float4> grad) const noexcept;
    void accumulate(const GeometryParameter &param, Expr<float2> p, Expr<float4> grad) const noexcept;

    void set_parameter(CommandBuffer &command_buffer, const ConstantParameter &param, BufferView<float> value) noexcept;
    void set_parameter(CommandBuffer &command_buffer, const TexturedParameter &param, BufferView<float> value) noexcept;
    void set_parameter(CommandBuffer &command_buffer, const GeometryParameter &param, BufferView<float> value) noexcept;

    std::tuple<luisa::vector<void*>,luisa::vector<void*>> get_gradients(Stream &stream);
    luisa::span<const Shape *const> _shapes;
};

}// namespace luisa::render
