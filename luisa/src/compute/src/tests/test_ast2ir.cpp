#include <iostream>
#include <chrono>
#include <numeric>
#include <fstream>

#include <luisa/core/clock.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/ast/interface.h>
#include <luisa/dsl/syntax.h>

#include <luisa/ir/ast2ir.h>

using namespace luisa;
using namespace luisa::compute;

struct Test {
    int3 something;
    float a;
};

LUISA_STRUCT(Test, something, a) {};

int main(int argc, char *argv[]) {

    constexpr auto f = 10;

    luisa::log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    auto buffer = device.create_buffer<float4>(1024u);
    auto float_buffer = device.create_buffer<float>(1024u);

    Callable c1 = [&](UInt a) noexcept {
        return buffer->read(a + thread_x());// captures buffer
    };

    Callable c2 = [&](UInt b) noexcept {
        // captures buffer (propagated from c1) and float_buffer
        return c1(b) + make_float4(float_buffer->read(b));
    };

    Kernel1D k1 = [&] {
        // captures buffer and float_buffer (propagated from c2)
        auto v = c2(dispatch_x());
        float_buffer->write(dispatch_x(), v.x + v.y + v.z);
    };

    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    Callable add_mul = [&](Var<int> a, Var<int> b) noexcept {
        return compose(cast<int>(float_buffer->read(a + b)), a * b);
    };

    Callable callable = [&](Var<int> a, Var<int> b, Var<float> c) noexcept {
        Constant int_consts = const_vector;
        return cast<float>(a) + int_consts[b].cast<float>() * c;
    };

    // binding to template lambdas
    Callable<int(int, int)> add = [&]<typename T>(Var<T> a, Var<T> b) noexcept {
        return cast<int>(c1(cast<uint>(a + b)).x);
    };

    Clock clock;
    Constant float_consts = {1.0f, 2.0f};
    Constant int_consts = const_vector;

    Kernel1D<Buffer<float>, uint> kernel_def = [&](BufferVar<float> buffer_float, Var<uint> count) noexcept -> void {
        for (auto n = 0u; n < 1u; n++) {
            Shared<float4> shared_floats{16};

            count += 1u;

            Constant float_consts = {1.0f, 2.0f};
            auto ff = float_consts.read(0);

            Var mat = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
            Var mat2 = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
            Var mat3 = mat * mat2;
            Var mat4 = mat3 * make_float2(2.f);

            Var v_int = 10;
            Var t = make_int3(1, 2, 3);
            Var vv = ite(t == 10, 1, 2);

            Var vvv = min(vv, 10);
            Var xxx = make_uint4(5);
            Var vvvv = min(xxx, 1u);

            Var am = add_mul(v_int, v_int);
            Var a_copy = am.get<0>();
            Var m_copy = am.get<1>();

            loop([] {
                if_(true, break_);
            });

            for (auto v : dynamic_range(v_int)) {
                v_int += v;
            }

            Var v_int_add_one = add(v_int, 1);
            Var vv_int = int_consts[v_int];
            Var v_float = buffer_float.read(count + thread_id().x);
            Var vv_float = float_consts[0];
            Var call_ret = callable(10, v_int, v_float);

            Var v_float_copy = v_float;

            Var z = -1 + v_int * v_float + 1.0f;
            z += 1;
            Var v_vec = float3{1.0f};
            Var v2 = float3{2.0f} - v_vec * 2.0f;
            v2 *= 5.0f + v_float;

            Var<float2> w{cast<float>(v_int), v_float};
            w *= float2{1.2f};

            if_(v_int == v_int, [] {
                Var a = 0.0f;
            }).else_([] {
                Var c = 2.0f;
            });

            switch_(123)
                .case_(1, [] {

                })
                .case_(2, [] {

                })
                .default_([] {

                });

            Var x = w.x;

            Var<int3> s;
            Var<Test> vvt{s, v_float_copy};
            Var<Test> vt{vvt};

            Var vt_copy = vt;
            Var c = 0.5f + vt.a * 1.0f;

            Var vec4 = buffer->read(10);           // indexing into captured buffer (with literal)
            Var another_vec4 = buffer->read(v_int);// indexing into captured buffer (with Var)
            buffer->write(v_int + 1, float4(123.0f));
        }
    };
    LUISA_INFO("Kernel definition parsed in {} ms.", clock.toc());

    // test ir
    clock.tic();
    auto ir = AST2IR::build_kernel(kernel_def.function()->function());
    LUISA_INFO("AST2IR done in {} ms.", clock.toc());

    // dump json
    {
        clock.tic();
        auto dump = ir::luisa_compute_ir_dump_json(&ir->get()->module);
        LUISA_INFO("IR json dump done in {} ms.", clock.toc());
        std::ofstream out{"test_ast2ir.json"};
        out << luisa::string_view{reinterpret_cast<const char *>(dump.ptr), dump.len};
    }

    // dump binary
    {
        clock.tic();
        auto binary = ir::luisa_compute_ir_dump_binary(&ir->get()->module);
        LUISA_INFO("IR binary dump done in {} ms.", clock.toc());

        std::ofstream bin_out{"test_ast2ir.bin", std::ios::binary};
        bin_out.write(reinterpret_cast<const char *>(binary.ptr),
                      static_cast<std::streamsize>(binary.len));
    }
}

