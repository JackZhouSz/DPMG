 // This file is generated by device_reduce.py
#include <luisa/backends/ext/cuda/lcub/device_reduce.h>
#include "private/lcub_utils.h"
#include "private/dcub/device_reduce.h"

namespace luisa::compute::cuda::lcub{
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html

void DeviceReduce::Sum(size_t &temp_storage_size, BufferView<int32_t>  d_in, BufferView<int32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Sum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Sum(BufferView<int> d_temp_storage, BufferView<int32_t>  d_in, BufferView<int32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Sum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Sum(size_t &temp_storage_size, BufferView<uint32_t>  d_in, BufferView<uint32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Sum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Sum(BufferView<int> d_temp_storage, BufferView<uint32_t>  d_in, BufferView<uint32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Sum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Sum(size_t &temp_storage_size, BufferView<int64_t>  d_in, BufferView<int64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Sum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Sum(BufferView<int> d_temp_storage, BufferView<int64_t>  d_in, BufferView<int64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Sum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Sum(size_t &temp_storage_size, BufferView<uint64_t>  d_in, BufferView<uint64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Sum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Sum(BufferView<int> d_temp_storage, BufferView<uint64_t>  d_in, BufferView<uint64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Sum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Sum(size_t &temp_storage_size, BufferView<float>  d_in, BufferView<float>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Sum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Sum(BufferView<int> d_temp_storage, BufferView<float>  d_in, BufferView<float>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Sum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Sum(size_t &temp_storage_size, BufferView<double>  d_in, BufferView<double>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Sum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Sum(BufferView<int> d_temp_storage, BufferView<double>  d_in, BufferView<double>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Sum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Max(size_t &temp_storage_size, BufferView<int32_t>  d_in, BufferView<int32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Max(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Max(BufferView<int> d_temp_storage, BufferView<int32_t>  d_in, BufferView<int32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Max(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Max(size_t &temp_storage_size, BufferView<uint32_t>  d_in, BufferView<uint32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Max(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Max(BufferView<int> d_temp_storage, BufferView<uint32_t>  d_in, BufferView<uint32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Max(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Max(size_t &temp_storage_size, BufferView<int64_t>  d_in, BufferView<int64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Max(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Max(BufferView<int> d_temp_storage, BufferView<int64_t>  d_in, BufferView<int64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Max(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Max(size_t &temp_storage_size, BufferView<uint64_t>  d_in, BufferView<uint64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Max(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Max(BufferView<int> d_temp_storage, BufferView<uint64_t>  d_in, BufferView<uint64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Max(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Max(size_t &temp_storage_size, BufferView<float>  d_in, BufferView<float>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Max(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Max(BufferView<int> d_temp_storage, BufferView<float>  d_in, BufferView<float>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Max(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Max(size_t &temp_storage_size, BufferView<double>  d_in, BufferView<double>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Max(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Max(BufferView<int> d_temp_storage, BufferView<double>  d_in, BufferView<double>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Max(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Min(size_t &temp_storage_size, BufferView<int32_t>  d_in, BufferView<int32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Min(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Min(BufferView<int> d_temp_storage, BufferView<int32_t>  d_in, BufferView<int32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Min(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Min(size_t &temp_storage_size, BufferView<uint32_t>  d_in, BufferView<uint32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Min(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Min(BufferView<int> d_temp_storage, BufferView<uint32_t>  d_in, BufferView<uint32_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Min(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Min(size_t &temp_storage_size, BufferView<int64_t>  d_in, BufferView<int64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Min(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Min(BufferView<int> d_temp_storage, BufferView<int64_t>  d_in, BufferView<int64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Min(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Min(size_t &temp_storage_size, BufferView<uint64_t>  d_in, BufferView<uint64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Min(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Min(BufferView<int> d_temp_storage, BufferView<uint64_t>  d_in, BufferView<uint64_t>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Min(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Min(size_t &temp_storage_size, BufferView<float>  d_in, BufferView<float>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Min(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Min(BufferView<int> d_temp_storage, BufferView<float>  d_in, BufferView<float>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Min(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::Min(size_t &temp_storage_size, BufferView<double>  d_in, BufferView<double>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::Min(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::Min(BufferView<int> d_temp_storage, BufferView<double>  d_in, BufferView<double>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::Min(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMin(size_t &temp_storage_size, BufferView<int32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMin(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMin(BufferView<int> d_temp_storage, BufferView<int32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMin(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMin(size_t &temp_storage_size, BufferView<uint32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMin(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMin(BufferView<int> d_temp_storage, BufferView<uint32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMin(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMin(size_t &temp_storage_size, BufferView<int64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMin(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMin(BufferView<int> d_temp_storage, BufferView<int64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMin(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMin(size_t &temp_storage_size, BufferView<uint64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMin(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMin(BufferView<int> d_temp_storage, BufferView<uint64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMin(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMin(size_t &temp_storage_size, BufferView<float>  d_in, BufferView<dcub::KeyValuePair<int32_t,float>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMin(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMin(BufferView<int> d_temp_storage, BufferView<float>  d_in, BufferView<dcub::KeyValuePair<int32_t,float>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMin(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMin(size_t &temp_storage_size, BufferView<double>  d_in, BufferView<dcub::KeyValuePair<int32_t,double>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMin(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMin(BufferView<int> d_temp_storage, BufferView<double>  d_in, BufferView<dcub::KeyValuePair<int32_t,double>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMin(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMax(size_t &temp_storage_size, BufferView<int32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMax(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMax(BufferView<int> d_temp_storage, BufferView<int32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMax(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMax(size_t &temp_storage_size, BufferView<uint32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMax(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMax(BufferView<int> d_temp_storage, BufferView<uint32_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint32_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMax(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMax(size_t &temp_storage_size, BufferView<int64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMax(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMax(BufferView<int> d_temp_storage, BufferView<int64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,int64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMax(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMax(size_t &temp_storage_size, BufferView<uint64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMax(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMax(BufferView<int> d_temp_storage, BufferView<uint64_t>  d_in, BufferView<dcub::KeyValuePair<int32_t,uint64_t>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMax(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMax(size_t &temp_storage_size, BufferView<float>  d_in, BufferView<dcub::KeyValuePair<int32_t,float>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMax(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMax(BufferView<int> d_temp_storage, BufferView<float>  d_in, BufferView<dcub::KeyValuePair<int32_t,float>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMax(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}


void DeviceReduce::ArgMax(size_t &temp_storage_size, BufferView<double>  d_in, BufferView<dcub::KeyValuePair<int32_t,double>>  d_out, int  num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t& temp_storage_bytes) {
    return dcub::DeviceReduce::ArgMax(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceReduce::UCommand DeviceReduce::ArgMax(BufferView<int> d_temp_storage, BufferView<double>  d_in, BufferView<dcub::KeyValuePair<int32_t,double>>  d_out, int  num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t& temp_storage_bytes) {
            return dcub::DeviceReduce::ArgMax(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}
}
