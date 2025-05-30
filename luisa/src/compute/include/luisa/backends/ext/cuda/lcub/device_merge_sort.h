// This file is generated by device_merge_sort.py
#pragma once
#include <luisa/core/dll_export.h>// for LC_BACKEND_API
#include <luisa/backends/ext/cuda/lcub/dcub/dcub_common.h>
#include <luisa/backends/ext/cuda/lcub/cuda_lcub_command.h>

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_merge_sort.html
class LC_BACKEND_API DeviceMergeSort {
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using UCommand = luisa::unique_ptr<luisa::compute::cuda::CudaLCubCommand>;
public:

    static void SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairsCopy(size_t &temp_storage_size, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairsCopy(BufferView<int> d_temp_storage, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairsCopy(size_t &temp_storage_size, BufferView<uint32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairsCopy(BufferView<int> d_temp_storage, BufferView<uint32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairsCopy(size_t &temp_storage_size, BufferView<int64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairsCopy(BufferView<int> d_temp_storage, BufferView<int64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairsCopy(size_t &temp_storage_size, BufferView<uint64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairsCopy(BufferView<int> d_temp_storage, BufferView<uint64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairsCopy(size_t &temp_storage_size, BufferView<float> d_input_keys, BufferView<int32_t> d_input_items, BufferView<float> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairsCopy(BufferView<int> d_temp_storage, BufferView<float> d_input_keys, BufferView<int32_t> d_input_items, BufferView<float> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortPairsCopy(size_t &temp_storage_size, BufferView<double> d_input_keys, BufferView<int32_t> d_input_items, BufferView<double> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortPairsCopy(BufferView<int> d_temp_storage, BufferView<double> d_input_keys, BufferView<int32_t> d_input_items, BufferView<double> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeys(size_t &temp_storage_size, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeys(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeys(size_t &temp_storage_size, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeys(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeys(size_t &temp_storage_size, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeys(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeys(size_t &temp_storage_size, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeys(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeys(size_t &temp_storage_size, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeys(BufferView<int> d_temp_storage, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeys(size_t &temp_storage_size, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeys(BufferView<int> d_temp_storage, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeysCopy(size_t &temp_storage_size, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeysCopy(BufferView<int> d_temp_storage, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeysCopy(size_t &temp_storage_size, BufferView<uint32_t> d_input_keys, BufferView<uint32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeysCopy(BufferView<int> d_temp_storage, BufferView<uint32_t> d_input_keys, BufferView<uint32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeysCopy(size_t &temp_storage_size, BufferView<int64_t> d_input_keys, BufferView<int64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeysCopy(BufferView<int> d_temp_storage, BufferView<int64_t> d_input_keys, BufferView<int64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeysCopy(size_t &temp_storage_size, BufferView<uint64_t> d_input_keys, BufferView<uint64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeysCopy(BufferView<int> d_temp_storage, BufferView<uint64_t> d_input_keys, BufferView<uint64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeysCopy(size_t &temp_storage_size, BufferView<float> d_input_keys, BufferView<float> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeysCopy(BufferView<int> d_temp_storage, BufferView<float> d_input_keys, BufferView<float> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void SortKeysCopy(size_t &temp_storage_size, BufferView<double> d_input_keys, BufferView<double> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand SortKeysCopy(BufferView<int> d_temp_storage, BufferView<double> d_input_keys, BufferView<double> d_output_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortPairs(size_t &temp_storage_size, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortPairs(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortPairs(size_t &temp_storage_size, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortPairs(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortPairs(size_t &temp_storage_size, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortPairs(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortPairs(size_t &temp_storage_size, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortPairs(BufferView<int> d_temp_storage, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortPairs(size_t &temp_storage_size, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortPairs(BufferView<int> d_temp_storage, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortKeys(size_t &temp_storage_size, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortKeys(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortKeys(size_t &temp_storage_size, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortKeys(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortKeys(size_t &temp_storage_size, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortKeys(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortKeys(size_t &temp_storage_size, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortKeys(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortKeys(size_t &temp_storage_size, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortKeys(BufferView<int> d_temp_storage, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;

    static void StableSortKeys(size_t &temp_storage_size, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
    static UCommand StableSortKeys(BufferView<int> d_temp_storage, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op = dcub::BinaryOperator::Max) noexcept;
};
}// namespace luisa::compute::cuda::lcub