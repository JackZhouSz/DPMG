// This file is generated by device_merge_sort.py
#pragma once

#include <luisa/backends/ext/cuda/lcub/dcub/dcub_common.h>

namespace luisa::compute::cuda::dcub {

class DCUB_API DeviceMergeSort {
    // DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_merge_sort.html
public:
    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_input_keys, const int32_t *d_input_items, int32_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_input_keys, const int32_t *d_input_items, uint32_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_input_keys, const int32_t *d_input_items, int64_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_input_keys, const int32_t *d_input_items, uint64_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_input_keys, const int32_t *d_input_items, float *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_input_keys, const int32_t *d_input_items, double *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_input_keys, int32_t *d_output_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_input_keys, uint32_t *d_output_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_input_keys, int64_t *d_output_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_input_keys, uint64_t *d_output_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_input_keys, float *d_output_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_input_keys, double *d_output_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);

    static cudaError_t StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int num_items, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr);
};
}// namespace luisa::compute::cuda::dcub