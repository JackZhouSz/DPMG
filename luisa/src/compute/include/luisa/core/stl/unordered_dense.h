///////////////////////// ankerl::unordered_dense::{map, set} /////////////////////////

// A fast & densely stored hashmap and hashset based on robin-hood backward shift deletion.
// Version 2.0.2
// https://github.com/martinus/unordered_dense
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Martin Leitner-Ankerl <martin.ankerl@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// LAST UPDATE IN LUISA COMPUTE: 2022-12-25

#ifndef ANKERL_UNORDERED_DENSE_H
#define ANKERL_UNORDERED_DENSE_H

// see https://semver.org/spec/v2.0.0.html
#define ANKERL_UNORDERED_DENSE_VERSION_MAJOR 2// NOLINT(cppcoreguidelines-macro-usage) incompatible API changes
#define ANKERL_UNORDERED_DENSE_VERSION_MINOR 0// NOLINT(cppcoreguidelines-macro-usage) backwards compatible functionality
#define ANKERL_UNORDERED_DENSE_VERSION_PATCH 2// NOLINT(cppcoreguidelines-macro-usage) backwards compatible bug fixes

// API versioning with inline namespace, see https://www.foonathan.net/2018/11/inline-namespaces/
#define ANKERL_UNORDERED_DENSE_VERSION_CONCAT1(major, minor, patch) v##major##_##minor##_##patch
#define ANKERL_UNORDERED_DENSE_VERSION_CONCAT(major, minor, patch) ANKERL_UNORDERED_DENSE_VERSION_CONCAT1(major, minor, patch)
#define ANKERL_UNORDERED_DENSE_NAMESPACE   \
    ANKERL_UNORDERED_DENSE_VERSION_CONCAT( \
        ANKERL_UNORDERED_DENSE_VERSION_MAJOR, ANKERL_UNORDERED_DENSE_VERSION_MINOR, ANKERL_UNORDERED_DENSE_VERSION_PATCH)

#if defined(_MSVC_LANG)
#define ANKERL_UNORDERED_DENSE_CPP_VERSION _MSVC_LANG
#else
#define ANKERL_UNORDERED_DENSE_CPP_VERSION __cplusplus
#endif

#if defined(__GNUC__)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ANKERL_UNORDERED_DENSE_PACK(decl) decl __attribute__((__packed__))
#elif defined(_MSC_VER)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ANKERL_UNORDERED_DENSE_PACK(decl) __pragma(pack(push, 1)) decl __pragma(pack(pop))
#endif

#if ANKERL_UNORDERED_DENSE_CPP_VERSION < 201703L
#error ankerl::unordered_dense requires C++17 or higher
#else
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <cassert>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#pragma intrinsic(_umul128)
#endif

#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#define ANKERL_UNORDERED_DENSE_LIKELY(x) __builtin_expect(x, 1)  // NOLINT(cppcoreguidelines-macro-usage)
#define ANKERL_UNORDERED_DENSE_UNLIKELY(x) __builtin_expect(x, 0)// NOLINT(cppcoreguidelines-macro-usage)
#else
#define ANKERL_UNORDERED_DENSE_LIKELY(x) (x)  // NOLINT(cppcoreguidelines-macro-usage)
#define ANKERL_UNORDERED_DENSE_UNLIKELY(x) (x)// NOLINT(cppcoreguidelines-macro-usage)
#endif

namespace ankerl::unordered_dense {
inline namespace ANKERL_UNORDERED_DENSE_NAMESPACE {

// hash ///////////////////////////////////////////////////////////////////////

// This is a stripped-down implementation of wyhash: https://github.com/wangyi-fudan/wyhash
// No big-endian support (because different values on different machines don't matter),
// hardcodes seed and the secret, reformattes the code, and clang-tidy fixes.
namespace detail::wyhash {

static inline void mum(uint64_t *a, uint64_t *b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = *a;
    r *= *b;
    *a = static_cast<uint64_t>(r);
    *b = static_cast<uint64_t>(r >> 64U);
#elif defined(_MSC_VER) && defined(_M_X64)
    *a = _umul128(*a, *b, b);
#else
    uint64_t ha = *a >> 32U;
    uint64_t hb = *b >> 32U;
    uint64_t la = static_cast<uint32_t>(*a);
    uint64_t lb = static_cast<uint32_t>(*b);
    uint64_t hi{};
    uint64_t lo{};
    uint64_t rh = ha * hb;
    uint64_t rm0 = ha * lb;
    uint64_t rm1 = hb * la;
    uint64_t rl = la * lb;
    uint64_t t = rl + (rm0 << 32U);
    auto c = static_cast<uint64_t>(t < rl);
    lo = t + (rm1 << 32U);
    c += static_cast<uint64_t>(lo < t);
    hi = rh + (rm0 >> 32U) + (rm1 >> 32U) + c;
    *a = lo;
    *b = hi;
#endif
}

// multiply and xor mix function, aka MUM
[[nodiscard]] static inline auto mix(uint64_t a, uint64_t b) -> uint64_t {
    mum(&a, &b);
    return a ^ b;
}

// read functions. WARNING: we don't care about endianness, so results are different on big endian!
[[nodiscard]] static inline auto r8(const uint8_t *p) -> uint64_t {
    uint64_t v{};
    std::memcpy(&v, p, 8U);
    return v;
}

[[nodiscard]] static inline auto r4(const uint8_t *p) -> uint64_t {
    uint32_t v{};
    std::memcpy(&v, p, 4);
    return v;
}

// reads 1, 2, or 3 bytes
[[nodiscard]] static inline auto r3(const uint8_t *p, size_t k) -> uint64_t {
    return (static_cast<uint64_t>(p[0]) << 16U) | (static_cast<uint64_t>(p[k >> 1U]) << 8U) | p[k - 1];
}

[[maybe_unused]] [[nodiscard]] static inline auto hash(void const *key, size_t len) -> uint64_t {
    static constexpr auto secret = std::array{UINT64_C(0xa0761d6478bd642f),
                                              UINT64_C(0xe7037ed1a0b428db),
                                              UINT64_C(0x8ebc6af09c88c6e3),
                                              UINT64_C(0x589965cc75374cc3)};

    auto const *p = static_cast<uint8_t const *>(key);
    uint64_t seed = secret[0];
    uint64_t a{};
    uint64_t b{};
    if (ANKERL_UNORDERED_DENSE_LIKELY(len <= 16)) {
        if (ANKERL_UNORDERED_DENSE_LIKELY(len >= 4)) {
            a = (r4(p) << 32U) | r4(p + ((len >> 3U) << 2U));
            b = (r4(p + len - 4) << 32U) | r4(p + len - 4 - ((len >> 3U) << 2U));
        } else if (ANKERL_UNORDERED_DENSE_LIKELY(len > 0)) {
            a = r3(p, len);
            b = 0;
        } else {
            a = 0;
            b = 0;
        }
    } else {
        size_t i = len;
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(i > 48)) {
            uint64_t see1 = seed;
            uint64_t see2 = seed;
            do {
                seed = mix(r8(p) ^ secret[1], r8(p + 8) ^ seed);
                see1 = mix(r8(p + 16) ^ secret[2], r8(p + 24) ^ see1);
                see2 = mix(r8(p + 32) ^ secret[3], r8(p + 40) ^ see2);
                p += 48;
                i -= 48;
            } while (ANKERL_UNORDERED_DENSE_LIKELY(i > 48));
            seed ^= see1 ^ see2;
        }
        while (ANKERL_UNORDERED_DENSE_UNLIKELY(i > 16)) {
            seed = mix(r8(p) ^ secret[1], r8(p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = r8(p + i - 16);
        b = r8(p + i - 8);
    }

    return mix(secret[1] ^ len, mix(a ^ secret[1], b ^ seed));
}

[[nodiscard]] static inline auto hash(uint64_t x) -> uint64_t {
    return detail::wyhash::mix(x, UINT64_C(0x9E3779B97F4A7C15));
}

}// namespace detail::wyhash
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// bucket_type //////////////////////////////////////////////////////////

namespace bucket_type {

struct standard {
    static constexpr uint32_t dist_inc = 1U << 8U;            // skip 1 byte fingerprint
    static constexpr uint32_t fingerprint_mask = dist_inc - 1;// mask for 1 byte of fingerprint

    uint32_t m_dist_and_fingerprint;// upper 3 byte: distance to original bucket. lower byte: fingerprint from hash
    uint32_t m_value_idx;           // index into the m_values vector.
};

ANKERL_UNORDERED_DENSE_PACK(struct big {
    static constexpr uint32_t dist_inc = 1U << 8U;            // skip 1 byte fingerprint
    static constexpr uint32_t fingerprint_mask = dist_inc - 1;// mask for 1 byte of fingerprint

    uint32_t m_dist_and_fingerprint;// upper 3 byte: distance to original bucket. lower byte: fingerprint from hash
    size_t m_value_idx;             // index into the m_values vector.
});

}// namespace bucket_type

namespace detail {

struct nonesuch {};

template<class Default, class AlwaysVoid, template<class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};

template<class Default, template<class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

template<template<class...> class Op, class... Args>
using is_detected = typename detail::detector<detail::nonesuch, void, Op, Args...>::value_t;

template<template<class...> class Op, class... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template<typename T>
using detect_avalanching = typename T::is_avalanching;

template<typename T>
using detect_is_transparent = typename T::is_transparent;

template<typename T>
using detect_iterator = typename T::iterator;

template<typename T>
using detect_reserve = decltype(std::declval<T &>().reserve(size_t{}));

// enable_if helpers

template<typename Mapped>
constexpr bool is_map_v = !std::is_void_v<Mapped>;

template<typename Hash, typename KeyEqual>
constexpr bool is_transparent_v = is_detected_v<detect_is_transparent, Hash> || is_detected_v<detect_is_transparent, KeyEqual>;

template<typename From, typename To1, typename To2>
constexpr bool is_neither_convertible_v = !std::is_convertible_v<From, To1> && !std::is_convertible_v<From, To2>;

template<typename T>
constexpr bool has_reserve = is_detected_v<detect_reserve, T>;

// base type for map has mapped_type
template<class T>
struct base_table_type_map {
    using mapped_type = T;
};

// base type for set doesn't have mapped_type
struct base_table_type_set {};

// This is it, the table. Doubles as map and set, and uses `void` for T when its used as a set.
template<class Key,
         class T,// when void, treat it as a set.
         class Hash,
         class KeyEqual,
         class Allocator,
         class Bucket,
         class Vector>
class table : public std::conditional_t<std::is_void_v<T>, base_table_type_set, base_table_type_map<T>> {
public:
    using value_container_type = Vector;

private:
    using bucket_alloc =
        typename std::allocator_traits<Allocator>::template rebind_alloc<Bucket>;
    using bucket_alloc_traits = std::allocator_traits<bucket_alloc>;

    static constexpr uint8_t initial_shifts = 64 - 3;// 2^(64-m_shift) number of buckets
    static constexpr float default_max_load_factor = 0.8F;

public:
    using key_type = Key;
    using value_type = typename value_container_type::value_type;
    using size_type = typename value_container_type::size_type;
    using difference_type = typename value_container_type::difference_type;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using allocator_type = typename value_container_type::allocator_type;
    using reference = typename value_container_type::reference;
    using const_reference = typename value_container_type::const_reference;
    using pointer = typename value_container_type::pointer;
    using const_pointer = typename value_container_type::const_pointer;
    using iterator = typename value_container_type::iterator;
    using const_iterator = typename value_container_type::const_iterator;
    using bucket_type = Bucket;

private:
    using value_idx_type = decltype(Bucket::m_value_idx);
    using dist_and_fingerprint_type = decltype(Bucket::m_dist_and_fingerprint);

    static_assert(std::is_trivially_destructible_v<Bucket>, "assert there's no need to call destructor / std::destroy");
    static_assert(std::is_trivially_copyable_v<Bucket>, "assert we can just memset / memcpy");

    value_container_type m_values{};// Contains all the key-value pairs in one densely stored container. No holes.
    typename std::allocator_traits<bucket_alloc>::pointer m_buckets{};
    size_t m_num_buckets = 0;
    size_t m_max_bucket_capacity = 0;
    float m_max_load_factor = default_max_load_factor;
    Hash m_hash{};
    KeyEqual m_equal{};
    uint8_t m_shifts = initial_shifts;

    [[nodiscard]] auto next(value_idx_type bucket_idx) const -> value_idx_type {
        return ANKERL_UNORDERED_DENSE_UNLIKELY(bucket_idx + 1U == m_num_buckets) ? 0 : static_cast<value_idx_type>(bucket_idx + 1U);
    }

    // Helper to access bucket through pointer types
    [[nodiscard]] static constexpr auto at(typename std::allocator_traits<bucket_alloc>::pointer bucket_ptr, size_t offset)
        -> Bucket & {
        return *(bucket_ptr + static_cast<typename std::allocator_traits<bucket_alloc>::difference_type>(offset));
    }

    // use the dist_inc and dist_dec functions so that uint16_t types work without warning
    [[nodiscard]] static constexpr auto dist_inc(dist_and_fingerprint_type x) -> dist_and_fingerprint_type {
        return static_cast<dist_and_fingerprint_type>(x + Bucket::dist_inc);
    }

    [[nodiscard]] static constexpr auto dist_dec(dist_and_fingerprint_type x) -> dist_and_fingerprint_type {
        return static_cast<dist_and_fingerprint_type>(x - Bucket::dist_inc);
    }

    // The goal of mixed_hash is to always produce a high quality 64bit hash.
    template<typename K>
    [[nodiscard]] constexpr auto mixed_hash(K const &key) const -> uint64_t {
        if constexpr (is_detected_v<detect_avalanching, Hash>) {
            // we know that the hash is good because is_avalanching.
            if constexpr (sizeof(decltype(m_hash(key))) < sizeof(uint64_t)) {
                // 32bit hash and is_avalanching => multiply with a constant to avalanche bits upwards
                return m_hash(key) * UINT64_C(0x9ddfea08eb382d69);
            } else {
                // 64bit and is_avalanching => only use the hash itself.
                return m_hash(key);
            }
        } else {
            // not is_avalanching => apply wyhash
            return wyhash::hash(m_hash(key));
        }
    }

    [[nodiscard]] constexpr auto dist_and_fingerprint_from_hash(uint64_t hash) const -> dist_and_fingerprint_type {
        return Bucket::dist_inc | (static_cast<dist_and_fingerprint_type>(hash) & Bucket::fingerprint_mask);
    }

    [[nodiscard]] constexpr auto bucket_idx_from_hash(uint64_t hash) const -> value_idx_type {
        return static_cast<value_idx_type>(hash >> m_shifts);
    }

    [[nodiscard]] static constexpr auto get_key(value_type const &vt) -> key_type const & {
        if constexpr (std::is_void_v<T>) {
            return vt;
        } else {
            return vt.first;
        }
    }

    template<typename K>
    [[nodiscard]] auto next_while_less(K const &key) const -> Bucket {
        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (dist_and_fingerprint < at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }
        return {dist_and_fingerprint, bucket_idx};
    }

    void place_and_shift_up(Bucket bucket, value_idx_type place) {
        while (0 != at(m_buckets, place).m_dist_and_fingerprint) {
            bucket = std::exchange(at(m_buckets, place), bucket);
            bucket.m_dist_and_fingerprint = dist_inc(bucket.m_dist_and_fingerprint);
            place = next(place);
        }
        at(m_buckets, place) = bucket;
    }

    [[nodiscard]] static constexpr auto calc_num_buckets(uint8_t shifts) -> size_t {
        return std::min(max_bucket_count(), size_t{1} << (64U - shifts));
    }

    [[nodiscard]] constexpr auto calc_shifts_for_size(size_t s) const -> uint8_t {
        auto shifts = initial_shifts;
        while (shifts > 0 && static_cast<size_t>(static_cast<float>(calc_num_buckets(shifts)) * max_load_factor()) < s) {
            --shifts;
        }
        return shifts;
    }

    // assumes m_values has data, m_buckets=m_buckets_end=nullptr, m_shifts is INITIAL_SHIFTS
    void copy_buckets(table const &other) {
        if (!empty()) {
            m_shifts = other.m_shifts;
            allocate_buckets_from_shift();
            std::memcpy(m_buckets, other.m_buckets, sizeof(Bucket) * bucket_count());
        }
    }

    /**
     * True when no element can be added any more without increasing the size
     */
    [[nodiscard]] auto is_full() const -> bool {
        return size() >= m_max_bucket_capacity;
    }

    void deallocate_buckets() {
        auto ba = bucket_alloc(Allocator{});
        if (nullptr != m_buckets) {
            bucket_alloc_traits::deallocate(ba, m_buckets, bucket_count());
        }
        m_buckets = nullptr;
        m_num_buckets = 0;
        m_max_bucket_capacity = 0;
    }

    void allocate_buckets_from_shift() {
        auto ba = bucket_alloc(Allocator{});
        m_num_buckets = calc_num_buckets(m_shifts);
        m_buckets = bucket_alloc_traits::allocate(ba, m_num_buckets);
        if (m_num_buckets == max_bucket_count()) {
            // reached the maximum, make sure we can use each bucket
            m_max_bucket_capacity = max_bucket_count();
        } else {
            m_max_bucket_capacity = static_cast<value_idx_type>(static_cast<float>(m_num_buckets) * max_load_factor());
        }
    }

    void clear_buckets() {
        if (m_buckets != nullptr) {
            std::memset(&*m_buckets, 0, sizeof(Bucket) * bucket_count());
        }
    }

    void clear_and_fill_buckets_from_values() {
        clear_buckets();
        for (value_idx_type value_idx = 0, end_idx = static_cast<value_idx_type>(m_values.size()); value_idx < end_idx;
             ++value_idx) {
            auto const &key = get_key(m_values[value_idx]);
            auto [dist_and_fingerprint, bucket] = next_while_less(key);

            // we know for certain that key has not yet been inserted, so no need to check it.
            place_and_shift_up({dist_and_fingerprint, value_idx}, bucket);
        }
    }

    void increase_size() {
        // "ankerl::unordered_dense: reached max bucket size, cannot increase size"
        assert(!(ANKERL_UNORDERED_DENSE_UNLIKELY(m_max_bucket_capacity == max_bucket_count())));
        --m_shifts;
        deallocate_buckets();
        allocate_buckets_from_shift();
        clear_and_fill_buckets_from_values();
    }

    void do_erase(value_idx_type bucket_idx) {
        auto const value_idx_to_remove = at(m_buckets, bucket_idx).m_value_idx;

        // shift down until either empty or an element with correct spot is found
        auto next_bucket_idx = next(bucket_idx);
        while (at(m_buckets, next_bucket_idx).m_dist_and_fingerprint >= Bucket::dist_inc * 2) {
            at(m_buckets, bucket_idx) = {dist_dec(at(m_buckets, next_bucket_idx).m_dist_and_fingerprint),
                                         at(m_buckets, next_bucket_idx).m_value_idx};
            bucket_idx = std::exchange(next_bucket_idx, next(next_bucket_idx));
        }
        at(m_buckets, bucket_idx) = {};

        // update m_values
        if (value_idx_to_remove != m_values.size() - 1) {
            // no luck, we'll have to replace the value with the last one and update the index accordingly
            auto &val = m_values[value_idx_to_remove];
            val = std::move(m_values.back());

            // update the values_idx of the moved entry. No need to play the info game, just look until we find the values_idx
            auto mh = mixed_hash(get_key(val));
            bucket_idx = bucket_idx_from_hash(mh);

            auto const values_idx_back = static_cast<value_idx_type>(m_values.size() - 1);
            while (values_idx_back != at(m_buckets, bucket_idx).m_value_idx) {
                bucket_idx = next(bucket_idx);
            }
            at(m_buckets, bucket_idx).m_value_idx = value_idx_to_remove;
        }
        m_values.pop_back();
    }

    template<typename K>
    auto do_erase_key(K &&key) -> size_t {
        if (empty()) {
            return 0;
        }

        auto [dist_and_fingerprint, bucket_idx] = next_while_less(key);

        while (dist_and_fingerprint == at(m_buckets, bucket_idx).m_dist_and_fingerprint &&
               !m_equal(key, get_key(m_values[at(m_buckets, bucket_idx).m_value_idx]))) {
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }

        if (dist_and_fingerprint != at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            return 0;
        }
        do_erase(bucket_idx);
        return 1;
    }

    template<class K, class M>
    auto do_insert_or_assign(K &&key, M &&mapped) -> std::pair<iterator, bool> {
        auto it_isinserted = try_emplace(std::forward<K>(key), std::forward<M>(mapped));
        if (!it_isinserted.second) {
            it_isinserted.first->second = std::forward<M>(mapped);
        }
        return it_isinserted;
    }

    template<typename K, typename... Args>
    auto do_place_element(dist_and_fingerprint_type dist_and_fingerprint, value_idx_type bucket_idx, K &&key, Args &&...args)
        -> std::pair<iterator, bool> {

        // emplace the new value. If that throws an exception, no harm done; index is still in a valid state
        m_values.emplace_back(std::piecewise_construct,
                              std::forward_as_tuple(std::forward<K>(key)),
                              std::forward_as_tuple(std::forward<Args>(args)...));

        // place element and shift up until we find an empty spot
        auto value_idx = static_cast<value_idx_type>(m_values.size() - 1);
        place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);
        return {begin() + static_cast<difference_type>(value_idx), true};
    }

    template<typename K, typename... Args>
    auto do_try_emplace(K &&key, Args &&...args) -> std::pair<iterator, bool> {
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(is_full())) {
            increase_size();
        }

        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (true) {
            auto *bucket = &at(m_buckets, bucket_idx);
            if (dist_and_fingerprint == bucket->m_dist_and_fingerprint) {
                if (m_equal(key, m_values[bucket->m_value_idx].first)) {
                    return {begin() + static_cast<difference_type>(bucket->m_value_idx), false};
                }
            } else if (dist_and_fingerprint > bucket->m_dist_and_fingerprint) {
                return do_place_element(dist_and_fingerprint, bucket_idx, std::forward<K>(key), std::forward<Args>(args)...);
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }
    }

    template<typename K>
    auto do_find(K const &key) -> iterator {
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(empty())) {
            return end();
        }

        auto mh = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(mh);
        auto bucket_idx = bucket_idx_from_hash(mh);
        auto *bucket = &at(m_buckets, bucket_idx);

        // unrolled loop. *Always* check a few directly, then enter the loop. This is faster.
        if (dist_and_fingerprint == bucket->m_dist_and_fingerprint && m_equal(key, get_key(m_values[bucket->m_value_idx]))) {
            return begin() + static_cast<difference_type>(bucket->m_value_idx);
        }
        dist_and_fingerprint = dist_inc(dist_and_fingerprint);
        bucket_idx = next(bucket_idx);
        bucket = &at(m_buckets, bucket_idx);

        if (dist_and_fingerprint == bucket->m_dist_and_fingerprint && m_equal(key, get_key(m_values[bucket->m_value_idx]))) {
            return begin() + static_cast<difference_type>(bucket->m_value_idx);
        }
        dist_and_fingerprint = dist_inc(dist_and_fingerprint);
        bucket_idx = next(bucket_idx);
        bucket = &at(m_buckets, bucket_idx);

        while (true) {
            if (dist_and_fingerprint == bucket->m_dist_and_fingerprint) {
                if (m_equal(key, get_key(m_values[bucket->m_value_idx]))) {
                    return begin() + static_cast<difference_type>(bucket->m_value_idx);
                }
            } else if (dist_and_fingerprint > bucket->m_dist_and_fingerprint) {
                return end();
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
            bucket = &at(m_buckets, bucket_idx);
        }
    }

    template<typename K>
    auto do_find(K const &key) const -> const_iterator {
        return const_cast<table *>(this)->do_find(key);// NOLINT(cppcoreguidelines-pro-type-const-cast)
    }

    template<typename K, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto do_at(K const &key) -> Q & {
        if (auto it = find(key); end() != it) {
            return it->second;
        }
        assert(false && "ankerl::unordered_dense::map::at(): key not found");
        abort();
    }

    template<typename K, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto do_at(K const &key) const -> Q const & {
        return const_cast<table *>(this)->at(key);// NOLINT(cppcoreguidelines-pro-type-const-cast)
    }

public:
    table()
        : table(0) {}

    explicit table(size_t bucket_count,
                   Hash const &hash = Hash(),
                   KeyEqual const &equal = KeyEqual(),
                   allocator_type const &alloc_or_container = allocator_type())
        : m_values(alloc_or_container), m_hash(hash), m_equal(equal) {
        if (0 != bucket_count) {
            reserve(bucket_count);
        }
    }

    table(size_t bucket_count, allocator_type const &alloc)
        : table(bucket_count, Hash(), KeyEqual(), alloc) {}

    table(size_t bucket_count, Hash const &hash, allocator_type const &alloc)
        : table(bucket_count, hash, KeyEqual(), alloc) {}

    explicit table(allocator_type const &alloc)
        : table(0, Hash(), KeyEqual(), alloc) {}

    template<class InputIt>
    table(InputIt first,
          InputIt last,
          size_type bucket_count = 0,
          Hash const &hash = Hash(),
          KeyEqual const &equal = KeyEqual(),
          allocator_type const &alloc = allocator_type())
        : table(bucket_count, hash, equal, alloc) {
        insert(first, last);
    }

    template<class InputIt>
    table(InputIt first, InputIt last, size_type bucket_count, allocator_type const &alloc)
        : table(first, last, bucket_count, Hash(), KeyEqual(), alloc) {}

    template<class InputIt>
    table(InputIt first, InputIt last, size_type bucket_count, Hash const &hash, allocator_type const &alloc)
        : table(first, last, bucket_count, hash, KeyEqual(), alloc) {}

    table(table const &other)
        : m_values(other.m_values), m_max_load_factor(other.m_max_load_factor), m_hash(other.m_hash), m_equal(other.m_equal) {
        copy_buckets(other);
    }
    table(table &&other) noexcept
        : m_values(std::move(other.m_values)), m_buckets(std::exchange(other.m_buckets, nullptr)), m_num_buckets(std::exchange(other.m_num_buckets, 0)), m_max_bucket_capacity(std::exchange(other.m_max_bucket_capacity, 0)), m_max_load_factor(std::exchange(other.m_max_load_factor, default_max_load_factor)), m_hash(std::exchange(other.m_hash, {})), m_equal(std::exchange(other.m_equal, {})), m_shifts(std::exchange(other.m_shifts, initial_shifts)) {
        other.m_values.clear();
    }

    table(std::initializer_list<value_type> ilist,
          size_t bucket_count = 0,
          Hash const &hash = Hash(),
          KeyEqual const &equal = KeyEqual(),
          allocator_type const &alloc = allocator_type())
        : table(bucket_count, hash, equal, alloc) {
        insert(ilist);
    }

    table(std::initializer_list<value_type> ilist, size_type bucket_count, allocator_type const &alloc)
        : table(ilist, bucket_count, Hash(), KeyEqual(), alloc) {}

    table(std::initializer_list<value_type> init, size_type bucket_count, Hash const &hash, allocator_type const &alloc)
        : table(init, bucket_count, hash, KeyEqual(), alloc) {}

    ~table() {
        auto ba = bucket_alloc(Allocator{});
        bucket_alloc_traits::deallocate(ba, m_buckets, bucket_count());
    }

    auto operator=(table const &other) -> table & {
        if (&other != this) {
            deallocate_buckets();// deallocate before m_values is set (might have another allocator)
            m_values = other.m_values;
            m_max_load_factor = other.m_max_load_factor;
            m_hash = other.m_hash;
            m_equal = other.m_equal;
            m_shifts = initial_shifts;
            copy_buckets(other);
        }
        return *this;
    }

    auto operator=(table &&other) noexcept(
        noexcept(std::is_nothrow_move_assignable_v<value_container_type> &&std::is_nothrow_move_assignable_v<Hash> &&
                     std::is_nothrow_move_assignable_v<KeyEqual>)) -> table & {
        if (&other != this) {
            deallocate_buckets();// deallocate before m_values is set (might have another allocator)
            m_values = std::move(other.m_values);
            m_buckets = std::exchange(other.m_buckets, nullptr);
            m_num_buckets = std::exchange(other.m_num_buckets, 0);
            m_max_bucket_capacity = std::exchange(other.m_max_bucket_capacity, 0);
            m_max_load_factor = std::exchange(other.m_max_load_factor, default_max_load_factor);
            m_hash = std::exchange(other.m_hash, {});
            m_equal = std::exchange(other.m_equal, {});
            m_shifts = std::exchange(other.m_shifts, initial_shifts);
            other.m_values.clear();
        }
        return *this;
    }

    auto operator=(std::initializer_list<value_type> ilist) -> table & {
        clear();
        insert(ilist);
        return *this;
    }

    auto get_allocator() const noexcept -> allocator_type {
        return Allocator{};
    }

    // iterators //////////////////////////////////////////////////////////////

    auto begin() noexcept -> iterator {
        return m_values.begin();
    }

    auto begin() const noexcept -> const_iterator {
        return m_values.begin();
    }

    auto cbegin() const noexcept -> const_iterator {
        return m_values.cbegin();
    }

    auto end() noexcept -> iterator {
        return m_values.end();
    }

    auto cend() const noexcept -> const_iterator {
        return m_values.cend();
    }

    auto end() const noexcept -> const_iterator {
        return m_values.end();
    }

    // capacity ///////////////////////////////////////////////////////////////

    [[nodiscard]] auto empty() const noexcept -> bool {
        return m_values.empty();
    }

    [[nodiscard]] auto size() const noexcept -> size_t {
        return m_values.size();
    }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_t {
        if constexpr (std::numeric_limits<value_idx_type>::max() == std::numeric_limits<size_t>::max()) {
            return size_t{1} << (sizeof(value_idx_type) * 8 - 1);
        } else {
            return size_t{1} << (sizeof(value_idx_type) * 8);
        }
    }

    // modifiers //////////////////////////////////////////////////////////////

    void clear() {
        m_values.clear();
        clear_buckets();
    }

    auto insert(value_type const &value) -> std::pair<iterator, bool> {
        return emplace(value);
    }

    auto insert(value_type &&value) -> std::pair<iterator, bool> {
        return emplace(std::move(value));
    }

    template<class P, std::enable_if_t<std::is_constructible_v<value_type, P &&>, bool> = true>
    auto insert(P &&value) -> std::pair<iterator, bool> {
        return emplace(std::forward<P>(value));
    }

    auto insert(const_iterator /*hint*/, value_type const &value) -> iterator {
        return insert(value).first;
    }

    auto insert(const_iterator /*hint*/, value_type &&value) -> iterator {
        return insert(std::move(value)).first;
    }

    template<class P, std::enable_if_t<std::is_constructible_v<value_type, P &&>, bool> = true>
    auto insert(const_iterator /*hint*/, P &&value) -> iterator {
        return insert(std::forward<P>(value)).first;
    }

    template<class InputIt>
    void insert(InputIt first, InputIt last) {
        while (first != last) {
            insert(*first);
            ++first;
        }
    }

    void insert(std::initializer_list<value_type> ilist) {
        insert(ilist.begin(), ilist.end());
    }

    // nonstandard API: *this is emptied.
    // Also see "A Standard flat_map" https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0429r9.pdf
    auto extract() && -> value_container_type {
        return std::move(m_values);
    }

    // nonstandard API:
    // Discards the internally held container and replaces it with the one passed. Erases non-unique elements.
    auto replace(value_container_type &&container) {
        // "ankerl::unordered_dense::map::replace(): too many elements"
        assert(container.size() <= max_size());

        auto shifts = calc_shifts_for_size(container.size());
        if (0 == m_num_buckets || shifts < m_shifts || container.get_allocator() != Allocator{}) {
            m_shifts = shifts;
            deallocate_buckets();
            allocate_buckets_from_shift();
        }
        clear_buckets();

        m_values = std::move(container);

        // can't use clear_and_fill_buckets_from_values() because container elements might not be unique
        auto value_idx = value_idx_type{};

        // loop until we reach the end of the container. duplicated entries will be replaced with back().
        while (value_idx != static_cast<value_idx_type>(m_values.size())) {
            auto const &key = get_key(m_values[value_idx]);

            auto hash = mixed_hash(key);
            auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
            auto bucket_idx = bucket_idx_from_hash(hash);

            bool key_found = false;
            while (true) {
                auto const &bucket = at(m_buckets, bucket_idx);
                if (dist_and_fingerprint > bucket.m_dist_and_fingerprint) {
                    break;
                }
                if (dist_and_fingerprint == bucket.m_dist_and_fingerprint &&
                    m_equal(key, m_values[bucket.m_value_idx].first)) {
                    key_found = true;
                    break;
                }
                dist_and_fingerprint = dist_inc(dist_and_fingerprint);
                bucket_idx = next(bucket_idx);
            }

            if (key_found) {
                if (value_idx != static_cast<value_idx_type>(m_values.size() - 1)) {
                    m_values[value_idx] = std::move(m_values.back());
                }
                m_values.pop_back();
            } else {
                place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);
                ++value_idx;
            }
        }
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(Key const &key, M &&mapped) -> std::pair<iterator, bool> {
        return do_insert_or_assign(key, std::forward<M>(mapped));
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(Key &&key, M &&mapped) -> std::pair<iterator, bool> {
        return do_insert_or_assign(std::move(key), std::forward<M>(mapped));
    }

    template<typename K,
             typename M,
             typename Q = T,
             typename H = Hash,
             typename KE = KeyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(K &&key, M &&mapped) -> std::pair<iterator, bool> {
        return do_insert_or_assign(std::forward<K>(key), std::forward<M>(mapped));
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key const &key, M &&mapped) -> iterator {
        return do_insert_or_assign(key, std::forward<M>(mapped)).first;
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key &&key, M &&mapped) -> iterator {
        return do_insert_or_assign(std::move(key), std::forward<M>(mapped)).first;
    }

    template<typename K,
             typename M,
             typename Q = T,
             typename H = Hash,
             typename KE = KeyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, K &&key, M &&mapped) -> iterator {
        return do_insert_or_assign(std::forward<K>(key), std::forward<M>(mapped)).first;
    }

    // Single arguments for unordered_set can be used without having to construct the value_type
    template<class K,
             typename Q = T,
             typename H = Hash,
             typename KE = KeyEqual,
             std::enable_if_t<!is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto emplace(K &&key) -> std::pair<iterator, bool> {
        if (is_full()) {
            increase_size();
        }

        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (dist_and_fingerprint <= at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            if (dist_and_fingerprint == at(m_buckets, bucket_idx).m_dist_and_fingerprint &&
                m_equal(key, m_values[at(m_buckets, bucket_idx).m_value_idx])) {
                // found it, return without ever actually creating anything
                return {begin() + static_cast<difference_type>(at(m_buckets, bucket_idx).m_value_idx), false};
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }

        // value is new, insert element first, so when exception happens we are in a valid state
        m_values.emplace_back(std::forward<K>(key));
        // now place the bucket and shift up until we find an empty spot
        auto value_idx = static_cast<value_idx_type>(m_values.size() - 1);
        place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);
        return {begin() + static_cast<difference_type>(value_idx), true};
    }

    template<class... Args>
    auto emplace(Args &&...args) -> std::pair<iterator, bool> {
        if (is_full()) {
            increase_size();
        }

        // we have to instantiate the value_type to be able to access the key.
        // 1. emplace_back the object so it is constructed. 2. If the key is already there, pop it later in the loop.
        auto &key = get_key(m_values.emplace_back(std::forward<Args>(args)...));
        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (dist_and_fingerprint <= at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            if (dist_and_fingerprint == at(m_buckets, bucket_idx).m_dist_and_fingerprint &&
                m_equal(key, get_key(m_values[at(m_buckets, bucket_idx).m_value_idx]))) {
                m_values.pop_back();// value was already there, so get rid of it
                return {begin() + static_cast<difference_type>(at(m_buckets, bucket_idx).m_value_idx), false};
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }

        // value is new, place the bucket and shift up until we find an empty spot
        auto value_idx = static_cast<value_idx_type>(m_values.size() - 1);
        place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);

        return {begin() + static_cast<difference_type>(value_idx), true};
    }

    template<class... Args>
    auto emplace_hint(const_iterator /*hint*/, Args &&...args) -> iterator {
        return emplace(std::forward<Args>(args)...).first;
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key const &key, Args &&...args) -> std::pair<iterator, bool> {
        return do_try_emplace(key, std::forward<Args>(args)...);
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key &&key, Args &&...args) -> std::pair<iterator, bool> {
        return do_try_emplace(std::move(key), std::forward<Args>(args)...);
    }
    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto force_emplace(Key const &key, Args &&...args) -> iterator {
        auto ret = do_try_emplace(key, std::forward<Args>(args)...);
        if constexpr (!std::is_same_v<T, void>) {
            if (!ret.second) {
                ret.first->second.~T();
                new (&ret.first->second) T(std::forward<Args>(args)...);
            }
        }
        return ret.first;
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto force_emplace(Key &&key, Args &&...args) -> iterator {
        auto ret = do_try_emplace(std::move(key), std::forward<Args>(args)...);
        if constexpr (!std::is_same_v<T, void>) {
            if (!ret.second) {
                ret.first->second.~T();
                new (&ret.first->second) T(std::forward<Args>(args)...);
            }
        }
        return ret.first;
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key const &key, Args &&...args) -> iterator {
        return do_try_emplace(key, std::forward<Args>(args)...).first;
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key &&key, Args &&...args) -> iterator {
        return do_try_emplace(std::move(key), std::forward<Args>(args)...).first;
    }

    template<
        typename K,
        typename... Args,
        typename Q = T,
        typename H = Hash,
        typename KE = KeyEqual,
        std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE> && is_neither_convertible_v<K &&, iterator, const_iterator>,
                         bool> = true>
    auto try_emplace(K &&key, Args &&...args) -> std::pair<iterator, bool> {
        return do_try_emplace(std::forward<K>(key), std::forward<Args>(args)...);
    }
    template<
        typename K,
        typename... Args,
        typename Q = T,
        typename H = Hash,
        typename KE = KeyEqual,
        std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE> && is_neither_convertible_v<K &&, iterator, const_iterator>,
                         bool> = true>
    auto force_emplace(K &&key, Args &&...args) -> iterator {
        auto ret = do_try_emplace(std::forward<K>(key), std::forward<Args>(args)...);
        if constexpr (!std::is_same_v<T, void>) {
            if (!ret.second) {
                ret.first->second.~T();
                new (&ret.first->second) T(std::forward<Args>(args)...);
            }
        }
        return ret.first;
    }

    template<
        typename K,
        typename... Args,
        typename Q = T,
        typename H = Hash,
        typename KE = KeyEqual,
        std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE> && is_neither_convertible_v<K &&, iterator, const_iterator>,
                         bool> = true>
    auto try_emplace(const_iterator /*hint*/, K &&key, Args &&...args) -> iterator {
        return do_try_emplace(std::forward<K>(key), std::forward<Args>(args)...).first;
    }

    auto erase(iterator it) -> iterator {
        auto hash = mixed_hash(get_key(*it));
        auto bucket_idx = bucket_idx_from_hash(hash);

        auto const value_idx_to_remove = static_cast<value_idx_type>(it - cbegin());
        while (at(m_buckets, bucket_idx).m_value_idx != value_idx_to_remove) {
            bucket_idx = next(bucket_idx);
        }

        do_erase(bucket_idx);
        return begin() + static_cast<difference_type>(value_idx_to_remove);
    }

    auto erase(const_iterator it) -> iterator {
        return erase(begin() + (it - cbegin()));
    }

    auto erase(const_iterator first, const_iterator last) -> iterator {
        auto const idx_first = first - cbegin();
        auto const idx_last = last - cbegin();
        auto const first_to_last = std::distance(first, last);
        auto const last_to_end = std::distance(last, cend());

        // remove elements from left to right which moves elements from the end back
        auto const mid = idx_first + std::min(first_to_last, last_to_end);
        auto idx = idx_first;
        while (idx != mid) {
            erase(begin() + idx);
            ++idx;
        }

        // all elements from the right are moved, now remove the last element until all done
        idx = idx_last;
        while (idx != mid) {
            --idx;
            erase(begin() + idx);
        }

        return begin() + idx_first;
    }

    auto erase(Key const &key) -> size_t {
        return do_erase_key(key);
    }

    template<class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto erase(K &&key) -> size_t {
        return do_erase_key(std::forward<K>(key));
    }

    void swap(table &other) noexcept(noexcept(std::is_nothrow_swappable_v<value_container_type> &&
                                                  std::is_nothrow_swappable_v<Hash> &&std::is_nothrow_swappable_v<KeyEqual>)) {
        using std::swap;
        swap(other, *this);
    }

    // lookup /////////////////////////////////////////////////////////////////

    template<typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto at(key_type const &key) -> Q & {
        return do_at(key);
    }

    template<typename K,
             typename Q = T,
             typename H = Hash,
             typename KE = KeyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto at(K const &key) -> Q & {
        return do_at(key);
    }

    template<typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto at(key_type const &key) const -> Q const & {
        return do_at(key);
    }

    template<typename K,
             typename Q = T,
             typename H = Hash,
             typename KE = KeyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto at(K const &key) const -> Q const & {
        return do_at(key);
    }

    template<typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto operator[](Key const &key) -> Q & {
        return try_emplace(key).first->second;
    }

    template<typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto operator[](Key &&key) -> Q & {
        return try_emplace(std::move(key)).first->second;
    }

    template<typename K,
             typename Q = T,
             typename H = Hash,
             typename KE = KeyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto operator[](K &&key) -> Q & {
        return try_emplace(std::forward<K>(key)).first->second;
    }

    auto count(Key const &key) const -> size_t {
        return find(key) == end() ? 0 : 1;
    }

    template<class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto count(K const &key) const -> size_t {
        return find(key) == end() ? 0 : 1;
    }

    auto find(Key const &key) -> iterator {
        return do_find(key);
    }

    auto find(Key const &key) const -> const_iterator {
        return do_find(key);
    }

    template<class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto find(K const &key) -> iterator {
        return do_find(key);
    }

    template<class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto find(K const &key) const -> const_iterator {
        return do_find(key);
    }

    auto contains(Key const &key) const -> bool {
        return find(key) != end();
    }

    template<class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto contains(K const &key) const -> bool {
        return find(key) != end();
    }

    auto equal_range(Key const &key) -> std::pair<iterator, iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    auto equal_range(const Key &key) const -> std::pair<const_iterator, const_iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    template<class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto equal_range(K const &key) -> std::pair<iterator, iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    template<class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto equal_range(K const &key) const -> std::pair<const_iterator, const_iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    // bucket interface ///////////////////////////////////////////////////////

    auto bucket_count() const noexcept -> size_t {// NOLINT(modernize-use-nodiscard)
        return m_num_buckets;
    }

    static constexpr auto max_bucket_count() noexcept -> size_t {// NOLINT(modernize-use-nodiscard)
        return max_size();
    }

    // hash policy ////////////////////////////////////////////////////////////

    [[nodiscard]] auto load_factor() const -> float {
        return bucket_count() ? static_cast<float>(size()) / static_cast<float>(bucket_count()) : 0.0F;
    }

    [[nodiscard]] auto max_load_factor() const -> float {
        return m_max_load_factor;
    }

    void max_load_factor(float ml) {
        m_max_load_factor = ml;
        if (m_num_buckets != max_bucket_count()) {
            m_max_bucket_capacity = static_cast<value_idx_type>(static_cast<float>(bucket_count()) * max_load_factor());
        }
    }

    void rehash(size_t count) {
        count = std::min(count, max_size());
        auto shifts = calc_shifts_for_size(std::max(count, size()));
        if (shifts != m_shifts) {
            m_shifts = shifts;
            deallocate_buckets();
            m_values.shrink_to_fit();
            allocate_buckets_from_shift();
            clear_and_fill_buckets_from_values();
        }
    }

    void reserve(size_t capa) {
        capa = std::min(capa, max_size());
        if constexpr (has_reserve<value_container_type>) {
            // std::deque doesn't have reserve(). Make sure we only call when available
            m_values.reserve(capa);
        }
        auto shifts = calc_shifts_for_size(std::max(capa, size()));
        if (0 == m_num_buckets || shifts < m_shifts) {
            m_shifts = shifts;
            deallocate_buckets();
            allocate_buckets_from_shift();
            clear_and_fill_buckets_from_values();
        }
    }

    // observers //////////////////////////////////////////////////////////////

    auto hash_function() const -> hasher {
        return m_hash;
    }

    auto key_eq() const -> key_equal {
        return m_equal;
    }

    // nonstandard API: expose the underlying values container
    [[nodiscard]] auto values() const noexcept -> value_container_type const & {
        return m_values;
    }

    // non-member functions ///////////////////////////////////////////////////

    friend auto operator==(table const &a, table const &b) -> bool {
        if (&a == &b) {
            return true;
        }
        if (a.size() != b.size()) {
            return false;
        }
        for (auto const &b_entry : b) {
            auto it = a.find(get_key(b_entry));
            if constexpr (std::is_void_v<T>) {
                // set: only check that the key is here
                if (a.end() == it) {
                    return false;
                }
            } else {
                // map: check that key is here, then also check that value is the same
                if (a.end() == it || !(b_entry.second == it->second)) {
                    return false;
                }
            }
        }
        return true;
    }

    friend auto operator!=(table const &a, table const &b) -> bool {
        return !(a == b);
    }
};

}// namespace detail

template<class Key,
         class T,
         class Hash,
         class KeyEqual,
         class Allocator,
         class Vector,
         class Bucket = bucket_type::standard>
using map = detail::table<Key, T, Hash, KeyEqual, Allocator, Bucket, Vector>;

template<class Key,
         class Hash,
         class KeyEqual,
         class Allocator,
         class Vector,
         class Bucket = bucket_type::standard>
using set = detail::table<Key, void, Hash, KeyEqual, Allocator, Bucket, Vector>;

// deduction guides ///////////////////////////////////////////////////////////

// deduction guides for alias templates are only possible since C++20
// see https://en.cppreference.com/w/cpp/language/class_template_argument_deduction

}
}// namespace ankerl::unordered_dense::ANKERL_UNORDERED_DENSE_NAMESPACE

// std extensions /////////////////////////////////////////////////////////////

namespace std {// NOLINT(cert-dcl58-cpp)

template<class Key, class T, class Hash, class KeyEqual, class Allocator, class Bucket, class Vector, class Pred>
auto erase_if(ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, Allocator, Bucket, Vector> &map, Pred pred)
    -> size_t {
    using map_t = ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, Allocator, Bucket, Vector>;

    // going back to front because erase() invalidates the end iterator
    auto const old_size = map.size();
    auto idx = old_size;
    while (idx) {
        --idx;
        auto it = map.begin() + static_cast<typename map_t::difference_type>(idx);
        if (pred(*it)) {
            map.erase(it);
        }
    }

    return map.size() - old_size;
}

}// namespace std

#endif
#endif

