//
// Created by Mike Smith on 2021/3/2.
//
#pragma once

#include <luisa/ast/constant_data.h>
#include <luisa/dsl/expr.h>

namespace luisa::compute {

/// Constant class
template<typename T>
class Constant {

private:
    ConstantData _data;

public:
    /// Default constructor for serialization
    Constant() noexcept = default;

    /// Construct constant from span
    Constant(luisa::span<const T> data) noexcept
        : _data{ConstantData::create(Type::array(Type::of<T>(), data.size()),
                                     data.data(), data.size_bytes())} {}

    /// Construct constant from array
    Constant(const T *data, size_t size) noexcept
        : Constant{luisa::span{data, size}} {}

    /// Construct constant from array-like data
    template<typename U>
    Constant(U &&data) noexcept
        : Constant{luisa::span<const T>{std::forward<U>(data)}} {}

    /// Construct constant from initializer list
    Constant(std::initializer_list<T> init) noexcept
        : Constant{luisa::span<const T>{init.begin(), init.end()}} {}

    Constant(Constant &&) noexcept = default;
    Constant(const Constant &) noexcept = delete;
    Constant &operator=(Constant &&) noexcept = delete;
    Constant &operator=(const Constant &) noexcept = delete;

    /// Access member of constant
    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] auto operator[](U &&index) const noexcept {
        return def<T>(detail::FunctionBuilder::current()->access(
            Type::of<T>(),
            detail::FunctionBuilder::current()->constant(_data),
            detail::extract_expression(std::forward<U>(index))));
    }

    /// Read at index. Same as operator[]
    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return (*this)[std::forward<I>(index)];
    }

    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<typename T>
Constant(luisa::span<T> data) -> Constant<T>;

template<typename T>
Constant(luisa::span<const T> data) -> Constant<T>;

template<typename T>
Constant(std::initializer_list<T>) -> Constant<T>;

template<concepts::container T>
Constant(T &&) -> Constant<std::remove_const_t<typename std::remove_cvref_t<T>::value_type>>;

}// namespace luisa::compute
