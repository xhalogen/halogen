#pragma once
#include "template.hpp"
#include <optional>
#include <functional>
#include <type_traits>

namespace halogen {
    template<typename T>
    requires std::is_arithmetic_v<T>
    struct identity_traits<std::plus<T>, T> {
        static constexpr T value() { return T(0); };
    };

    template<typename T>
    requires std::is_arithmetic_v<T>
    struct identity_traits<std::multiplies<T>, T> {
        static constexpr T value() { return T(1); };
    };

    template<typename T>
    requires std::is_arithmetic_v<T>
    struct inverse_traits<std::plus<T>, T> {
        static constexpr T of(T a) {
            return -a;
        };
    };

    template<typename T>
    requires std::is_floating_point_v<T>
    struct inverse_traits<std::multiplies<T>, T> {
        static constexpr std::optional<T> of(T a) {
            if (a == T(0)) return std::nullopt;
            return T(1) / a;
        };
    };

    template<typename T>
    requires std::is_floating_point_v<T>
    struct r_div_traits<std::multiplies<T>, T> {
        static constexpr std::optional<T> of(T a, T b) {
            if (b == T(0)) return std::nullopt;
            return a / b;
        };
    };

    template<typename T>
    requires std::is_floating_point_v<T>
    struct l_div_traits<std::multiplies<T>, T> {
        static constexpr std::optional<T> of(T a, T b) {
            if (b == T(0)) return std::nullopt;
            return a / b;
        };
    };

    template<typename F, typename T>
    requires std::is_arithmetic_v<T>
    struct associative<F, T> : std::true_type {};

    template<typename F, typename T>
    requires std::is_arithmetic_v<T>
    struct commutative<F, T> : std::true_type {};

    template<typename T>
    requires std::is_arithmetic_v<T>
    struct distributive<std::plus<T>, std::multiplies<T>, T> : std::true_type {};
};
