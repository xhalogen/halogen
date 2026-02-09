#pragma once
#include "template.hpp"
#include <optional>
#include <functional>
#include <concepts>
#include <type_traits>

namespace halogen {
    namespace detail {
        template<typename T>
        concept arithmetic_number = std::is_arithmetic_v<T>;
    }

    template<detail::arithmetic_number T>
    struct identity_traits<std::plus<T>, T> {
        static constexpr T value() { return T(0); };
    };

    template<detail::arithmetic_number T>
    struct identity_traits<std::multiplies<T>, T> {
        static constexpr T value() { return T(1); };
    };

    template<detail::arithmetic_number T>
    struct inverse_traits<std::plus<T>, T> {
        static constexpr T of(T a) {
            return -a;
        };
    };

    template<std::floating_point T>
    struct inverse_traits<std::multiplies<T>, T> {
        static constexpr std::optional<T> of(T a) {
            if (a == T(0)) return std::nullopt;
            return T(1) / a;
        };
    };

    template<std::floating_point T>
    struct r_div_traits<std::multiplies<T>, T> {
        static constexpr std::optional<T> of(T a, T b) {
            if (b == T(0)) return std::nullopt;
            return a / b;
        };
    };

    template<std::floating_point T>
    struct l_div_traits<std::multiplies<T>, T> {
        static constexpr std::optional<T> of(T a, T b) {
            if (b == T(0)) return std::nullopt;
            return a / b;
        };
    };

    template<typename F, detail::arithmetic_number T>
    struct associative<F, T> : std::true_type {};

    template<typename F, detail::arithmetic_number T>
    struct commutative<F, T> : std::true_type {};

    template<detail::arithmetic_number T>
    struct distributive<std::plus<T>, std::multiplies<T>, T> : std::true_type {};
};
