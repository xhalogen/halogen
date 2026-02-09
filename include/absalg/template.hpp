#pragma once
#include <type_traits>

namespace halogen {
    // static constexpr T value() { return std::nullopt; };
    template<typename F, typename T> struct identity_traits;

    // static constexpr std::optional<T> of(T a) { return std::nullopt; };
    template<typename F, typename T> struct inverse_traits;

    // static constexpr std::optional<T> of(T a, T b) { return std::nullopt; };
    template<typename F, typename T> struct r_div_traits;

    // static constexpr std::optional<T> of(T a, T b) { return std::nullopt; };
    template<typename F, typename T> struct l_div_traits;

    template<typename F, typename T, typename = void>
    struct identifiable : std::false_type {};

    template<typename F, typename T>
    struct identifiable<F, T, std::void_t<decltype(identity_traits<F, T>::value())>>
        : std::true_type {};

    template<typename F, typename T>
    inline constexpr bool identifiable_v = identifiable<F, T>::value;

    template<typename F, typename T, typename = void>
    struct invertible : std::false_type {};

    template<typename F, typename T>
    struct invertible<F, T, std::void_t<decltype(inverse_traits<F, T>::of(std::declval<T>()))>>
        : std::true_type {};

    template<typename F, typename T>
    inline constexpr bool invertible_v = invertible<F, T>::value;

    template<typename F, typename T, typename = void>
    struct divisable : std::false_type {};

    template<typename F, typename T>
    struct divisable<F, T, std::void_t<
            decltype(r_div_traits<F, T>::of(std::declval<T>(), std::declval<T>())),
            decltype(l_div_traits<F, T>::of(std::declval<T>(), std::declval<T>()))
        >>
        : std::true_type {};

    template<typename F, typename T>
    inline constexpr bool divisable_v = divisable<F, T>::value;

    // These are just promises
    template<typename F, typename T>
    struct associative {
        static constexpr bool value = false;
    };

    template<typename F, typename T>
    inline constexpr bool associative_v = associative<F, T>::value;

    template<typename F, typename T>
    struct commutative {
        static constexpr bool value = false;
    };

    template<typename F, typename T>
    inline constexpr bool commutative_v = commutative<F, T>::value;

    template<typename FA, typename FM, typename T>
    struct distributive {
        static constexpr bool value = false;
    };

    template<typename FA, typename FM, typename T>
    inline constexpr bool distributive_v = distributive<FA, FM, T>::value;
};
