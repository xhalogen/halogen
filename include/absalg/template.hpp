#pragma once
#include <concepts>
#include <type_traits>
#include <functional>

namespace halogen {
    template<typename F, typename T>
    struct identity_traits {
        // static constexpr T value() { return std::nullopt; };
    };

    template<typename F, typename T>
    struct inverse_traits {
        // static constexpr std::optional<T> of(T a) { return std::nullopt; };
    };

    template<typename F, typename T>
    struct r_div_traits {
        // static constexpr std::optional<T> of(T a, T b) { return std::nullopt; };
    };

    template<typename F, typename T>
    struct l_div_traits {
        // static constexpr std::optional<T> of(T a, T b) { return std::nullopt; };
    };

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
    inline constexpr bool associative_v = identifiable<F, T>::value;

    template<typename F, typename T>
    struct commutative {
        static constexpr bool value = false;
    };

    template<typename F, typename T>
    inline constexpr bool commutative_v = identifiable<F, T>::value;

    template<typename FA, typename FM, typename T>
    struct distributive {
        static constexpr bool value = false;
    };

    template<typename FA, typename FM, typename T>
    inline constexpr bool distributive_v = distributive<FA, FM, T>::value;

    // Magmas

    template<typename F, typename T>
    concept Magma = std::invocable<F, T, T>
        && std::convertible_to<std::invoke_result_t<F, T, T>, T>;

    template<typename F, typename T>
    concept UnitalMagma = Magma<F, T> && identifiable_v<F, T>;

    template<typename F, typename T>
    concept QuasiGroup = Magma<F, T> && divisable_v<F, T>;

    template<typename F, typename T>
    concept Loop = QuasiGroup<F, T> && identifiable_v<F, T>;

    template<typename F, typename T>
    concept SemiGroup = Magma<F, T> && associative_v<F, T>;

    template<typename F, typename T>
    concept Monoid = SemiGroup<F, T> && identifiable_v<F, T>;

    template<typename F, typename T>
    concept Group =
        (Monoid<F, T> && invertible_v<F, T>) ||
        (Loop<F, T> && associative_v<F, T>);

    template<typename F, typename T>
    concept CommutativeGroup = Group<F, T> && commutative_v<F, T>;

    // Generic Rings

    template<typename FA, typename FM, typename T>
    concept GenericRing =
        CommutativeGroup<FA, T> &&
        SemiGroup<FM, T> &&
        distributive_v<FA, FM, T>;

    template<typename FA, typename FM, typename T>
    concept GenericCommutativeRing = GenericRing<FA, FM, T> && commutative_v<FM, T>;

    template<typename FA, typename FM, typename T>
    concept GenericUnitalRing = GenericRing<FA, FM, T> && identifiable_v<FM, T>;

    template<typename FA, typename FM, typename T>
    concept GenericDivisionRing = GenericUnitalRing<FA, FM, T> && invertible_v<FM, T>;

    template<typename FA, typename FM, typename T>
    concept GenericField = GenericDivisionRing<FA, FM, T> && GenericCommutativeRing<FA, FM, T>;

    // Rings

    template<typename T>
    concept Ring = GenericRing<std::plus<T>, std::multiplies<T>, T>;

    template<typename T>
    concept CommutativeRing = GenericCommutativeRing<std::plus<T>, std::multiplies<T>, T>;

    template<typename T>
    concept UnitalRing = GenericUnitalRing<std::plus<T>, std::multiplies<T>, T>;

    template<typename T>
    concept DivisionRing = GenericDivisionRing<std::plus<T>, std::multiplies<T>, T>;

    template<typename T>
    concept Field = GenericField<std::plus<T>, std::multiplies<T>, T>;
};
