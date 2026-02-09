#pragma once
#include "template.hpp"
#include <functional>
#include <concepts>

namespace halogen {
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
}
