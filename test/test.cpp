#include "absalg/absalg.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <absalg/traits.hpp>

using namespace halogen;

static_assert(Field<double>);
static_assert(Field<float>);

static_assert(!Field<int>);
static_assert(CommutativeRing<int>);

static_assert(UnitalRing<int>);
static_assert(UnitalRing<double>);

int main (void) {
    double a = 42.42;
    auto add_id = identity_traits<std::plus<double>, double>::value();
    assert(a+add_id == a);
    auto add_inv = inverse_traits<std::plus<double>, double>::of(a);
    assert(a+add_inv == add_id);
    auto mul_id = identity_traits<std::multiplies<double>, double>::value();
    assert(a*mul_id == a);
    auto mul_inv = inverse_traits<std::multiplies<double>, double>::of(a).value();
    assert(a*mul_inv == mul_id);
    std::cout << "Hello, world\n";
}
