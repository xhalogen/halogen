use paste::paste;
use quote::format_ident;
use syn::Ident;

const HALOGEN_PREFIX: &str = "__halogen_einsum_";

macro_rules! def_get_halogen_einsum_i {
    ($name: ident) => {
        paste! {
            pub fn [<get_ $name _i>](i: usize) -> Ident {
                format_ident!("{}{}{}", HALOGEN_PREFIX, stringify!($name), i)
            }
        }
    };
}

macro_rules! def_get_halogen_einsum_v {
    ($name: ident) => {
        paste! {
            pub fn [<get_ $name _v>]() -> Ident {
                format_ident!("{}{}", HALOGEN_PREFIX, stringify!($name))
            }
        }
    };
}

def_get_halogen_einsum_i!(stride);
def_get_halogen_einsum_i!(shape);
def_get_halogen_einsum_i!(inner);
def_get_halogen_einsum_i!(outer);
def_get_halogen_einsum_i!(input);
def_get_halogen_einsum_v!(offset);
def_get_halogen_einsum_v!(shape);
def_get_halogen_einsum_v!(acc);
def_get_halogen_einsum_v!(tmp);
def_get_halogen_einsum_v!(data);
def_get_halogen_einsum_v!(from_vec);
pub fn get_t_v() -> Ident {
    format_ident!("__HALOGEN_EINSUM_T")
}
