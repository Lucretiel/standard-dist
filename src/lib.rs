/*!
`standard-dist` is a library for automatically deriving a `rand` standard
distribution for your types via a derive macro.

# Usage examples

```
use rand::distributions::Uniform;
use standard_dist::StandardDist;

// Select heads or tails with equal probability
#[derive(Debug, Clone, Copy, PartialEq, Eq, StandardDist)]
enum Coin {
    Heads,
    Tails,
}

// Flip 3 coins, independently
#[derive(Debug, Clone, Copy, PartialEq, Eq, StandardDist)]
struct Coins {
    first: Coin,
    second: Coin,
    third: Coin,
}

// Use the `#[distribution]` attribute to customize the distribution used on
// a field
#[derive(Debug, Clone, Copy, PartialEq, Eq, StandardDist)]
struct Die {
    #[distribution(Uniform::from(1..=6))]
    value: u8
}

// Use the `#[weight]` attribute to customize the relative probabilities of
// enum variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, StandardDist)]
enum D20 {
    #[weight(18)]
    Normal,

    Critical,
    CriticalFail,
}
```

[`rand`] generates typed random values via the [`Distribution`] trait, which
uses a [source of randomness] to produce values of the given type. Of particular
note is the [`Standard`] distribution, which is the stateless "default" way to
produce random values of a particular type. For instance:
- For ints, this randomly chooses from all possible values for that int type
- For bools, it chooses true or false with 50/50 probability
- For `Option<T>`, it chooses `None` or `Some` with 50/50 probability, and uses
  [`Standard`] to randomly populate the inner `Some` value.

# Structs

When you derive `StandardDist` for one of your own structs, it creates an
`impl Distribution<YourStruct> for Standard` implementation, allowing you to
create randomized instances of the struct via [`Rng::gen`]. This implementation
will in turn use the `Standard` distribution to populate all the fields of
your type.

```rust
use standard_dist::StandardDist;

#[derive(StandardDist)]
struct SimpleStruct {
    coin: bool,
    percent: f64,
}

let mut heads = 0;

for _ in 0..2000 {
    let s: SimpleStruct = rand::random();
    assert!(0.0 <= s.percent);
    assert!(s.percent < 1.0);
    if s.coin {
        heads += 1;
    }
}

assert!(900 < heads, "heads: {}", heads);
assert!(heads < 1100, "heads: {}", heads);
```

## Custom Distributions

You can customize the distribution used for any field with the `#[distribution]`
attribute:

```rust
use std::collections::HashMap;
use standard_dist::StandardDist;
use rand::distributions::Uniform;

#[derive(StandardDist)]
struct Die {
    #[distribution(Uniform::from(1..=6))]
    value: u8
}

let mut counter: HashMap<u8, u32> = HashMap::new();

for _ in 0..6000 {
    let die: Die = rand::random();
    *counter.entry(die.value).or_insert(0) += 1;
}

assert_eq!(counter.len(), 6);

for i in 1..=6 {
    let count = counter[&i];
    assert!(900 < count, "{}: {}", i, count);
    assert!(count < 1100, "{}: {}", i, count);
}
```

# Enums

When applied to an enum type, the implementation will randomly select a variant
(where each variant has an equal probability) and then populate all the fields
of that variant in the same manner as with a struct. Enum variant fields may
have custom distributions applied via `#[distribution]`, just like struct
fields.

```rust
use standard_dist::StandardDist;

#[derive(PartialEq, Eq, StandardDist)]
enum Coin {
    Heads,
    Tails,
}

let mut heads = 0;

for _ in 0..2000 {
    let coin: Coin = rand::random();
    if coin == Coin::Heads {
        heads += 1;
    }
}

assert!(900 < heads, "heads: {}", heads);
assert!(heads < 1100, "heads: {}", heads);
```

## Weights

Enum variants may be weighted with the `#[weight]` attribute to make them
relatively more or less likely to be randomly selected. A weight of 0 means
that the variant will never be selected. Any untagged variants will have a
weight of 1.

```rust
use standard_dist::StandardDist;

#[derive(StandardDist)]
enum D20 {
    #[weight(18)]
    Normal,

    CriticalHit,
    CriticalMiss,
}

let mut crits = 0;

for _ in 0..20000 {
    let roll: D20 = rand::random();
    if matches!(roll, D20::CriticalHit) {
        crits += 1;
    }
}

assert!(900 < crits, "crits: {}", crits);
assert!(crits < 1100, "crits: {}", crits);
```

# Advanced custom distributions

## Distribution types

You may optionally explicitly specify a type for your distributions; this can
sometimes be necessary when using generic types.

```rust
use std::collections::HashMap;
use standard_dist::StandardDist;
use rand::distributions::Uniform;

#[derive(StandardDist)]
struct Die {
    #[distribution(Uniform<u8> = Uniform::from(1..=6))]
    value: u8
}

let mut counter: HashMap<u8, u32> = HashMap::new();

for _ in 0..6000 {
    let die: Die = rand::random();
    *counter.entry(die.value).or_insert(0) += 1;
}

assert_eq!(counter.len(), 6);

for i in 1..=6 {
    let count = counter[&i];
    assert!(900 < count, "{}: {}", i, count);
    assert!(count < 1100, "{}: {}", i, count);
}
```

## Distribution caching

In some cases, you may wish to cache a `Distribution` instance for reuse. Many
distributions perform some initial calculations when constructed, and it can
help performance to reuse existing distributions rather than recreate them
every time a value is generated. `standard-dist` provides to ways to cache
distributions: `static` and `once`. A `static` distribution is stored as a
global static variable; this is the preferable option, but it requires the
initializer to be usable in a `const` context. A `once` distribution is stored
in a `once_cell::sync::OnceCell`; it is initialized the first time it's used,
and then reused on subsequent invocations.

In either case, a cache policy is specified by prefixing the type with `once` or
`static`. The type must be specified in order to use a cache policy.

```rust
use std::collections::HashMap;
use std::time::{Instant, Duration};
use standard_dist::StandardDist;
use rand::prelude::*;
use rand::distributions::Uniform;

#[derive(StandardDist)]
struct Die {
    #[distribution(Uniform::from(1..=6))]
    value: u8
}

#[derive(StandardDist)]
struct CachedDie {
    #[distribution(once Uniform<u8> = Uniform::from(1..=6))]
    value: u8
}

fn timed<T>(task: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    (task(), start.elapsed())
}

// Count the 6s
let mut rng = StdRng::from_entropy();

let (count, plain_die_duration) = timed(|| (0..600000)
    .map(|_| rng.gen())
    .filter(|&Die{ value }| value == 6)
    .count()
);

assert!(90000 < count);
assert!(count < 110000);

let (count, cache_die_duration) = timed(|| (0..600000)
    .map(|_| rng.gen())
    .filter(|&CachedDie{ value }| value == 6)
    .count()
);

assert!(90000 < count);
assert!(count < 110000);

assert!(
    cache_die_duration < plain_die_duration,
    "cache: {:?}, plain: {:?}",
    cache_die_duration,
    plain_die_duration,
);
```

Note that, unless you're generating a huge quantity of random objects, using
`cell` is likely a pessimization because of the upfront cost to initializing
the cell. Make sure to benchmark your specific use case if performance is a
concern.


[`rand`]: https://docs.rs/rand/
[`Distribution`]: https://docs.rs/rand/latest/rand/distributions/trait.Distribution.html
[`Standard`]: https://docs.rs/rand/latest/rand/distributions/struct.Standard.html
[source of randomness]: https://docs.rs/rand/latest/rand/trait.Rng.html
[`Rng::gen`]: https://docs.rs/rand/latest/rand/trait.Rng.html#method.gen
*/
use std::{collections::HashSet, iter};

use itertools::Itertools;
use parse::ParseStream;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parse,
    parse::{discouraged::Speculative, Parse},
    parse_quote,
    spanned::Spanned,
    DeriveInput, Error, Expr, Field, Fields, LitInt, Token, Type, Variant,
};

/// A particular field type, paired with the type of the distribution used
/// to produce it. Used to create `where` bindings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FieldDistributionBinding<'a> {
    field_type: &'a Type,
    distribution_type: Type,
}

/// Given a list of fields (as from a struct or enum variant), return a list
/// of all the types of those fields, paired with the associated distribution
/// types.
fn fields_types(fields: &Fields) -> impl Iterator<Item = syn::Result<FieldDistributionBinding>> {
    fields.iter().filter_map(|field| {
        field_distribution(field)
            .map(|spec| {
                spec.container.map(|container| FieldDistributionBinding {
                    field_type: &field.ty,
                    distribution_type: container.ty,
                })
            })
            .transpose()
    })
}

/// Given a type definition- a struct or enum- return an iterator over
/// all the types of all the fields in that type, paired with the associated
/// distribution types.
fn item_subtypes(
    input: &DeriveInput,
) -> Box<dyn Iterator<Item = syn::Result<FieldDistributionBinding<'_>>> + '_> {
    match &input.data {
        syn::Data::Struct(data) => Box::new(fields_types(&data.fields)),
        syn::Data::Enum(data) => Box::new(
            data.variants
                .iter()
                .flat_map(|variant| fields_types(&variant.fields)),
        ),
        syn::Data::Union(_) => Box::new(iter::empty()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldDistributionStorage {
    Local,
    Once,
    Static,
}

impl Parse for FieldDistributionStorage {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        use FieldDistributionStorage::*;

        input.step(|cursor| match cursor.ident() {
            Some((ident, tail)) if ident == "static" => Ok((Static, tail)),
            Some((ident, tail)) if ident == "once" => Ok((Once, tail)),
            _ => Ok((Local, *cursor)),
        })
    }
}

#[derive(Debug, Clone)]
struct FieldDistributionContainer {
    ty: Type,
    storage: FieldDistributionStorage,
}

#[derive(Debug, Clone)]
struct FieldDistributionSpec {
    init: Expr,
    container: Option<FieldDistributionContainer>,
}

impl Parse for FieldDistributionSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let storage: FieldDistributionStorage = input.parse()?;

        if storage == FieldDistributionStorage::Local {
            // There was no storage specifier. Try to parse `type =`, but
            // fall back to just an expression.
            let input_with_type = input.fork();

            if let Ok(ty) = input_with_type.parse() {
                if let Ok(_eq) = input_with_type.parse::<Token![=]>() {
                    // We got "type =", so proceed unconditionally this way
                    input.advance_to(&input_with_type);
                    let original = input.fork();
                    let init = input.parse().map_err(|_| {
                        Error::new(original.span(), "expected a distribution expression")
                    })?;
                    return Ok(FieldDistributionSpec {
                        init,
                        container: Some(FieldDistributionContainer { ty, storage }),
                    });
                }
            }

            let original = input.fork();

            // Failed to parse "type =". Attempt to just parse the expression.
            input
                .parse()
                .map(|init| FieldDistributionSpec {
                    init,
                    container: None,
                })
                .map_err(|_| Error::new(original.span(), "expected a distribution expression"))
        } else {
            // If we had a storage specifier, we now must have a type
            let ty = input
                .parse()
                .map_err(|_| Error::new(input.span(), "expected a distribution type"))?;
            let _equals: Token![=] = input.parse()?;
            let init = input
                .parse()
                .map_err(|_| Error::new(input.span(), "expected a distribution expression"))?;
            Ok(FieldDistributionSpec {
                init,
                container: Some(FieldDistributionContainer { ty, storage }),
            })
        }
    }
}

/// Given a field, look at the #[distribution] attribute of the field to
/// determine what distribution should be used. Returns the Standard
/// distribution if there is no such attribute. The returned token stream
/// should be an expression which can be passed to rng.sample.
fn field_distribution(field: &Field) -> syn::Result<FieldDistributionSpec> {
    match field
        .attrs
        .iter()
        .find(|attr| attr.path.is_ident("distribution"))
    {
        None => Ok(FieldDistributionSpec {
            init: parse_quote! {::rand::distributions::Standard},
            container: Some(FieldDistributionContainer {
                ty: parse_quote! {::rand::distributions::Standard},
                storage: FieldDistributionStorage::Local,
            }),
        }),
        Some(attr) => attr.parse_args(),
    }
}

/// Given a list of fields, create a comma-separated series of initializers
/// suited for initializing a type containing those fields. Return something
/// resembling "field1: value1, field2: value2," for fields with names, and
/// "value1, value2," for fields without names.
///
/// The initializers are specifically the invocations of
/// `rng.sample(distribution)`.
fn field_inits<'a>(
    rng: &Ident,
    fields: impl Iterator<Item = &'a Field>,
) -> syn::Result<TokenStream2> {
    fields
        .map(|field| {
            let field_type = &field.ty;
            let distribution = field_distribution(&field)?;
            let (dist_ty, dist_init) = match distribution.container {
                None => (parse_quote! {_}, distribution.init),
                Some(container) => {
                    let ty = container.ty;
                    let init = distribution.init;

                    match container.storage {
                        FieldDistributionStorage::Local => (ty, init),
                        FieldDistributionStorage::Once => (
                            parse_quote! {&'static #ty},
                            parse_quote! {{
                                static DISTRIBUTION: ::once_cell::sync::OnceCell<#ty> =
                                    ::once_cell::sync::OnceCell::new();

                                DISTRIBUTION.get_or_init(move || #init)
                            }},
                        ),
                        FieldDistributionStorage::Static => (
                            parse_quote! {&'static #ty},
                            parse_quote! {{
                                static DISTRIBUTION: #ty = #init;

                                &DISTRIBUTION
                            }},
                        ),
                    }
                }
            };

            let init = quote! { ::rand::Rng::sample::<#field_type, #dist_ty>(#rng, #dist_init), };
            Ok(match &field.ident {
                Some(field_ident) => quote! { #field_ident: #init },
                None => init,
            })
        })
        .collect()
}

/// Create a literal expression initializing a value of the given `type`
/// consisting of the given fields. Used to create expressions to initialize
/// structs and enum variants.
fn init_value_of_type(
    type_path: TokenStream2,
    rng: &Ident,
    fields: &Fields,
) -> syn::Result<TokenStream2> {
    match fields {
        Fields::Named(fields) => {
            let field_inits = field_inits(rng, fields.named.iter())?;

            Ok(quote! {
                #type_path {
                    #field_inits
                }
            })
        }
        Fields::Unnamed(fields) => {
            let field_inits = field_inits(rng, fields.unnamed.iter())?;

            Ok(quote! {
                #type_path (
                    #field_inits
                )
            })
        }
        Fields::Unit => Ok(type_path),
    }
}

/// Look at the #[weight] attribute of an enum variant to determine what weight
/// it should be given in random generation. Returns 1 if there is no such
/// attribute, or an error if the attribute is malformed.
fn enum_variant_weight(variant: &Variant) -> syn::Result<u64> {
    match variant
        .attrs
        .iter()
        .find(|attr| attr.path.is_ident("weight"))
    {
        None => Ok(1),
        Some(attr) => attr.parse_args::<LitInt>()?.base10_parse(),
    }
}

/// Similar to `try!`, this macro wraps a `syn::Result`, and converts the
/// error to a compile error and returns it in the event of an error.
macro_rules! syn_unwrap {
    ($input:expr) => {
        match ($input) {
            Ok(value) => value,
            Err(err @ syn::Error { .. }) => return err.into_compile_error().into(),
        }
    };
}

#[proc_macro_derive(StandardDist, attributes(weight, distribution))]
pub fn standard_dist(item: TokenStream) -> TokenStream {
    let input: DeriveInput = match parse(item) {
        Ok(input) => input,
        Err(err) => return err.into_compile_error().into(),
    };

    let type_ident = &input.ident;
    let rng = Ident::new("rng", Span::mixed_site());

    let sample_body = match &input.data {
        syn::Data::Struct(data) => syn_unwrap!(init_value_of_type(
            type_ident.to_token_stream(),
            &rng,
            &data.fields
        )),
        syn::Data::Enum(data) => {
            // The total weights that have been accumulated for all variants.
            let mut cumulative_weight = Some(0u64);

            // TODO: There's enough weird control flow and statefulness here
            // that it should probably be a plain for loop. The problem,
            // ironically, is that it's actually easier to use an iterator
            // chain, because we can use `?`. This should all be refactored
            // into a function returning a syn::Result.
            let match_arms = data
                .variants
                .iter()
                // For each variant, compute the weight. The weight is given
                // via a #[weight(10)] annotation, defaulting to 1. May return
                // an error for a malformed annotation.
                .map(|variant| enum_variant_weight(variant).map(|weight| (variant, weight)))
                // Skip variants with a weight of 0.
                .filter_ok(|&(_, weight)| weight != 0)
                // Create a match arm for each variant
                .map(|state| {
                    let (variant, weight) = state?;

                    // Process the cumulative weights. Compute the inclusive lower
                    // and upper bounds for this variant, and update the cumulative
                    // weight.
                    let lower_bound = cumulative_weight.ok_or_else(|| {
                        Error::new(variant.span(), "enum variant weight overflow")
                    })?;
                    let upper_bound = lower_bound.checked_add(weight - 1).ok_or_else(|| {
                        Error::new(variant.span(), "enum variant weight overflow")
                    })?;
                    cumulative_weight = upper_bound.checked_add(1);

                    // Create a match arm for each variant
                    let variant_ident = &variant.ident;
                    let variant_path = quote! {#type_ident::#variant_ident};
                    let gen_variant = init_value_of_type(variant_path, &rng, &variant.fields)?;
                    let pattern = quote! {#lower_bound ..= #upper_bound};
                    Ok(quote! {#pattern => #gen_variant,})
                })
                .collect();

            let match_arms: TokenStream2 = syn_unwrap!(match_arms);

            // In the likely event that we didn't use an entire u64's worth of
            // weights, create a trailing catch-all arm with an `unreachable`
            let trailing_arm = cumulative_weight.map(|cumulative_weight| {
                quote! {
                    n => ::std::unreachable!(
                        "The enum {} only has {} total weight, but the rng returned {}",
                        ::std::stringify!(#type_ident),
                        #cumulative_weight,
                        n
                    ),
                }
            });

            // Create the expression that actually produces a random integer
            // which is used to randomly select a variant.
            let gen_variant_selector = match cumulative_weight {
                None => quote! { ::rand::Rng::gen(#rng) },
                Some(0) => {
                    return Error::new(
                        input.span(),
                        match data.variants.len() {
                            0 => "cannot derive StandardDist for empty enums",
                            _ => "must have at least one variant with a nonzero weight",
                        },
                    )
                    .into_compile_error()
                    .into()
                }
                Some(upper_bound) => quote! { ::rand::Rng::gen_range(#rng, 0u64..#upper_bound) },
            };

            quote! {
                match #gen_variant_selector {
                    #match_arms
                    #trailing_arm
                }
            }
        }
        syn::Data::Union(..) => {
            return Error::new(input.span(), "cannot derive `StandardDist` on a union")
                .into_compile_error()
                .into()
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let where_clause = if !input.generics.params.is_empty() {
        let type_bindings: HashSet<FieldDistributionBinding> =
            syn_unwrap!(item_subtypes(&input).collect());

        let type_bindings = type_bindings.iter().map(
            |FieldDistributionBinding {
                 field_type,
                 distribution_type,
             }| quote!( #distribution_type: ::rand::distributions::Distribution<#field_type> ),
        );

        let type_bindings = type_bindings.chain(
            where_clause
                .into_iter()
                .flat_map(|clause| clause.predicates.iter().map(|pred| pred.to_token_stream())),
        );

        quote! {where #(#type_bindings),*}
    } else {
        quote! {#where_clause}
    };

    let distribution_impl = quote! {
        impl #impl_generics ::rand::distributions::Distribution<#type_ident #ty_generics > for ::rand::distributions::Standard
            #where_clause
        {
            fn sample<R: ::rand::Rng + ?::std::marker::Sized>(&self, #rng: &mut R) -> #type_ident #ty_generics {
                #sample_body
            }
        }
    };

    distribution_impl.into()
}
