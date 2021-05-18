use std::collections::HashMap;

use itertools::Itertools;
use rand::distributions::Uniform;
use rand::prelude::*;
use standard_dist::StandardDist;

#[derive(Debug, Clone, Copy, PartialEq, Eq, StandardDist)]
enum Coin {
    Heads,
    Tails,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, StandardDist)]
struct Die<const N: u32> {
    #[distribution(Uniform::new(0, N))]
    value: u32,
}

#[derive(Debug, Clone, Copy, StandardDist)]
enum Fancy {
    Unit,

    Field(Coin),

    Pair(Coin, Die<6>),

    #[weight(3)]
    WeightedStruct {
        coin: Coin,
        d8: Die<8>,
        d12: Die<12>,

        #[distribution(once Uniform<f64> = Uniform::new(-1.0, 1.0))]
        float: f64,
    },
}

#[test]
fn test_fancy() {
    let mut rng = StdRng::from_entropy();

    let mut d6_counts: HashMap<u32, u32> = HashMap::new();
    let mut d8_counts: HashMap<u32, u32> = HashMap::new();
    let mut d12_counts: HashMap<u32, u32> = HashMap::new();

    let mut heads_count = 0;
    let mut tails_count = 0;

    // There should be 50/50 weighted struct vs everything else
    let mut weighted_count = 0;

    for _ in 0..100000 {
        let fancy: Fancy = rng.gen();

        match fancy {
            Fancy::Unit => {}
            Fancy::Field(coin) => match coin {
                Coin::Heads => heads_count += 1,
                Coin::Tails => tails_count += 1,
            },
            Fancy::Pair(coin, d6) => {
                match coin {
                    Coin::Heads => heads_count += 1,
                    Coin::Tails => tails_count += 1,
                }
                *d6_counts.entry(d6.value).or_default() += 1;
            }
            Fancy::WeightedStruct {
                coin,
                d8,
                d12,
                float,
            } => {
                weighted_count += 1;
                match coin {
                    Coin::Heads => heads_count += 1,
                    Coin::Tails => tails_count += 1,
                }
                *d8_counts.entry(d8.value).or_default() += 1;
                *d12_counts.entry(d12.value).or_default() += 1;
                assert!(-1.0 <= float);
                assert!(float < 1.0);
            }
        }
    }

    // Heuristic: min == 90% of max at worst

    let unweighted_count = 100000 - weighted_count;
    let weight_ratio = weighted_count as f64 / unweighted_count as f64;
    assert!(0.9 < weight_ratio);
    assert!(weight_ratio < 1.1);

    let coin_ratio = heads_count as f64 / tails_count as f64;
    assert!(0.9 < coin_ratio);
    assert!(coin_ratio < 1.1);

    assert_eq!(d6_counts.len(), 6);
    assert_eq!(d8_counts.len(), 8);
    assert_eq!(d12_counts.len(), 12);

    for counts in &[d6_counts, d8_counts, d12_counts] {
        let (min, max) = counts.values().copied().minmax().into_option().unwrap();
        let ratio = min as f64 / max as f64;
        assert!(0.9 < ratio);
        assert!(ratio <= 1.0);
    }
}

#[derive(Debug, Clone, StandardDist)]
enum Generics<T, U> {
    Unit,
    One(u32),
    Gen(T),
    Pair(T, U),
    Group { t: T, u: U, f: f64 },
}

#[test]
fn test_generic() {
    let _gen: Generics<Coin, Die<6>> = random();
    // TODO: fill in tests. For now we're content that if it compiles, it
    // works.
}
