use num_traits::Float;
use num_traits::NumCast;
use num_traits::One;
use num_traits::Zero;

use crate::minimizer::Minimizer;
use crate::vector::Array;
use crate::vector::Vector;

use std::cmp::Ordering;
use std::ops::Mul;

#[derive(Clone)]
pub(crate) struct Pair<A: Array> {
    pub(crate) f: A::Item,
    pub(crate) x: Vector<A>,
}

impl<A: Array> Pair<A> {
    pub(crate) fn new(f: A::Item, x: Vector<A>) -> Pair<A> {
        Pair { f, x }
    }
}

pub(crate) struct Simplex<A: Array> {
    pairs: Vec<Pair<A>>,
    dim: usize,
    inv_dim: A::Item,
}

impl<A: Array> Simplex<A> {
    pub(crate) fn new<F>(slice: &[A::Item], mut f: F, minimizer: &Minimizer<A::Item>) -> Simplex<A>
    where
        F: FnMut(&[A::Item]) -> A::Item,
        A::Item: Float,
    {
        let dim = <A::Item as NumCast>::from(slice.len()).unwrap();
        let inv_dim = dim.recip();
        let x0 = Vector::<A>::from_slice(slice);

        let mut pairs = Vec::new();
        let pair = Pair::new(f(&x0), x0.clone());
        pairs.push(pair);

        for (idx, _) in x0.iter().enumerate() {
            let mut x = x0.clone();

            {
                let xi = &mut x[idx];
                *xi = if xi.is_zero() {
                    minimizer.step_zero
                } else {
                    *xi * (A::Item::one() + minimizer.step)
                };
            }

            let pair = Pair::new(f(&x), x.clone());
            pairs.push(pair);
        }

        Simplex {
            pairs,
            dim: slice.len(),
            inv_dim,
        }
    }

    pub(crate) fn centroid(&self) -> Vector<A>
    where
        A::Item: Float,
    {
        self.pairs
            .iter()
            .rev()
            .skip(1)
            .map(|&Pair { ref x, .. }| x)
            .fold(Vector::zeros(self.dim), |acc, x| acc + x)
            .mul(self.inv_dim)
    }

    pub(crate) fn sort_unstable(&mut self)
    where
        A::Item: Float,
    {
        self.pairs
            .sort_unstable_by(|a, b| a.f.partial_cmp(&b.f).unwrap_or(Ordering::Equal));
    }

    /// Gives the best estimation,
    /// but it requires to call `sort_unstable` once before.
    pub(crate) fn best(&self) -> Option<&Pair<A>> {
        self.pairs.first()
    }

    /// Gives the worst estimation,
    /// but it requires to call `sort_unstable` once before.
    pub(crate) fn worst(&self) -> Option<&Pair<A>> {
        self.pairs.last()
    }

    /// Gives the second worst estimation,
    /// but it requires to call `sort_unstable` once before.
    pub(crate) fn second_worst(&self) -> Option<&Pair<A>> {
        let (_last, rest) = self.pairs.split_last()?;
        let (second_to_last, _rest) = rest.split_last()?;
        Some(second_to_last)
    }

    pub(crate) fn shrink<F>(&mut self, mut f: F, minimizer: &Minimizer<A::Item>)
    where
        F: FnMut(&[A::Item]) -> A::Item,
        A::Item: Float,
    {
        let best = self.best().unwrap().x.clone();
        for pair in self.pairs.iter_mut().skip(1) {
            pair.x = pair.x.clone() * minimizer.d;
            pair.x.scaled_add(A::Item::one() - minimizer.d, &best);
            pair.f = f(&pair.x);
        }
    }

    pub(crate) fn update(&mut self, pair: Pair<A>) {
        self.pairs.pop();
        self.pairs.push(pair);
    }
}
