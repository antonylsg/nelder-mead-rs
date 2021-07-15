use num_traits::Float;
use num_traits::NumCast;

use crate::simplex::Pair;
use crate::simplex::Simplex;
use crate::vector::Array;

use std::cmp::Ordering;

/// Maximal iteration reached.
#[derive(Debug)]
pub struct MaxIterError(pub usize);

impl std::fmt::Display for MaxIterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Maximal iteration ({}) reached", self.0)
    }
}

impl std::error::Error for MaxIterError {}

/// Output data.
#[derive(Debug)]
pub struct Output<A: Array> {
    pub f_min: A::Item,
    pub x_min: A,
    pub iter: usize,
}

pub type Result<A> = std::result::Result<Output<A>, MaxIterError>;

/// A structure that holds all the minimization parameters.
#[derive(Debug)]
pub struct Minimizer<A: Array> {
    // Reflection parameter
    a: A::Item,

    // Contraction parameter
    b: A::Item,

    // Expansion parameter
    c: A::Item,

    // Shrinkage parameter
    pub(crate) d: A::Item,

    // Initialization parameters
    pub(crate) step: A::Item,
    pub(crate) step_zero: A::Item,

    // Tolerance (function) parameter
    tol_f: A::Item,

    // Tolerance (point) parameter
    tol_x: A::Item,

    // Iterations parameter
    max_iter: usize,
}

impl<A: Array> Default for Minimizer<A>
where
    A::Item: Float,
{
    fn default() -> Minimizer<A> {
        Minimizer {
            a: <A::Item as NumCast>::from(1.0).unwrap(),
            b: <A::Item as NumCast>::from(0.5).unwrap(),
            c: <A::Item as NumCast>::from(2.0).unwrap(),
            d: <A::Item as NumCast>::from(0.5).unwrap(),
            step: <A::Item as NumCast>::from(0.01).unwrap(),
            step_zero: <A::Item as NumCast>::from(0.00025).unwrap(),
            tol_f: <A::Item as NumCast>::from(1e-4).unwrap(),
            tol_x: <A::Item as NumCast>::from(1e-4).unwrap(),
            max_iter: 200,
        }
    }
}

impl<A: Array> Minimizer<A>
where
    A::Item: Float,
{
    /// Minimizes the function `f` with the seed `x0`.
    pub fn minimize<F>(&self, x0: &[A::Item], mut f: F) -> Result<A>
    where
        F: FnMut(&A) -> A::Item,
        A::Item: Clone,
    {
        // Init
        let max_iter = x0.len() * self.max_iter;
        let mut simplex = Simplex::new(x0, &mut f, self);

        // Sort
        simplex.sort_unstable();

        for iter in 0..max_iter {
            // Centroid
            let centroid = simplex.centroid();

            // Best
            let fb = simplex.best().unwrap().f;

            // Worst
            let mut worst = simplex.worst().cloned().unwrap();

            // Reflection
            let reflect = {
                let x = &centroid + (&centroid - &worst.x) * self.a;
                Pair::new(f(&x), x)
            };

            // Second-worst
            let fs = simplex.second_worst().unwrap().f;

            // Reflection accepted
            if reflect.f < fs {
                worst = reflect.clone();

                // Expansion
                if reflect.f < fb {
                    let expan = {
                        let x = &centroid + (reflect.x - &centroid) * self.c;
                        Pair::new(f(&x), x)
                    };

                    // Expansion accepted
                    if expan.f < fb {
                        worst = expan;
                    }
                }
            } else {
                // Contraction
                let contr = {
                    let x = if reflect.f < worst.f {
                        // Outside contraction
                        &centroid + (&centroid - &worst.x) * self.b
                    } else {
                        // Inside contraction
                        &centroid + (&worst.x - &centroid) * self.b
                    };
                    Pair::new(f(&x), x)
                };

                // Contraction accepted
                let min = if reflect.f < worst.f {
                    reflect.f
                } else {
                    worst.f
                };
                if contr.f < min {
                    worst = contr;
                } else {
                    // Shrinkage
                    simplex.shrink(&mut f, self);
                }
            }

            // Pull update
            simplex.update(worst);

            // Sort
            simplex.sort_unstable();

            // Termination tests
            let best = simplex.best().unwrap();
            let worst = simplex.worst().unwrap();

            // Domain convergence test
            let mut buf = &worst.x - &best.x;
            buf.iter_mut().for_each(|x| *x = x.abs());
            buf.as_mut()
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let test_x = *buf.as_ref().last().unwrap();

            // Function value convergence test
            let test_f = (worst.f - best.f).abs();

            // Termination test
            if test_f <= self.tol_f && test_x <= self.tol_x {
                return Ok(Output {
                    f_min: best.f,
                    x_min: best.x.0.clone(),
                    iter,
                });
            }
        }

        Err(MaxIterError(max_iter))
    }
}
