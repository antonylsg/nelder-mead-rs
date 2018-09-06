extern crate num_traits;

use self::num_traits::Float;

use simplex::Pair;
use simplex::Simplex;

use std::cmp::Ordering;
use std::fmt;
use std::ops::MulAssign;
use std::result;

/// A custom Error for `Minimizer`.
#[derive(Debug)]
pub struct MaxIterError(usize);

impl fmt::Display for MaxIterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Maximal iteration ({}) reached", self.0)
    }
}

type Result<T> = result::Result<Output<T>, MaxIterError>;

/// Output data.
#[derive(Debug)]
pub struct Output<T> {
    pub f_min: T,
    pub x_min: Vec<T>,
    pub iter: usize,
}

/// A structure that holds all the minimization parameters.
#[derive(Debug)]
pub struct Minimizer<T> {
    // Reflection parameter
    a: T,

    // Contraction parameter
    b: T,

    // Expansion parameter
    c: T,

    // Shrinkage parameter
    pub(crate) d: T,

    // Initialization parameters
    pub(crate) step: T,
    pub(crate) step_zero: T,

    // Tolerance (function) parameter
    tol_f: T,

    // Tolerance (point) parameter
    tol_x: T,

    // Iterations parameter
    max_iter: usize,
}

impl<T: Float> Default for Minimizer<T> {
    fn default() -> Minimizer<T> {
        Minimizer {
            a: T::from(1.0).unwrap(),
            b: T::from(0.5).unwrap(),
            c: T::from(2.0).unwrap(),
            d: T::from(0.5).unwrap(),
            step: T::from(0.01).unwrap(),
            step_zero: T::from(0.00025).unwrap(),
            tol_f: T::from(1e-4).unwrap(),
            tol_x: T::from(1e-4).unwrap(),
            max_iter: 200,
        }
    }
}

impl<T> Minimizer<T>
where
    T: Float,
{
    /// Minimizes the function `f` with the seed `x0`.
    pub fn minimize<F>(&self, x0: &[T], mut f: F) -> Result<T>
    where
        F: FnMut(&[T]) -> T,
        T: Clone + MulAssign,
    {
        // Init
        let max_iter = x0.len() * self.max_iter;
        let mut simplex = Simplex::<[T; 8]>::new(x0, &mut f, self);

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
            buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let test_x = *buf.last().unwrap();

            // Function value convergence test
            let test_f = (worst.f - best.f).abs();

            // Termination test
            if test_f <= self.tol_f && test_x <= self.tol_x {
                return Ok(Output {
                    f_min: best.f,
                    x_min: best.x.to_vec(),
                    iter,
                });
            }
        }

        Err(MaxIterError(max_iter))
    }
}
