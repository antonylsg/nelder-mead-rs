extern crate ndarray;

use self::ndarray::Array1;


/// Output data
#[derive(Debug)]
pub struct Output {
    pub f_min: f64,
    pub x_min: Vec<f64>,
    pub iter: usize,
}


/// A structure that holds all the minimization parameters
#[derive(Debug)]
pub struct Minimizer {
    /// Reflection parameter
    a: f64,
    
    /// Contraction parameter
    b: f64,

    /// Expansion parameter
    c: f64,

    /// Shrinkage parameter
    d: f64,

    /// Initialization parameter
    step: f64,
    step_zero: f64,

    /// Tolerance (function) parameter
    tol_f: f64,

    /// Tolerance (point) parameter
    tol_x: f64,

    /// Iterations parameter
    max_iter: usize,
}

impl Default for Minimizer {
    fn default() -> Minimizer {
        Minimizer {
            a: 1.0,
            b: 0.5,
            c: 2.0,
            d: 0.5,
            step: 0.01,
            step_zero: 0.00025,
            tol_f: 1e-4,
            tol_x: 1e-4,
            max_iter: 200,
        }
    }
}

impl Minimizer {
    /// Minimize the function `func` with the seed `x0`
    pub fn minimize<F>(&self, func: F, x0: Vec<f64>) -> Result<Output, ()>
        where F: Fn(Vec<f64>) -> f64 {

        use std::cmp::Ordering;


        // Init
        let x0 = Array1::from_vec(x0);
        let inv_dim = (x0.dim() as f64).recip();
        let mut pairs = Vec::new();
        let pair = (func(x0.to_vec()), x0.clone());
        pairs.push(pair);
        
        for (idx, _) in x0.iter().enumerate() {
            let mut x = x0.clone();

            unsafe {
                let xi = x.uget_mut(idx);
                *xi = if *xi == 0.0 {
                    self.step_zero
                }
                else {
                    *xi * (1.0 + self.step)
                };
            }

            let pair = (func(x.to_vec()), x.clone());
            pairs.push(pair);
        }


        for iter in 0 .. x0.dim() * self.max_iter {
            // Sort
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal));
                    
            // Centroid
            let centroid = pairs
                .iter()
                .rev()
                .skip(1)
                .map(|p| &p.1)
                .fold(Array1::zeros(x0.dim()), |acc, x| acc + x) * inv_dim;


            let (mut fw, mut xw) = pairs.last().cloned().unwrap();

            // Reflection
            let xr = (1.0 + self.a) * &centroid - self.a * xw.clone();
            let fr = func(xr.to_vec());
            let fs = pairs.iter().rev().skip(1).rev().last().unwrap().0;

            // Reflection accepted
            if fr < fs {
                xw = xr.clone();
                fw = fr;

                // Expansion
                let fb = pairs.first().unwrap().0;
                if fr < fb {
                    let xe = (1.0 - self.c) * &centroid + self.c * xr;
                    let fe = func(xe.to_vec());

                    // Expansion accepted
                    if fe < fb {
                        xw = xe;
                        fw = fe;
                    }
                }
            }
            else {
                // Contraction
                let xc = if fr < fw {
                    // Outside contraction
                    (1.0 + self.b) * &centroid - self.b * &xw
                }
                else {
                    // Inside contraction
                    (1.0 - self.b) * &centroid + self.b * &xw
                };
                let fc = func(xc.to_vec());

                // Contraction accepted
                let min = if fr < fw { fr } else { fw };
                if fc < min {
                    xw = xc;
                    fw = fc;
                }
                else {
                    // Shrinkage
                    let xb = pairs.first().unwrap().1.clone();
                    for &mut (_, ref mut x) in pairs.iter_mut().skip(1) {
                        *x *= self.d;
                        x.scaled_add(1.0 - self.d, &xb);
                    }
                }
            }

            // Pull update
            pairs.pop();
            pairs.push((fw, xw));

            // Sort
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal));


            // Termination tests
            let &(fb, ref xb) = pairs.first().unwrap();
            let &(fw, ref xw) = pairs.last().unwrap();

            // Domain convergence test
            let mut sorted = (xw - xb)
                .into_iter()
                .map(|xi| xi.abs())
                .collect::<Vec<f64>>();
            sorted.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));
            let test_x = sorted.last().unwrap().clone();

            // Function value convergence test
            let test_f = (fw - fb).abs();


            // Termination test
            if test_f <= self.tol_f && test_x <= self.tol_x {
                return Ok(Output { f_min: fb, x_min: xb.to_vec(), iter: iter });
            }
        }

        Err(())
    }
}
