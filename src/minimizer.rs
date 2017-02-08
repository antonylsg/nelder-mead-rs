#![allow(dead_code)]
extern crate ndarray;

use self::ndarray::Array1;


#[derive(Debug)]
pub struct Output {
    pub f_min: f64,
    pub x_min: Vec<f64>,
    pub iter: usize,
}


#[derive(Debug)]
pub struct Minimizer {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    h: f64,
    tol: f64,
    max: usize,
}

impl Default for Minimizer {
    fn default() -> Minimizer {
        Minimizer {
            // Reflection parameter
            a: 1.0,

            // Contraction parameter
            b: 0.5,

            // Expansion parameter
            c: 2.0,

            // Shrinkage parameter
            d: 0.5,

            // Initialization parameter
            h: 0.05,

            // Tolerance parameter
            tol: 1e-4,

            // Iterations parameter
            max: 200,
        }
    }
}

impl Minimizer {
    pub fn tol(&self) -> f64 {
        self.tol
    }

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
                // *x.uget_mut(idx) += self.h;
                let xi = x.uget_mut(idx);
                *xi = if *xi == 0.0 {
                    self.h
                }
                else {
                    *xi * self.h
                };
            }

            let pair = (func(x.to_vec()), x.clone());
            pairs.push(pair);
        }


        // Sort
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal));


        // Centroid
        let mut centroid = pairs
            .iter()
            .rev()
            .skip(1)
            .map(|p| &p.1)
            .fold(Array1::zeros(x0.dim()), |acc, x| acc + x) * inv_dim;


        for iter in 0 .. self.max * x0.dim() {
            let (mut fh, mut xh) = pairs.last().cloned().unwrap();

            // Save old xh
            let old_xh = xh.clone();

            // Reflection
            let xr = (1.0 + self.a) * &centroid - self.a * xh.clone();
            let fr = func(xr.to_vec());
            let fs = pairs.iter().rev().skip(1).rev().last().unwrap().0;

            // Reflection accepted
            if fr < fs {
                xh = xr.clone();
                fh = fr;

                // Expansion
                let fl = pairs.first().unwrap().0;
                if fr < fl {
                    let xe = (1.0 - self.c) * &centroid + self.c * xr;
                    let fe = func(xe.to_vec());

                    // Expansion accepted
                    if fe < fl {
                        xh = xe;
                        fh = fe;
                    }
                }

                // Pull update
                pairs.pop();
                pairs.push((fh, xh));

                // Sort
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0)
                    .unwrap_or(Ordering::Equal));

                // Update centroid
                let new_xh = &pairs.last().unwrap().1;
                let dxh = old_xh - new_xh;
                centroid.scaled_add(inv_dim, &dxh);
            }
            else {
                // Contraction
                let xc = if fr < fh {
                    // Outside contraction
                    (1.0 - self.b) * &centroid + self.b * xr
                }
                else {
                    // Inside contraction
                    (1.0 - self.b) * &centroid + self.b * &xh
                };
                let fc = func(xc.to_vec());

                // Contraction accepted
                let min = if fr < fh { fr } else { fh };
                if fc < min {
                    xh = xc;
                    fh = fc;

                    // Pull update
                    pairs.pop();
                    pairs.push((fh, xh));

                    // Sort
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0)
                        .unwrap_or(Ordering::Equal));

                    // Update centroid
                    let new_xh = &pairs.last().unwrap().1;
                    let dxh = old_xh - new_xh;
                    centroid.scaled_add(inv_dim, &dxh);
                }
                else {
                    // Shrinkage
                    let xl = pairs.first().unwrap().1.clone();
                    for &mut (_, ref mut x) in pairs.iter_mut().skip(1) {
                        // *x = (1.0 - self.d) * &xl + self.d * x.clone();
                        *x *= self.d;
                        x.scaled_add(1.0 - self.d, &xl);
                    }

                    // Pull update
                    pairs.pop();
                    pairs.push((fh, xh));

                    // Sort
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0)
                        .unwrap_or(Ordering::Equal));
                    
                    // Update centroid
                    let new_xh = &pairs.last().unwrap().1;
                    let dxh = old_xh - new_xh;
                    centroid = (1.0 - self.d) * xl + self.d * centroid;
                    centroid.scaled_add(inv_dim, &dxh);
                }
            }

            let (fl, fh) = (pairs.first().unwrap().0, pairs.last().unwrap().0);
            // let end = 2.0 * (fh - fl).abs() / (fh.abs() + fl.abs()) <= self.tol;
            // let end = (fh - fl).abs() <= self.tol;

            if (fh - fl).abs() <= self.tol {
                let pair = pairs.first().unwrap();
                
                return Ok(Output {
                    f_min: pair.0,
                    x_min: pair.1.to_vec(),
                    iter: iter,
                });
            }
        }

        Err(())
    }
}
