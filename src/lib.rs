mod minimizer;
#[cfg(test)]
mod tests;

pub use minimizer::error::Error;
pub use minimizer::{Minimizer, Output, Result};
