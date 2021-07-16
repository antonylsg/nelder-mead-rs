use num_traits::Float;
use num_traits::Zero;

use std::convert::AsMut;
use std::convert::AsRef;
use std::iter;
use std::iter::FromIterator;
use std::mem;
use std::ops::Add;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;

#[derive(Debug, Clone, Copy)]
pub struct NotEnoughItems;

pub trait TryFromIterator<A> {
    type Error: std::fmt::Debug;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = A>,
        Self: Sized;
}

fn try_from_iter<I, T, const N: usize>(iter: I) -> Result<[T; N], NotEnoughItems>
where
    I: IntoIterator<Item = T>,
{
    let mut iter = iter.into_iter();
    let mut buffer = mem::MaybeUninit::<[T; N]>::uninit();
    let ptr: *mut T = unsafe { mem::transmute(&mut buffer) };

    for i in 0..N {
        if let Some(next) = iter.next() {
            unsafe { ptr.add(i).write(next) };
        } else {
            return Err(NotEnoughItems);
        }
    }

    Ok(unsafe { buffer.assume_init() })
}

impl<T, const N: usize> TryFromIterator<T> for [T; N] {
    type Error = NotEnoughItems;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>,
    {
        try_from_iter(iter)
    }
}

impl<T> TryFromIterator<T> for Vec<T> {
    type Error = std::convert::Infallible;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>,
    {
        Ok(iter.into_iter().collect())
    }
}

pub trait Array:
    TryFromIterator<Self::Item>
    + IntoIterator
    + AsRef<[Self::Item]>
    + AsMut<[Self::Item]>
    + Index<usize, Output = Self::Item>
    + IndexMut<usize>
    + Clone
{
}

impl<T, U> Array for T where
    T: TryFromIterator<U>
        + IntoIterator<Item = U>
        + AsRef<[Self::Item]>
        + AsMut<[Self::Item]>
        + Index<usize, Output = U>
        + IndexMut<usize>
        + Clone
{
}

#[derive(Debug, Clone)]
pub(crate) struct Vector<A: Array>(pub(crate) A);

impl<A: Array> Vector<A> {
    pub(crate) fn from_slice(slice: &[A::Item]) -> Vector<A>
    where
        A::Item: Copy,
    {
        Vector(A::try_from_iter(slice.iter().copied()).unwrap())
    }

    pub(crate) fn scaled_add(&mut self, a: A::Item, rhs: &A)
    where
        A::Item: Float,
    {
        self.iter_mut()
            .zip(rhs.as_ref().iter())
            .for_each(|(x, y)| *x = y.mul_add(a, *x))
    }

    pub(crate) fn zeros(dim: usize) -> Vector<A>
    where
        A::Item: Zero,
    {
        iter::repeat_with(A::Item::zero).take(dim).collect()
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, A::Item> {
        self.0.as_ref().iter()
    }

    pub(crate) fn iter_mut(&mut self) -> std::slice::IterMut<'_, A::Item> {
        self.0.as_mut().iter_mut()
    }
}

impl<A: Array> Index<usize> for Vector<A> {
    type Output = A::Item;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<A: Array> IndexMut<usize> for Vector<A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<A: Array> FromIterator<A::Item> for Vector<A> {
    fn from_iter<I: IntoIterator<Item = A::Item>>(iter: I) -> Self {
        Vector(A::try_from_iter(iter).unwrap())
    }
}

impl<A: Array> Deref for Vector<A> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A: Array> DerefMut for Vector<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<A: Array> Add for Vector<A>
where
    A::Item: Copy + Add<Output = A::Item>,
{
    type Output = Vector<A>;

    fn add(self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x + y).collect()
    }
}

impl<A: Array> Add for &Vector<A>
where
    A::Item: Copy + Add<Output = A::Item>,
{
    type Output = Vector<A>;

    fn add(self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x + y).collect()
    }
}

impl<A: Array> Add<&Vector<A>> for Vector<A>
where
    A::Item: Copy + Add<Output = A::Item>,
{
    type Output = Vector<A>;

    fn add(self, rhs: &Vector<A>) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x + y).collect()
    }
}

impl<A: Array> Add<Vector<A>> for &Vector<A>
where
    A::Item: Copy + Add<Output = A::Item>,
{
    type Output = Vector<A>;

    fn add(self, rhs: Vector<A>) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x + y).collect()
    }
}

impl<A: Array> Mul<A::Item> for Vector<A>
where
    A::Item: Copy + Mul<Output = A::Item>,
{
    type Output = Vector<A>;

    fn mul(self, rhs: A::Item) -> Self::Output {
        self.iter().map(|&x| x * rhs).collect()
    }
}

impl<A: Array> Mul<A::Item> for &Vector<A>
where
    A::Item: Copy + Mul<Output = A::Item>,
{
    type Output = Vector<A>;

    fn mul(self, rhs: A::Item) -> Self::Output {
        self.iter().map(|&x| x * rhs).collect()
    }
}

impl<A: Array> MulAssign<A::Item> for Vector<A>
where
    A::Item: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: A::Item) {
        self.iter_mut().for_each(|x| *x *= rhs)
    }
}

impl<A: Array> Sub for Vector<A>
where
    A::Item: Copy + Sub<Output = A::Item>,
{
    type Output = Vector<A>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x - y).collect()
    }
}

impl<A: Array> Sub for &Vector<A>
where
    A::Item: Copy + Sub<Output = A::Item>,
{
    type Output = Vector<A>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x - y).collect()
    }
}

impl<A: Array> Sub<&Vector<A>> for Vector<A>
where
    A::Item: Copy + Sub<Output = A::Item>,
{
    type Output = Vector<A>;

    fn sub(self, rhs: &Vector<A>) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x - y).collect()
    }
}

impl<A: Array> Sub<Vector<A>> for &Vector<A>
where
    A::Item: Copy + Sub<Output = A::Item>,
{
    type Output = Vector<A>;

    fn sub(self, rhs: Vector<A>) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&x, &y)| x - y).collect()
    }
}
