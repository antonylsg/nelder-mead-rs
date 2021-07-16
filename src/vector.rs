use array::Array;
use num_traits::Float;
use num_traits::Zero;

use std::iter;
use std::iter::FromIterator;
use std::ops::Add;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;

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
