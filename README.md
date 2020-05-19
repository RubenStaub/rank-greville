# rank-greville
Recursive least squares solver in Python3, based on rank-decomposition and inspired by Greville's algorithm

## Context
#### Minimum-norm least-squares solution
Let us consider the **minimum-norm least-squares solution X** to a system of linear equations **AX = Y**. This solution is unique and defined whether the system of linear equation is overdetermined, underdetermined or both (i.e. for any matrix A). Still, we'll assume here that the entries of A are **real numbers**.

#### Moore-Penrose pseudoinverse
Penrose showed that this solution X can be computed using the **pseudoinverse A<sup>+</sup>** (also called generalized inverse) of A: X = A<sup>+</sup>Y. Therefore, computing the pseudoinverse A<sup>+</sup> is particularly relevant for updating the least-squares solution X' on altered regressand Y'.

#### Recursive least-squares
What if a new equation aX = y is added? One could recompute from scratch the least-squares solution X' and/or the pseudoinverse A'<sup>+</sup> of the row-augmented matrix A'. However, this would be particularly time-consuming.

Instead, a **recursive least-squares solver** is designed to update the least-squares solution X' from the previous solution X, saving in practice computational time by re-using previous work. Such solver is particularly relevant when a constantly up-to-date solution is required, allowing near real-time applications to process new information on-the-fly.

#### Rank-Greville algorithm
The **rank-Greville algorithm** implemented here is a simple recursive least-squares solver based on a rank-decomposition update inspired by the Greville algorithm. More details about this algorithm (and linear algebra derivations associated) can be found in the related paper: _Add article link_

Please cite this paper if this tool is relevant to your work.

## Versatile solver
### Least squares update

### Pseudoinverse update

### Covariance matrix update

## Three variants
### Original rank-Greville

### Orthogonal rank-Greville

### Orthonormal rank-Greville

## How to install

## Some examples

## Fractions support

## Designed with Numpy
This module heavily relies on the Numpy library for a fast and multi-threading ready implementation.

## Dependencies
Numpy
