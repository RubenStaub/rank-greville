# Rank-Greville <img src="logo.svg" alt="rank-greville logo" height="32">
Recursive least squares solver in Python3, based on rank-decomposition and inspired by Greville's algorithm

## Context
#### Minimum-norm least-squares solution
Let us consider the **minimum-norm least-squares solution X** to a system of linear equations **AX = Y**. This solution is unique and defined whether the system of linear equation is overdetermined, underdetermined or both. In other word, A can have any shape, and even be rank-deficient: **A is a n x m matrix of rank r**. Still, we'll assume here that the entries of A are **real numbers**.

#### Moore-Penrose pseudoinverse
Penrose showed that this solution X can be computed using the **pseudoinverse A<sup>+</sup>** (also called generalized inverse) of A: X = A<sup>+</sup>Y. Therefore, computing the pseudoinverse A<sup>+</sup> is particularly relevant for recomputing the least-squares solution X' with a different regressand Y'.

#### Recursive least-squares
What if a new equation aX = y is added? One could recompute from scratch the least-squares solution X' and/or the pseudoinverse A'<sup>+</sup> of the row-augmented matrix A'. However, this would be particularly time-consuming.

Instead, a **recursive least-squares solver** is designed to update the least-squares solution X' from the previous solution X, saving in practice computational time by re-using previous work. Such solver is particularly relevant when a constantly up-to-date solution is required, allowing near real-time applications to process new information on-the-fly.

#### Rank-Greville algorithm
The **rank-Greville algorithm** implemented here is a simple recursive least-squares solver based on a rank-decomposition update inspired by the Greville algorithm. More details about this algorithm (and linear algebra derivations associated) can be found in the related paper: _Add article link_

Please cite this paper if this tool is relevant to your work.

## A first example
```python
from rank_greville import RecursiveModel

# Initialize model
model = RecursiveModel()

# Add observations
model.add_observation([1, 1, 1], 1)
model.add_observation([2, 0, 0], 2)

# Print model parameters X (minimum-norm least-squares solution)
print('solution:', model.parameters)

# Add other observations
update = model.add_observation([0, 3, 3], 3)

# Print parameters update and new solution
print('solution update:', update)
print('new solution:', model.parameters)

# model.parameters is now the (best) solution X to equation AX = Y:
# [[1, 1, 1],       [[1],
#  [2, 0, 0], @ X =  [2],
#  [0, 3, 3]]        [3]]
```

## Versatile solver
The rank-Greville algorithm is implemented through the `RecursiveModel` class. This interface can be instantiated with optional arguments modifying its behavior:

### Least squares update
By default, a `RecursiveModel` instance `model = RecursiveModel()` is configured to maintain and update the minimum-norm least-squares solution only.

The always up-to-date least-squares solution is available via the data attribute `model.parameters`. 

Updating the model by adding an equation (i.e. calling `model.add_observation`) requires **O(mr) operations** if only the least-squares solution is requested to be updated.

### Pseudoinverse update
Along with the minimum-norm least-squares solution X, the rank-Greville algorithm can also maintain the Moore-Penrose pseudoinverse A<sup>+</sup> by supplying the additional argument `pseudo_inverse_update=True` during the instantiation of `RecursiveModel`.

The always up-to-date pseudoinverse is available via the data attribute `model.pseudo_inverse`.

Updating the model by adding an equation (i.e. calling `model.add_observation`) requires **O(mn) operations** if the pseudoinverse update is requested.

### Covariance matrix update
Finally, in addition to the least-squares solution X, this rank-Greville implementation can also maintain the covariance matrix Var(X):

> Var(X) = Var(A<sup>+</sup>ε) = A<sup>+</sup><sup>T</sup>Var(ε)A<sup>+</sup>

by supplying the additional argument `covariance_update=True` during the instantiation of `RecursiveModel`.

The always up-to-date covariance matrix for the model parameters, Var(X), is available via the data attribute `model.variance_parameters`.

Updating the model by adding an equation (i.e. calling `model.add_observation`) requires **O(m²) operations** if the covariance matrix update is requested.

### Storage only update
Alternatively to expansive updates of the pseudoinverse or covariance matrix, this rank-Greville implementation allows to maintain only the internal storage required for computing these items afterward at a lower cost (i.e. O(nmr) operations).

The pseudoinverse internal storage update can be switched on with the flag `pseudo_inverse_support=True` at an **additional O(nr) operations per update**. The covariance matrix internal storage update can be switched on with the flag `covariance_support=True` at an **additional O(nm) operations per update (see Notes below)**. Finally, all internal storage can be updated with the flag `full_storage=True`.

### Synoptic view
A summary of the flags/methods and complexities described above can be found in the following tables.

Flags/methods:
Operation | Least-squares solution | Pseudoinverse | Covariance matrix
------------ | ------------- | ------------- | -------------
**Attribute name** | `model.parameters` | `model.pseudo_inverse` | `model.variance_parameters`
**Attribute update** | Default behavior | `pseudo_inverse_update=True` flag | `covariance_update=True` flag
**Storage update** | N.A. | `pseudo_inverse_support=True` or `full_storage=True` flag | `covariance_support=True` or `full_storage=True` flag
**Computation from storage** | N.A. | `model.scratch_pseudo_inverse_computation` method | `model.scratch_covariance_computation` method

Time complexities:
Operation | Least-squares solution | Pseudoinverse | Covariance matrix
------------ | ------------- | ------------- | -------------
**Attribute update** | O(mr) | O(mn) | O(m²)
**Storage update** | N.A. | O((m+n)r) | O(mn) (see Notes below)
**Computation from storage** | N.A. | O(mnr) | O(mnr)

**Note:** This O(mn) complexity comes from simply maintaining a copy of A, and could straightforwardly be reduced to O(nr). If you are interested by such feature, feel free to request it by opening an issue, and @RubenStaub will focus on implementing it.

## Three variants
This Python3 module implements all three variants of the rank-Greville algorithm discussed in the related paper. These variants differ mainly in the internal observations basis used (i.e. the type of rank decomposition used).

### Original rank-Greville
In this version, the observations basis is an arbitrary maximal set of linearly independent observations among the rows of A.

The original rank-Greville algorithm is implemented via the `RecursiveModel` class.

### Orthogonal rank-Greville
In this version, the observations basis is an orthogonal basis composed of the non-null rejection vectors observed (i.e. component of each observation that is linearly indepedent from previously seen observations).

The orthogonal rank-Greville algorithm is implemented via the `RecursiveModelOrthogonal` class.

### Orthonormal rank-Greville
In this version, the observations basis is an orthonormal basis composed of the non-null rejection vectors observed rescaled to unit-length.

The orthonormal rank-Greville algorithm is implemented via the `RecursiveModelOrthonormal` class.

Note that this variant requires algebraic operations not supported on rationals, and is therefore not compatible with Python3's `Fractions` module (see Fractions support below).

## How to install

## Covariance support
This implementation includes support for computing the covariance matrix Var(X) of the minimum-norm least-squares solution X (i.e. the linear model parameters). The covariance matrix Var(X) is estimated by considering the hypotheses of an ordinary least squares: in particular the homoscedasticity and non-autocorrelation of the errors ε along with the exogeneity of the regressors. Therefore:

> Var(X) = Var(A<sup>+</sup>ε) = A<sup>+</sup><sup>T</sup>A<sup>+</sup>\*s<sup>2</sup>

> where s<sup>2</sup> is the unbiased estimator of the variance of the noise: s<sup>2</sup> = ε<sup>T</sup>ε/(n-r)

This is only well defined if A is a thin full rank matrix (i.e. overdetermined system), since, in this case, Im(A<sup>+</sup>) = ℝ<sup>m</sup>, meaning that A<sup>+</sup>ε can span over the whole ambient space of the model parameters.

However, if A is rank-deficient, then the kernel of A<sup>+</sup><sup>T</sup> is non-null and no empirical covariance information can be found for this subspace (since this subspace is simply unobserved). Nonetheless, it is still possible to consider a guess g for the least-squares solution X (e.g. g = 0 leads to the minimum-norm least-squares solution), and its associated covariance matrix Var(g).

But first, let us play with this concept by writing the system of linear equations A(X-g) = Y-Ag ⇔ AX' = Y'. Solving this system for X' leads to X-g being the minimum-norm solution such that the error A(X-g) - (Y-Ag) = AX-Y has minimum norm (i.e. X is a least-squares solution of AX = Y). As a consequence, by using a guess g, one can have find the least-squares solution X closest (with respect to the euclidean norm) to g. Using the pseudoinverse A<sup>+</sup>, we find:

> X-g = A<sup>+</sup>Y - A<sup>+</sup>Ag ⇔ X = A<sup>+</sup>Y + (1-A<sup>+</sup>A)g = A<sup>+</sup>Y + P<sub>Ker</sub> g

> where P<sub>Ker</sub> is the projector onto Ker(A), the kernel of A (i.e. the unobserved regressors subspace).

Note that considering a guess g provides with a generalization of the minimum-norm least-square method. In particular, taking g = 0 leads to the usual minimum-norm least-squares solution. This generalization also applies to the covariance matrix, assuming (quite reasonably) that g is uncorrelated with the error ε:

> Var(X) = Var(A<sup>+</sup>ε + P<sub>Ker</sub> g) = Var(A<sup>+</sup>ε) + Var(P<sub>Ker</sub> g)

This formula is used in this implementation for computing/updating the covariance matrix Var(X), and it requires a guess g and its associated covariance matrix Var(g) that can be provided by the `model.add_new_regressors` method.

By default, when new regressors are encountered (for the first observation, or if m increases) their associated guess is set to zero, and assumed non-autocorrelated with undefined variance (i.e. 𝔼(g) = 0, Var(g) = diag(NaN)). If covariance matrix computations are requested, a different guess should be provided by calling `model.add_new_regressors` beforehand with keyword argument `new_variances_guess=Var(g)`. Note that this method can also be used to provide with a non-null guess `new_param_guess=g` (𝔼(g) technically). More information can be found in the documentation of this function: `help(RecursiveModel.add_new_regressors)`.

## Fractions support and floating point precision

## Some examples
### Fractions support

### Covariance matrix and confidence intervals

## Designed with Numpy
This module heavily relies on the Numpy library for a fast and multi-threading ready implementation.

## Dependencies
Numpy
