from __future__ import annotations

from dataclasses import replace
from functools import reduce
from math import prod
from typing import cast, overload

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .dataarray import IndexType
from .layout import Layout, promote_layouts
from .materialized_qarray import MaterializedQArray
from .qarray import QArray, QArrayLike

__all__ = []


class CompositeTerm(eqx.Module):
    r"""One separable term $c \, A_0 \otimes \cdots \otimes A_{N-1}$ in a
    :class:`CompositeQArray`.  Holds the bulk of the lazy logic; most ``LAZY``
    methods on :class:`CompositeQArray` delegate to a corresponding method here.

    Attributes:
        operators: Per-subsystem local operators (one :class:`MaterializedQArray` per
            subsystem). Operators, kets and bras are all supported, so a term is not
            necessarily square.
        coeff: Scalar coefficient; may be a JAX array for batched use. Defaults to 1.
    """

    # TODO: maybe rename "operators" back to "factors"?
    operators: tuple[MaterializedQArray, ...]
    coeff: ArrayLike = 1.0

    # === Lifecycle ===

    def __check_init__(self):
        # check that there is at least one operator
        if not self.operators:
            raise ValueError('A `CompositeTerm` must have at least one operator.')

        # check that the batch shapes of the operators and of the coefficient are
        # broadcastable
        try:
            self._broadcast_batch_shape()
        except ValueError as e:
            raise ValueError(
                'All operators of a `CompositeTerm` must have broadcastable batch '
                'shapes, and the coefficient must be broadcastable against them.'
            ) from e

    def _broadcast_batch_shape(self) -> tuple[int, ...]:
        """Return the broadcast batch shape of the operators and of the coefficient."""
        return jnp.broadcast_shapes(
            *(operator.shape[:-2] for operator in self.operators), jnp.shape(self.coeff)
        )

    # === Materialization ===

    def _materialize(self) -> MaterializedQArray:
        """Coeff * (A_0 ⊗ … ⊗ A_{N-1}); reduce via op.__and__ then __mul__(coeff)."""
        operator = reduce(lambda x, y: x & y, self.operators)

        # a batched coefficient must be given two trailing dimensions to multiply the
        # matrix axes of the operator
        coeff = self.coeff
        if jnp.ndim(coeff) > 0:
            coeff = jnp.asarray(coeff)[..., None, None]

        return cast(MaterializedQArray, operator * coeff)

    # === Properties ===

    @property
    def dtype(self) -> jnp.dtype:
        # jnp.result_type over each op's .dtype + coeff.
        return jnp.result_type(
            *(operator.dtype for operator in self.operators), self.coeff
        )

    @property
    def shape(self) -> tuple[int, ...]:
        # (*batch, prod(n_k), prod(m_k)); batch axes broadcast across ops/coeff.
        # Taking the axes seperately allows for kets, bras, or square operators.
        n = prod(operator.shape[-2] for operator in self.operators)
        m = prod(operator.shape[-1] for operator in self.operators)
        return (*self._broadcast_batch_shape(), n, m)

    @property
    def layout(self) -> Layout:
        # aggregate over op's .layout (e.g. dense if any op is dense, else dia).
        return reduce(promote_layouts, (operator.layout for operator in self.operators))

    @property
    def mT(self) -> CompositeTerm:
        # (c·⊗A_k)^T = c·⊗A_k^T → each op's .mT.
        operators = tuple(operator.mT for operator in self.operators)
        return replace(self, operators=operators)

    # === Array methods ===

    def conj(self) -> CompositeTerm:
        # conj(c·⊗A_k) = conj(c)·⊗conj(A_k) → each op's .conj() + jnp.conj(coeff).
        operators = tuple(operator.conj() for operator in self.operators)
        return replace(self, operators=operators, coeff=jnp.conj(self.coeff))

    def broadcast_to(self, *shape: int) -> CompositeTerm:
        # batch axes only → each op's .broadcast_to() + jnp.broadcast_to(coeff, ...).
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because the last two dimensions do '
                f'not match current shape dimensions, {self.shape}.'
            )

        # broadcasting the coefficient alone is enough and the operators are untouched
        bshape = shape[:-2]
        try:
            broadcastable = (
                jnp.broadcast_shapes(self._broadcast_batch_shape(), bshape) == bshape
            )
        except ValueError:
            broadcastable = False
        if not broadcastable:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because it is incompatible with '
                f'the current shape {self.shape}.'
            )

        return replace(self, coeff=jnp.broadcast_to(self.coeff, bshape))

    def trace(self) -> Array:
        # tr(c·⊗A_k) = c·Π_k tr(A_k) → each op's .trace().
        raise NotImplementedError

    def sum(self, axis: int | tuple[int, ...] | None = None) -> CompositeTerm:
        # MATERIALIZE → _materialize().sum(axis).
        raise NotImplementedError

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> CompositeTerm:
        # batch axes only → each op's .squeeze(axis) + jnp.squeeze(coeff, axis).
        raise NotImplementedError

    def powm(self, n: int) -> CompositeTerm:
        # (c·⊗A_k)^n = c^n·⊗A_k^n → each op's .powm(n).
        raise NotImplementedError

    def expm(self, *, max_squarings: int = 16) -> MaterializedQArray:
        # exp(c·⊗A_k) = (⊗V_k)·diag(exp(c·∏λ_k))·(⊗V_k)^†; returns MaterializedQArray.
        # → each op's ._eigh().
        raise NotImplementedError

    def norm(self, *, psd: bool = False) -> Array:
        # LAZY if psd=False: ‖c·⊗A_k‖_F = |c|·Π_k‖A_k‖_F.
        # psd=True: trace shortcut only if known PSD; otherwise materialize.
        raise NotImplementedError

    def _eig(self) -> tuple[Array, MaterializedQArray]:
        # eigenvalues = c·Cartesian(λ_k), eigenvectors = ⊗V_k (materialized)
        # → each op's ._eig().
        raise NotImplementedError

    def _eigh(self) -> tuple[Array, Array]:
        # Hermitian variant; returns raw JAX arrays → each op's ._eigh().
        raise NotImplementedError

    def _eigvals(self) -> Array:
        # c · Cartesian product of per-op eigenvalues → each op's ._eigvals().
        raise NotImplementedError

    def _eigvalsh(self) -> Array:
        # Hermitian variant → each op's ._eigvalsh().
        raise NotImplementedError

    def devices(self) -> set[Device]:
        # must all be the same by convention ?
        raise NotImplementedError

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        # Sufficient (not necessary): coeff real AND all ops .isherm().
        # False here is not conclusive for multi-term CompositeQArray.
        raise NotImplementedError

    def block_until_ready(self) -> CompositeTerm:
        # → each op's .block_until_ready().
        raise NotImplementedError

    # === Quantum methods ===

    def ptrace(self, keep: tuple[int, ...]) -> CompositeTerm:
        # ptrace_{∉keep}(c·⊗A_j) = c·(Π_{j∉keep} tr(A_j))·⊗_{k∈keep} A_k
        # → .trace() on each traced-out op.
        raise NotImplementedError

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> CompositeTerm:
        # batch axes only → each op's __getitem__.
        # Matrix-axis keys: caller materializes.
        raise NotImplementedError

    # === Arithmetic ===

    def __mul__(self, y: QArrayLike) -> CompositeTerm:
        # y·(c·⊗A_k) = (y·c)·⊗A_k; only touches coeff.
        raise NotImplementedError

    def __matmul__(self, other: CompositeTerm) -> CompositeTerm:
        # is the main mpoint of the feature
        raise NotImplementedError

    def __and__(self, other: CompositeTerm) -> CompositeTerm:
        # (c·⊗A_k)⊗(d·⊗B_l) = (c·d)·(A_*,B_*); tuple concat + coeff multiply.
        raise NotImplementedError


class CompositeQArray(QArray):
    r"""Lazy sum of separable tensor-product operators.

    $H = \sum_j c_j A_{j,0} \otimes \cdots \otimes A_{j,N-1}$, stored in factored form
    to avoid the exponential cost of the full $n \times n$ matrix.

    ``dims`` is inherited from :class:`QArray`.

    Strategy tags used in method comments:

    - ``LAZY``: implemented term-wise; no full matrix built.
    - ``MATERIALIZE``: falls back to ``_materialize().<method>(...)``.
    - ``MIXED``: LAZY for batch axes, MATERIALIZE for matrix axes.
    - ``1-term``: single-term fast path that skips full materialization.
    - ``★``: big-win lazy methods (core motivation for this class).

    Attributes:
        terms: Tuple of :class:`CompositeTerm` objects that sum to the operator.
    """

    terms: tuple[CompositeTerm, ...]

    # === Lifecycle ===

    def __check_init__(self):
        # Check types in sum are the same and check init of terms is ok
        pass

    # === Materialization ===

    def _materialize(self) -> MaterializedQArray:
        """Sum of term._materialize() over all terms.

        Fallback for MATERIALIZE methods.
        """
        raise NotImplementedError

    # === Properties ===

    @property
    def dtype(self) -> jnp.dtype:
        # LAZY → term.dtype; must all match
        raise NotImplementedError

    @property
    def layout(self) -> Layout:
        # CONVENTION → term.layout; aggregate (e.g. dense if any is dense).
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        # LAZY → term.shape; broadcast batch axes across terms.
        raise NotImplementedError

    @property
    def mT(self) -> QArray:
        # LAZY (A⊗B)^T=A^T⊗B^T → term.mT.
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        # LAZY → term.ndim; must all match.
        raise NotImplementedError

    # === Array methods ===

    def conj(self) -> QArray:
        # LAZY → term.conj().
        raise NotImplementedError

    def reshape(self, *shape: int) -> QArray:
        # MATERIALIZE → _materialize().reshape(*shape).
        raise NotImplementedError

    def _reshape_unchecked(self, *shape: int) -> QArray:
        # MATERIALIZE → _materialize()._reshape_unchecked(*shape).
        raise NotImplementedError

    def broadcast_to(self, *shape: int) -> QArray:
        # LAZY batch axes only → term.broadcast_to(...).
        raise NotImplementedError

    def powm(self, n: int) -> QArray:
        # MATERIALIZE | 1-term (c·⊗A_k)^n=c^n·⊗A_k^n → term.powm(n).
        raise NotImplementedError

    def expm(self, *, max_squarings: int = 16) -> QArray:
        # MATERIALIZE | 1-term per-factor spectral path → term.expm(...).
        raise NotImplementedError

    def norm(self, *, psd: bool = False) -> Array:
        # LAZY if psd=False: Gram sum over term pairs using local traces.
        # psd=True: trace shortcut only if known PSD; otherwise materialize.
        # can be unstable
        raise NotImplementedError

    def trace(self) -> Array:
        # LAZY tr(c·⊗A_k)=c·Π tr(A_k) → sum(term.trace()).
        raise NotImplementedError

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # MATERIALIZE → _materialize().sum(axis).
        raise NotImplementedError

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # LAZY → term.squeeze(axis).
        raise NotImplementedError

    def _eig(self) -> tuple[Array, QArray]:
        # MATERIALIZE | 1-term eigenvalues=c·Cartesian(λ_k), eigenvecs=⊗V_k
        # → term._eig().
        raise NotImplementedError

    def _eigh(self) -> tuple[Array, Array]:
        # MATERIALIZE | 1-term Hermitian variant → term._eigh().
        raise NotImplementedError

    def _eigvals(self) -> Array:
        # MATERIALIZE | 1-term → term._eigvals().
        raise NotImplementedError

    def _eigvalsh(self) -> Array:
        # MATERIALIZE | 1-term → term._eigvalsh().
        raise NotImplementedError

    def devices(self) -> set[Device]:
        # LAZY → all must be on same device ? .
        raise NotImplementedError

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        # MATERIALIZE | 1-term sufficient check → term.isherm(rtol, atol).
        raise NotImplementedError

    def block_until_ready(self) -> QArray:
        # LAZY → term.block_until_ready().
        raise NotImplementedError

    # === Quantum methods ===

    def ptrace(self, *keep: int) -> QArray:
        # LAZY → term.ptrace(keep).
        raise NotImplementedError

    # === Conversion ===

    def to_qutip(self) -> Qobj | list[Qobj]:
        # MATERIALIZE → _materialize().to_qutip().
        raise NotImplementedError

    def to_jax(self) -> Array:
        # MATERIALIZE → _materialize().to_jax().
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        # MATERIALIZE → _materialize().to_numpy().
        raise NotImplementedError

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        # MATERIALIZE → _materialize().__array__(dtype, copy).
        raise NotImplementedError

    def asdense(self) -> QArray:
        # MATERIALIZE → _materialize().asdense().
        raise NotImplementedError

    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> QArray:
        # MATERIALIZE → _materialize().assparsedia(offsets).
        raise NotImplementedError

    # === Repr ===

    def __repr__(self) -> str:
        # LAZY; print dims, n_terms, shape, dtype, layout.
        raise NotImplementedError

    # === Arithmetic ===

    def __mul__(self, y: QArrayLike) -> QArray:
        # LAZY y·Σc_j⊗A_{jk}=Σ(y·c_j)⊗A_{jk} → term.__mul__(y).
        raise NotImplementedError

    def __add__(self, y: QArrayLike) -> QArray:
        # LAZY ★ two composites: self.terms + other.terms.
        # Non-composite y: wrap as single-operator CompositeTerm first.
        raise NotImplementedError

    @overload
    def __matmul__(self, y: QArray) -> QArray: ...

    @overload
    def __matmul__(self, y: ArrayLike) -> Array: ...

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        # LAZY ★ (Σc_j⊗A_jk)·(Σd_l⊗B_lk)=Σ_{j,l}(c_j·d_l)⊗(A_jk·B_lk) → term_j @ term_l.
        raise NotImplementedError

    @overload
    def __rmatmul__(self, y: QArray) -> QArray: ...

    @overload
    def __rmatmul__(self, y: ArrayLike) -> Array: ...

    def __rmatmul__(self, y: QArrayLike) -> QArray | Array:
        # LAZY symmetric to __matmul__ → term_other @ term_self.
        raise NotImplementedError

    def __and__(self, y: QArray) -> QArray:
        # LAZY ★ (Σc_j⊗A_jk)⊗(Σd_l⊗B_lk)=Σ_{j,l}(c_j·d_l)⊗(A_j*,B_l*) → term_j & term_l.
        raise NotImplementedError

    # === Element-wise ===

    def addscalar(self, y: ArrayLike) -> QArray:
        # MATERIALIZE → _materialize().addscalar(y).
        raise NotImplementedError

    def elmul(self, y: QArrayLike) -> QArray:
        # MATERIALIZE → _materialize().elmul(y).
        raise NotImplementedError

    def elpow(self, power: int) -> QArray:
        # MATERIALIZE → _materialize().elpow(power).
        raise NotImplementedError

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> QArray:
        # MIXED batch: term[key] | matrix: _materialize()[key].
        raise NotImplementedError
