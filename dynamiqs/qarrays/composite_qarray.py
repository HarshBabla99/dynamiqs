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

from .._utils import is_batched_scalar
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
        # Taking the axes separately allows for kets, bras, or square operators.
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
        return jnp.asarray(
            self.coeff * prod(operator.trace() for operator in self.operators)
        )

    def sum(self, axis: int | tuple[int, ...] | None = None) -> CompositeTerm:
        # MATERIALIZE → _materialize().sum(axis).
        raise NotImplementedError

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> CompositeTerm:
        # batch axes only → each op's .squeeze(axis) + jnp.squeeze(coeff, axis).
        raise NotImplementedError

    def powm(self, n: int) -> CompositeTerm:
        # (c·⊗A_k)^n = c^n·⊗A_k^n → each op's .powm(n).
        operators = tuple(operator.powm(n) for operator in self.operators)
        return replace(self, operators=operators, coeff=self.coeff**n)

    def expm(self) -> MaterializedQArray:
        # exp(c·⊗A_k) = (⊗V_k)·diag(exp(c·∏λ_k))·(⊗V_k)^-1; returns a
        # MaterializedQArray. Assumes that each operator is diagonalizable.
        from .dense_dataarray import _bkron  # noqa: PLC0415
        from .utils import asqarray  # noqa: PLC0415

        # NOTE: Diagonalizes each subsystem using _eig()
        evals, evecs = [], []
        for operator in self.operators:
            operator_evals, operator_evecs = operator._eig()
            evals.append(operator_evals)
            evecs.append(operator_evecs.to_jax())

        # (⊗V_k)^-1 = ⊗(V_k^-1), so invert for each subsystem independently
        V = reduce(_bkron, evecs)
        Vinv = reduce(_bkron, [jnp.linalg.inv(evec) for evec in evecs])

        # exp(V·Λ·V^-1) = V·exp(Λ)·V^-1, where multiplying by the diagonal exp(Λ) is
        # just a scaling of the columns of V
        data = (V * jnp.exp(self._combine_evals(evals))[..., None, :]) @ Vinv

        dims = tuple(d for operator in self.operators for d in operator.dims)
        return cast('MaterializedQArray', asqarray(data, dims=dims))

    def norm(self, *, psd: bool = False) -> Array:
        if psd:
            return self.trace().real

        # LAZY if psd=False: ‖c·⊗A_k‖ = |c|·Π_k‖A_k‖
        # trace norm if matrix, else L2 norm of bra/ket
        # both are multiplicative under the tensor product
        return jnp.asarray(
            jnp.abs(self.coeff) * prod(operator.norm() for operator in self.operators)
        )

    def _combine_evals(self, evals: list[Array]) -> Array:
        """Eigenvalues of the CompositeTerm are simply the permutations of the
        per-operator eigenvalues. These can simply be combined by an outer product.

        Args:
            evals: List of eigenvalues, `evals[i]` being an Array of eigenvalues for
                operator `i`.
        """
        # get all permutations of evals, across the subsystems
        # similar to .dense_dataarray._bkron, but 1D
        _flattened_outer_prod = jnp.vectorize(jnp.kron, signature='(a),(b)->(ab)')
        composite_term_evals = reduce(_flattened_outer_prod, evals)

        # for batched coeffs, add a trailing dim to broadcast against eval axis
        coeff = self.coeff
        if jnp.ndim(coeff) > 0:
            coeff = jnp.asarray(coeff)[..., None]
        return jnp.asarray(coeff * composite_term_evals)

    def _eig(self) -> tuple[Array, MaterializedQArray]:
        # eigenvalues = c·Cartesian(λ_k), eigenvectors = ⊗V_k (materialized)
        # → each op's ._eig().
        evals, evecs = [], []
        for operator in self.operators:
            operator_evals, operator_evecs = operator._eig()
            evals.append(operator_evals)
            evecs.append(operator_evecs)

        # `jnp.linalg.eig` does not guarantee an ordering, so we don't need to either
        # TODO: evecs can be a list of CompositeQArrays
        evals = self._combine_evals(evals)
        evecs = reduce(lambda x, y: x & y, evecs)

        # for batched coeffs, _combine_evals introduces batch dims for evals
        # do the same for evecs.
        bshape = self._broadcast_batch_shape()
        evecs = evecs.broadcast_to(*bshape, *evecs.shape[-2:])

        return evals, cast('MaterializedQArray', evecs)

    def _eigh(self) -> tuple[Array, Array]:
        # Hermitian variant; returns raw JAX arrays → each op's ._eigh().
        from .dense_dataarray import _bkron  # noqa: PLC0415

        # NOTE: assumes each operator is Hermitian
        # as explained in isherm, a Hermitian CompositeTerm could be made up of
        # an even number of anti-Hermitian ops.
        evals, evecs = [], []
        for operator in self.operators:
            operator_evals, operator_evecs = operator._eigh()
            evals.append(operator_evals)
            evecs.append(operator_evecs)

        # NOTE: DenseDataArray._eigh() returns evecs as a JAX Array, not QArray
        # Unlike _eig above, we can't use __and__ directly.
        # TODO: evecs can be a list of CompositeQArrays.
        # NOTE: the imag values of complex coeffs are silently dropped
        evals = self._combine_evals(evals).real
        evecs = reduce(_bkron, evecs)

        # for batched coeffs, _combine_evals introduces batch dims for evals
        # do the same for evecs.
        bshape = self._broadcast_batch_shape()
        evecs = jnp.broadcast_to(evecs, (*bshape, *evecs.shape[-2:]))

        # sort in ascending order
        order = jnp.argsort(evals, axis=-1)
        evals = jnp.take_along_axis(evals, order, axis=-1)
        evecs = jnp.take_along_axis(evecs, order[..., None, :], axis=-1)

        # NOTE: returns JAX Array, following DenseDataArray._eigh()
        return evals, evecs

    def _eigvals(self) -> Array:
        # c · Cartesian product of per-op eigenvalues → each op's ._eigvals().
        return self._combine_evals([operator._eigvals() for operator in self.operators])

    def _eigvalsh(self) -> Array:
        # Hermitian variant → each op's ._eigvalsh().
        # NOTE: the imag values of complex coeffs are silently dropped
        evals = self._combine_evals(
            [operator._eigvalsh() for operator in self.operators]
        ).real
        return jnp.sort(evals, axis=-1)

    def devices(self) -> set[Device]:
        # the operators are not required to share a device, so all of them are reported
        return set().union(*(operator.devices() for operator in self.operators))

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        # Sufficient (not necessary): coeff real AND all ops .isherm().
        # False here is not conclusive for a CompositeTerm
        # e.g. the product of two anti-Hermitian operators is Hermitian.
        isherm = jnp.all(jnp.isreal(self.coeff))
        for operator in self.operators:
            isherm = jnp.logical_and(isherm, operator.isherm(rtol, atol))
        return cast('bool', isherm)

    def block_until_ready(self) -> CompositeTerm:
        # → each op's .block_until_ready().
        for operator in self.operators:
            operator.block_until_ready()
        return self

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
        if not is_batched_scalar(y):
            return NotImplemented

        # a batched scalar is shaped (1,) or (..., 1, 1) so that it broadcasts against
        # the matrix axes of a qarray. A coefficient has no matrix axes, so these
        # trailing dimensions are dropped before it is multiplied in.
        if jnp.ndim(y) > 0:
            y = jnp.asarray(y)
            y = y[0] if y.shape == (1,) else y[..., 0, 0]

        return replace(self, coeff=self.coeff * y)

    def __matmul__(self, other: CompositeTerm) -> CompositeTerm:
        # is the main mpoint of the feature
        # (c·⊗A_k)·(d·⊗B_k) = (c·d)·⊗(A_k·B_k), since the tensor product acts on
        # different subsystems: the two terms are multiplied subsystem by subsystem.
        if not isinstance(other, CompositeTerm):
            return NotImplemented

        if len(self.operators) != len(other.operators):
            raise ValueError(
                'Cannot matrix multiply two `CompositeTerm`s defined over a different '
                f'number of subsystems, but got {len(self.operators)} and '
                f'{len(other.operators)}.'
            )

        operators = cast(
            'tuple[MaterializedQArray, ...]',
            tuple(
                operator_a @ operator_b
                for operator_a, operator_b in zip(
                    self.operators, other.operators, strict=True
                )
            ),
        )
        return CompositeTerm(operators, self.coeff * other.coeff)

    def __and__(self, other: CompositeTerm) -> CompositeTerm:
        # (c·⊗A_k)⊗(d·⊗B_l) = (c·d)·(A_*,B_*); tuple concat + coeff multiply.
        if not isinstance(other, CompositeTerm):
            return NotImplemented

        operators = self.operators + other.operators
        return CompositeTerm(operators, self.coeff * other.coeff)


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
