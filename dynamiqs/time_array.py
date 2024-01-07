from __future__ import annotations

from abc import abstractmethod
from typing import get_args

import numpy as np
from jax import Array
from jax import numpy as jnp

from ._utils import check_time_array, obj_type_str, type_str
from .types import Scalar
from .utils.array_types import ArrayLike, dtype_complex_to_real, get_cdtype

__all__ = ['totime']


def totime(
    x: (
        ArrayLike
        | callable[[float], Array]
        | tuple[ArrayLike, ArrayLike, ArrayLike]
        | tuple[callable[[float], Array], ArrayLike]
    ),
    *,
    dtype: jnp.dtype | None = None,
) -> TimeArray:
    dtype = dtype or get_cdtype(dtype)  # assume complex by default

    # PWC time array
    if isinstance(x, tuple) and len(x) == 3:
        return _factory_pwc(x, dtype=dtype)
    # modulated time array
    if isinstance(x, tuple) and len(x) == 2:
        return _factory_modulated(x, dtype=dtype)
    # constant time array
    if isinstance(x, get_args(ArrayLike)):
        return _factory_constant(x, dtype=dtype)
    # callable time array
    elif callable(x):
        return _factory_callable(x, dtype=dtype)
    else:
        raise TypeError(
            'For time-dependent arrays, argument `x` must be one of 4 types: (1)'
            ' ArrayLike; (2) 2-tuple with type (function, ArrayLike) where function'
            ' has signature (t: float) -> Array; (3) 3-tuple with type (ArrayLike,'
            ' ArrayLike, ArrayLike); (4) function with signature (t: float) -> Array.'
            f' The provided `x` has type {obj_type_str(x)}.'
        )


def _factory_constant(x: ArrayLike, *, dtype: jnp.dtype) -> ConstantTimeArray:
    x = jnp.asarray(x, dtype=dtype)
    return ConstantTimeArray(x)


def _factory_callable(
    x: callable[[float], Array], *, dtype: jnp.dtype
) -> CallableTimeArray:
    f0 = x(0.0)

    # check type, dtype and device match
    if not isinstance(f0, Array):
        raise TypeError(
            f'The time-dependent operator must be a {type_str(Array)}, but has'
            f' type {obj_type_str(f0)}. The provided function must return an array,'
            ' to avoid costly type conversion at each time solver step.'
        )
    elif f0.dtype != dtype:
        raise TypeError(
            f'The time-dependent operator must have dtype `{dtype}`, but has dtype'
            f' `{f0.dtype}`. The provided function must return an array with the'
            ' same `dtype` as provided to the solver, to avoid costly dtype'
            ' conversion at each solver time step.'
        )

    return CallableTimeArray(x, f0)


def _factory_pwc(
    x: tuple[ArrayLike, ArrayLike, ArrayLike],
    *,
    dtype: jnp.dtype,
) -> PWCTimeArray:
    times, values, array = x

    # get real-valued dtype
    if dtype in (jnp.complex64, jnp.complex128):
        rdtype = dtype_complex_to_real(dtype)
    else:
        rdtype = dtype

    # times
    times = jnp.asarray(times, dtype=rdtype)
    check_time_array(times, arg_name='times')

    # values
    values = jnp.asarray(values, dtype=dtype)
    if values.shape[0] != len(times) - 1:
        raise TypeError(
            'For a PWC array `(times, values, array)`, argument `values` must'
            ' have shape `(len(times)-1, ...)`, but has shape'
            f' {tuple(values.shape)}.'
        )

    # array
    array = jnp.asarray(array, dtype=dtype)
    if array.ndim != 2 or array.shape[-1] != array.shape[-2]:
        raise TypeError(
            'For a PWC array `(times, values, array)`, argument `array` must be'
            f' a square matrix, but has shape {tuple(array.shape)}.'
        )

    factors = [_PWCFactor(times, values)]
    arrays = array[None, ...]  # (1, n, n)
    return PWCTimeArray(factors, arrays)


def _factory_modulated(
    x: tuple[callable[[float], Array], Array],
    *,
    dtype: jnp.dtype,
) -> ModulatedTimeArray:
    f, array = x

    # get real-valued dtype
    if dtype in (jnp.complex64, jnp.complex128):
        rdtype = dtype_complex_to_real(dtype)
    else:
        rdtype = dtype

    # check f
    if not callable(f):
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `f` must'
            f' be a function, but has type {obj_type_str(f)}.'
        )
    f0 = f(0.0)
    if not isinstance(f0, Array):
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `f` must'
            f' return an array, but returns type {obj_type_str(f0)}.'
        )
    if f0.dtype not in [dtype, rdtype]:
        dtypes = f'`{dtype}`' if dtype == rdtype else f'`{dtype}` or `{rdtype}`'
        raise TypeError(
            'For a modulated time array, the array returned by the function must'
            f' have dtype `{dtypes}`, but has dtype `{f0.dtype}`. This is necessary'
            ' to avoid costly dtype conversion at each solver time step.'
        )

    # array
    array = jnp.asarray(array, dtype=dtype)
    if array.ndim != 2 or array.shape[-1] != array.shape[-2]:
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `array` must'
            f' be a square matrix, but has shape {tuple(array.shape)}.'
        )

    factors = [_ModulatedFactor(f, f0)]
    arrays = array[None, ...]  # (1, n, n)
    return ModulatedTimeArray(factors, arrays)


class TimeArray:
    # Subclasses should implement:
    # - the properties: dtype, shape, mT
    # - the methods: __call__, reshape, conj, __neg__, __mul__, __add__

    # Note that a subclass implementation of `__add__` only need to support addition
    # with `Array`, `ConstantTimeArray` and the subclass type itself.

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The data type (numpy.dtype) of the array."""
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the array."""
        pass

    @property
    @abstractmethod
    def mT(self) -> TimeArray:
        """Transposes the last two dimensions of x."""

    @property
    def ndim(self) -> int:
        """The number of dimensions in the array."""
        return len(self.shape)

    @abstractmethod
    def __call__(self, t: Scalar) -> Array:
        """Evaluate at a given time."""
        pass

    @abstractmethod
    def reshape(self, *args: int) -> TimeArray:
        """Returns an array containing the same data with a new shape."""
        pass

    @abstractmethod
    def conj(self) -> TimeArray:
        """Return the complex conjugate, element-wise."""
        pass

    @abstractmethod
    def __neg__(self) -> TimeArray:
        pass

    @abstractmethod
    def __mul__(self, y: ArrayLike) -> TimeArray:
        pass

    def __rmul__(self, y: ArrayLike) -> TimeArray:
        return self * y

    @abstractmethod
    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        pass

    def __radd__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return self + y

    def __sub__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return self + (-y)

    def __rsub__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return y + (-self)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype})'

    def __str__(self) -> str:
        return self.__repr__()


class ConstantTimeArray(TimeArray):
    def __init__(self, x: Array):
        self.x = x

    @property
    def dtype(self) -> np.dtype:
        return self.x.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.x.shape

    @property
    def mT(self) -> TimeArray:
        return ConstantTimeArray(self.x.mT)

    def __call__(self, t: Scalar) -> Array:
        return self.x

    def reshape(self, *args: int) -> TimeArray:
        return ConstantTimeArray(self.x.reshape(*args))

    def conj(self) -> TimeArray:
        return ConstantTimeArray(self.x.conj())

    def __neg__(self) -> TimeArray:
        return ConstantTimeArray(-self.x)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return ConstantTimeArray(self.x * y)

    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(y, get_args(ArrayLike)):
            return ConstantTimeArray(self.x + y)
        elif isinstance(y, ConstantTimeArray):
            return ConstantTimeArray(self.x + y.x)
        else:
            return NotImplemented


class CallableTimeArray(TimeArray):
    def __init__(self, f: callable[[float], Array], f0: Array):
        # f0 carries all the transformation on the shape
        self.f = f
        self.f0 = f0

    @property
    def dtype(self) -> np.dtype:
        return self.f0.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.f0.shape

    @property
    def mT(self) -> TimeArray:
        f = lambda t: self.f(t).mT
        f0 = self.f0.mT
        return CallableTimeArray(f, f0)

    def __call__(self, t: float) -> Array:
        return self.f(t).reshape(*self.shape)

    def reshape(self, *args: int) -> TimeArray:
        f = self.f
        f0 = self.f0.reshape(*args)
        return CallableTimeArray(f, f0)

    def conj(self) -> TimeArray:
        f = lambda t: self.f(t).conj()
        f0 = self.f0.conj()
        return CallableTimeArray(f, f0)

    def __neg__(self) -> TimeArray:
        f = lambda t: -self.f(t)
        f0 = -self.f0
        return CallableTimeArray(f, f0)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        f = lambda t: self.f(t) * y
        f0 = self.f0 * y
        return CallableTimeArray(f, f0)

    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(y, get_args(ArrayLike)):
            f = lambda t: self.f(t) + y
            f0 = self.f0 + y
            return CallableTimeArray(f, f0)
        elif isinstance(y, ConstantTimeArray):
            f = lambda t: self.f(t) + y.x
            f0 = self.f0 + y.x
            return CallableTimeArray(f, f0)
        elif isinstance(y, CallableTimeArray):
            f = lambda t: self.f(t) + y.f(t)
            f0 = self.f0 + y.f0
            return CallableTimeArray(f, f0)
        else:
            return NotImplemented


class _Factor:
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def conj(self) -> _Factor:
        pass

    @abstractmethod
    def __call__(self, t: Scalar) -> Array:
        pass

    @abstractmethod
    def reshape(self, *args: int) -> _Factor:
        pass


class _PWCFactor(_Factor):
    # Defined by a tuple of 2 arrays (times, values), where
    # - times: (nv+1) are the time points between which the PWC factor takes constant
    #          values, where nv is the number of time intervals
    # - values: (..., nv) are the constant values for each time interval, where (...)
    #           is an arbitrary batching size

    def __init__(self, times: Array, values: Array):
        self.times = times  # (nv+1)
        self.values = values  # (..., nv)
        self.nv = self.values.shape[-1]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape[:-1]  # (...)

    def conj(self) -> _Factor:
        return _PWCFactor(self.times, self.values.conj())

    def __call__(self, t: Scalar) -> Array:
        if t < self.times[0] or t >= self.times[-1]:
            return jnp.zeros_like(self.values[..., 0])  # (...)
        else:
            # find the index $k$ such that $t \in [t_k, t_{k+1})$
            idx = jnp.searchsorted(self.times, t, side='right') - 1
            return self.values[..., idx]  # (...)

    def reshape(self, *args: int) -> _Factor:
        return _PWCFactor(self.times, self.values.reshape(*args, self.nv))


class _ModulatedFactor(_Factor):
    # Defined by two objects (f, f0), where
    # - f is a callable that takes a time and returns an array of shape (...)
    # - f0 is the array of shape (...) returned by f(0.0)
    # f0 holds information about the shape of the array returned by f(t).

    def __init__(self, f: callable[[float], Array], f0: Array):
        self.f = f  # (float) -> (...)
        self.f0 = f0  # (...)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.f0.shape

    def conj(self) -> _Factor:
        f = lambda t: self.f(t).conj()
        f0 = self.f0.conj()
        return _ModulatedFactor(f, f0)

    def __call__(self, t: Scalar) -> Array:
        return self.f(t).reshape(self.shape)

    def reshape(self, *args: int) -> _Factor:
        f = self.f
        f0 = self.f0.reshape(*args)
        return _ModulatedFactor(f, f0)


class FactorTimeArray(TimeArray):
    def __init__(
        self, factors: list[_Factor], arrays: Array, static: Array | None = None
    ):
        # factors must be non-empty
        self.factors = factors  # list of length (nf)
        self.arrays = arrays  # (nf, n, n)
        self.n = arrays.shape[-1]
        self.static = jnp.zeros_like(self.arrays[0]) if static is None else static

    @property
    def dtype(self) -> np.dtype:
        return self.arrays.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.factors[0].shape, self.n, self.n)  # (..., n, n)

    def __call__(self, t: float) -> Array:
        values = jnp.stack([x(t) for x in self.factors], axis=-1)  # (..., nf)
        values = values.reshape(*values.shape, 1, 1)  # (..., nf, n, n)
        return (values * self.arrays).sum(-3) + self.static  # (..., n, n)

    def reshape(self, *args: int) -> TimeArray:
        # shape: (..., n, n)
        factors = [x.reshape(*args[:-2]) for x in self.factors]
        return self.__class__(factors, self.arrays, static=self.static)

    def conj(self) -> TimeArray:
        factors = [x.conj() for x in self.factors]
        return self.__class__(factors, self.arrays.conj(), static=self.static.conj())

    def __neg__(self) -> TimeArray:
        return self.__class__(self.factors, -self.arrays, static=-self.static)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return self.__class__(self.factors, self.arrays * y, static=self.static * y)

    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(y, get_args(ArrayLike)):
            static = self.static + y
            return self.__class__(self.factors, self.arrays, static=static)
        elif isinstance(y, ConstantTimeArray):
            static = self.static + y.x
            return self.__class__(self.factors, self.arrays, static=static)
        elif isinstance(y, self.__class__):
            factors = self.factors + y.factors  # list of length (nf1 + nf2)
            arrays = jnp.concatenate((self.arrays, y.arrays))  # (nf1 + nf2, n, n)
            static = self.static + y.static  # (n, n)
            return self.__class__(factors, arrays, static=static)
        else:
            return NotImplemented


class PWCTimeArray(FactorTimeArray):
    # Arbitrary sum of arrays with PWC factors.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # merge all times
        self.times = jnp.unique(jnp.concatenate([x.times for x in self.factors]))


class ModulatedTimeArray(FactorTimeArray):
    # Sum of arrays with callable factors.
    pass
