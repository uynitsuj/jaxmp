"""
Collision checking, by calling the mjx collision functions.

Some caveats:
- Returns all collision data: distance, normal, and point.
`from mujoco.mjx._src.math.make_frame` may slow things down.

- Seems ~2x slower than the old implementation.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float

from mujoco.mjx._src.collision_driver import _COLLISION_FUNC
from mujoco.mjx import GeomType

from jaxmp.coll._coll_mjx_types import (
    CollGeom,
    Plane,
    Sphere,
    Capsule,
    Ellipsoid,
    Convex,
    Cylinder,
)


MJX_TO_COLL: dict[GeomType, type[CollGeom]] = {
    GeomType.PLANE: Plane,
    GeomType.SPHERE: Sphere,
    GeomType.CAPSULE: Capsule,
    GeomType.ELLIPSOID: Ellipsoid,
    GeomType.MESH: Convex,
    GeomType.CYLINDER: Cylinder,
}
COLL_TO_MJX: dict[type[CollGeom], GeomType] = {v: k for k, v in MJX_TO_COLL.items()}


@jdc.pytree_dataclass
class Collision:
    # Copy of `mujoco.mjx._src.collision_types`, but as a dataclass.
    # Track the largest contact interpenetration / closest distance, reducing over codim.
    dist: Float[jax.Array, "*batch"]
    pos: Float[jax.Array, "*batch 3"]
    frame: Float[jax.Array, "*batch 3 3"]

    @staticmethod
    def from_broadcast_shape(broadcast_shape) -> Collision:
        return Collision(
            dist=jnp.full(broadcast_shape, jnp.inf),
            pos=jnp.zeros(broadcast_shape + (3,)),
            frame=jnp.zeros(broadcast_shape + (3, 3)),
        )


def colldist_from_sdf(
    _dist: jax.Array,
    eta: float = 0.05,
) -> jax.Array:
    """
    Convert a signed distance field to a collision distance field,
    based on https://arxiv.org/pdf/2310.17274#page=7.39.

    Args:
        dist: Signed distance field values, where sdf < 0 means collision (n_dists,).
        eta: Distance threshold for the SDF.

    Returns:
        Collision distance field values (n_dists,).
    """
    _dist = -_dist
    _dist = jnp.maximum(_dist, -eta)
    _dist = jnp.where(_dist > 0, _dist + 0.5 * eta, 0.5 / eta * (_dist + eta) ** 2)
    _dist = jnp.maximum(_dist, 0.0)
    _dist = -_dist
    return _dist


@jdc.jit
def collide(geom_0: CollGeom, geom_1: CollGeom) -> Collision:
    if not isinstance(geom_0, Convex) and not isinstance(geom_1, Convex):
        collision = _collide(geom_0, geom_1)

    elif not isinstance(geom_0, Convex) and isinstance(geom_1, Convex):
        collision = _collide_coll_convex(geom_0, geom_1)

    elif isinstance(geom_0, Convex) and not isinstance(geom_1, Convex):
        collision = _collide_coll_convex(geom_1, geom_0)

    elif isinstance(geom_0, Convex) and isinstance(geom_1, Convex):
        collision = _collide_convex_convex(geom_0, geom_1)

    else:
        raise ValueError(f"Cannot collide {geom_0} and {geom_1}.")

    return collision


@jdc.jit
def _collide(
    geom_0: CollGeom,
    geom_1: CollGeom,
) -> Collision:
    # Broadcast, and flatten, before feeding it in as input.
    # Each `Convex` body should have a single unique mesh!
    broadcast_shape = jnp.broadcast_shapes(
        geom_0.get_batch_axes(), geom_1.get_batch_axes()
    )
    geom_0 = geom_0.broadcast_to(*broadcast_shape).reshape(-1,)
    geom_1 = geom_1.broadcast_to(*broadcast_shape).reshape(-1,)

    # Retrieve the function -- while also shuffling all arguments.
    func, geom_0, geom_1 = _get_coll_func(geom_0, geom_1)

    # Quick wrapper for mjx collision functions...
    @dataclass
    class Model:
        geom_size: jax.Array
        mesh_convex: list

    @dataclass
    class Data:
        geom_xpos: jax.Array
        geom_xmat: jax.Array

    @jdc.pytree_dataclass
    class FunctionKey:
        types: jdc.Static[tuple]
        data_ids: jdc.Static[tuple]

    model = Model(
        geom_size=jnp.array([geom_0.size, geom_1.size]),
        mesh_convex=[
            None
            if not isinstance(geom_0, Convex)
            else jax.tree.map(
                lambda x: x, geom_0.mesh_info
            ),  # to shake off __mutability__.
            None
            if not isinstance(geom_1, Convex)
            else jax.tree.map(lambda x: x, geom_1.mesh_info),
        ],
    )
    data = Data(
        geom_xpos=jnp.array(
            [
                geom_0.pos
                if not isinstance(geom_0, Convex)
                else geom_0.pos + geom_0.offset_to_origin,
                geom_1.pos
                if not isinstance(geom_1, Convex)
                else geom_1.pos + geom_1.offset_to_origin,
            ]
        ),
        geom_xmat=jnp.array([geom_0.mat, geom_1.mat]),
    )
    key = FunctionKey(
        types=(COLL_TO_MJX[type(geom_0)], COLL_TO_MJX[type(geom_1)]),
        data_ids=(0, 1),
    )

    result = func(model, data, key, jnp.array([0, 1]))
    dist = result[0].reshape(broadcast_shape + (-1,))
    codim = dist.shape[-1]
    pos = result[1].reshape(broadcast_shape + (3, codim))
    frame = result[2].reshape(broadcast_shape + (3, 3, codim))

    # For now, return the _closest_ distance / collision.
    idx = jnp.argmin(dist, axis=-1)
    dist = jnp.take_along_axis(dist, idx[..., None], axis=-1)[..., 0]
    pos = jnp.take_along_axis(pos, idx[..., None, None], axis=-1)[..., 0]
    frame = jnp.take_along_axis(frame, idx[..., None, None, None], axis=-1)[..., 0]
    return Collision(dist=dist, pos=pos, frame=frame)


def _get_coll_func(geom_0, geom_1) -> tuple[Callable, CollGeom, CollGeom]:
    for key, func in _COLLISION_FUNC.items():
        if key[0] not in MJX_TO_COLL or key[1] not in MJX_TO_COLL:
            continue
        if isinstance(geom_0, MJX_TO_COLL[key[0]]) and isinstance(
            geom_1, MJX_TO_COLL[key[1]]
        ):
            return func, geom_0, geom_1
        elif isinstance(geom_0, MJX_TO_COLL[key[1]]) and isinstance(
            geom_1, MJX_TO_COLL[key[0]]
        ):
            return func, geom_1, geom_0
    else:
        raise NotImplementedError(
            f"Collision function not found for {geom_0} and {geom_1}."
        )


def _set_value_at_axis(
    arr: jax.Array, value: jax.Array, idx: jax.Array, axis: int | tuple[int]
):
    if isinstance(axis, int):
        axis = (axis,)
    axis_det = list(range(len(axis)))

    # Trick to set values at the correct axis.
    # The expand-dims is important! E.g.,
    # > foo = jnp.zeros((2,))
    # > foo.at[0].set(foo) --> Fails
    # > jnp.expand_dims(foo, 0).at[:, 0].set(foo) --> Works!
    arr = jnp.moveaxis(arr, axis, axis_det)
    arr = jnp.expand_dims(arr, 0).at[:, idx].set(value)[0]

    return jnp.moveaxis(arr, axis_det, axis)


def _collide_coll_convex(geom_0: CollGeom, geom_1: Convex) -> Collision:
    # Will broadcast across meshes -- so num_meshes should be 1.
    broadcast_shape = jnp.broadcast_shapes(
        geom_0.get_batch_axes(), geom_1.get_batch_axes()
    )
    _geom_0 = geom_0.broadcast_to(*broadcast_shape)
    _geom_1 = geom_1.broadcast_to(*broadcast_shape)

    if isinstance(_geom_0, Convex):
        in_axes_0 = _get_mapping_axes_convex(_geom_0, mesh_axis=_geom_1.mesh_axis)[1]
    else:
        in_axes_0 = _geom_1.mesh_axis

    _geom_1, in_axes_1, out_axes_1 = _get_mapping_axes_convex(_geom_1)
    collision = jax.vmap(
        _collide,
        in_axes=(in_axes_0, in_axes_1),
        out_axes=out_axes_1,
    )(_geom_0, _geom_1)
    breakpoint()
    return collision


def _collide_convex_convex(geom_0: Convex, geom_1: Convex) -> Collision:
    broadcast_shape = jnp.broadcast_shapes(
        geom_0.get_batch_axes(), geom_1.get_batch_axes()
    )
    _geom_0 = geom_0.broadcast_to(*broadcast_shape)
    _geom_1 = geom_1.broadcast_to(*broadcast_shape)

    if _geom_0.mesh_axis == _geom_1.mesh_axis:
        _geom_0, in_axes_0, out_axes_0 = _get_mapping_axes_convex(_geom_0)
        _geom_1, in_axes_1, out_axes_1 = _get_mapping_axes_convex(_geom_1)

        collision = jax.vmap(
            _collide,
            in_axes=(in_axes_0, in_axes_1),
            out_axes=out_axes_0,
        )(_geom_0, _geom_1)

    else:
        # Known bug: (collision.dist).shape != broadcast_shape.
        # Nested `vmap` will blow up memory requirements for `jaxls`.
        init_coll = Collision.from_broadcast_shape(broadcast_shape)

        def body_fun(i, collision):
            coll = _collide_coll_convex(
                Convex.slice_along_mesh_axis(_geom_0, i, keepdim=False),
                _geom_1,
            )
            collision = jax.tree.map(
                lambda x, y: _set_value_at_axis(x, y, i, _geom_0.mesh_axis),
                collision,
                coll,
            )
            return collision

        collision = jax.lax.fori_loop(0, _geom_0.num_meshes, body_fun, init_coll)

    return collision

def _get_mapping_axes_convex(geom: Convex, mesh_axis: int | None = None) -> tuple[Convex, Convex, Collision]:
    in_axes = jax.tree.map(lambda _: 0, geom)

    # If `geom` is a single mesh, we pre-emptively slice it,
    # and don't broadcast across the mesh axis.
    # If num_mesh=1, then we allow mesh_axis.ndim != num_mesh.

    if mesh_axis is None:
        mesh_axis = geom.mesh_axis

    if geom.num_meshes == 1:
        mesh_info = jax.tree_map(lambda x: jnp.take(x, 0, 0), geom.mesh_info)
        with jdc.copy_and_mutate(geom, validate=False) as geom:
            geom.offset_to_origin = jnp.take(geom.offset_to_origin, 0, 0)
            geom.mesh_info = mesh_info
        mesh_info_axes = jax.tree_map(lambda x: None, geom.mesh_info)
        with jdc.copy_and_mutate(in_axes, validate=False) as in_axes:
            in_axes.mesh_info = mesh_info_axes
            in_axes.offset_to_origin = None

    with jdc.copy_and_mutate(in_axes, validate=False) as in_axes:
        in_axes.pos = mesh_axis
        in_axes.mat = mesh_axis
        in_axes.size = mesh_axis
        in_axes.num_meshes = 1

    with jdc.copy_and_mutate(geom, validate=False) as geom:
        geom.num_meshes = 1

    out_axes = Collision(
        dist=mesh_axis,
        pos=mesh_axis,
        frame=mesh_axis
    )

    return geom, in_axes, out_axes