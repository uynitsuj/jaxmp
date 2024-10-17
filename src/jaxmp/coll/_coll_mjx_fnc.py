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

from mujoco.mjx._src.collision_types import Collision
from mujoco.mjx._src.collision_driver import _COLLISION_FUNC
from mujoco.mjx import GeomType

from jaxmp.coll._coll_mjx_types import CollGeom, Plane, Sphere, Capsule, Ellipsoid
from jaxmp.coll._coll_robot import RobotColl


COLL_TYPES = {
    Plane: GeomType.PLANE,
    Sphere: GeomType.SPHERE,
    Capsule: GeomType.CAPSULE,
    Ellipsoid: GeomType.ELLIPSOID,
    RobotColl: GeomType.CAPSULE,
}

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
    _dist = jnp.where(
        _dist > 0,
        _dist + 0.5 * eta,
        0.5 / eta * (_dist + eta)**2
    )
    _dist = jnp.maximum(_dist, 0.0)
    _dist = -_dist
    return _dist


def collide(geom_0: CollGeom, geom_1: CollGeom) -> Collision:
    broadcast_shape = jnp.broadcast_shapes(
        geom_0.get_batch_axes(), geom_1.get_batch_axes()
    )
    geom_0 = geom_0.broadcast_to(*broadcast_shape).reshape(-1,)
    geom_1 = geom_1.broadcast_to(*broadcast_shape).reshape(-1,)

    func, geom_0, geom_1 = _get_coll_func(geom_0, geom_1)

    # Quick wrapper for mjx collision functions...
    @dataclass
    class Model:
        geom_size: jax.Array

    @dataclass
    class Data:
        geom_xpos: jax.Array
        geom_xmat: jax.Array

    model = Model(geom_size=jnp.array([geom_0.size, geom_1.size]))
    data = Data(
        geom_xpos=jnp.array([geom_0.pos, geom_1.pos]),
        geom_xmat=jnp.array([geom_0.mat, geom_1.mat])
    )

    result = func(model, data, None, jnp.array([0, 1]))
    result = jax.tree_util.tree_map(
        lambda x: x.reshape(broadcast_shape + (-1,)), result
    )
    return result


def _get_coll_func(geom_0, geom_1) -> tuple[Callable, CollGeom, CollGeom]:
    type_0, type_1 = COLL_TYPES[type(geom_0)], COLL_TYPES[type(geom_1)]
    for key, func in _COLLISION_FUNC.items():
        if key == (type_0, type_1):
            return func, geom_0, geom_1
        elif key == (type_1, type_0):
            return func, geom_1, geom_0
    else:
        raise NotImplementedError(
            f"Collision function not found for {geom_0} and {geom_1}."
        )
