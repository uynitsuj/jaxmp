"""
Differentiable robot collision model, implemented in JAX.
Supports World- and self- collision detection (returns signed distance).
"""

from __future__ import annotations

from typing import Optional, cast

import jax_dataclasses as jdc

from jax import Array
from jaxtyping import Int, Float
import jax.numpy as jnp

from jaxmp.collision_types import CollBody, SphereColl, CapsuleColl, PlaneColl, HalfSpaceColl, RobotColl


def colldist_from_sdf(
    _dist: Float[Array, "n_dists"],
    eta: float = 0.05,
) -> Float[Array, "n_dists"]:
    """
    Convert a signed distance field to a collision distance field,
    based on https://arxiv.org/pdf/2310.17274#page=7.39.

    Args: 
        dist: Signed distance field values, where sdf > 0 means collision (n_dists,).
        eta: Distance threshold for the SDF.

    Returns:
        Collision distance field values (n_dists,).
    """
    _dist = jnp.maximum(_dist, -eta)
    _dist = jnp.where(
        _dist > 0,
        _dist + 0.5 * eta,
        0.5 / eta * (_dist + eta)**2
    )
    _dist = jnp.maximum(_dist, 0.0)
    return _dist

def sdf_collbody(
    coll_0: CollBody,
    coll_1: CollBody,
) -> Float[Array, "coll_0 coll_1"]:
    """Distance between two collision bodies. sdf > 0 means collision."""
    # Robot self-collision
    if isinstance(coll_0, RobotColl) and isinstance(coll_1, RobotColl) and coll_0 == coll_1:
        return _dist_cap_cap(coll_0, coll_1) * coll_0.self_coll_matrix

    # Sphere-Sphere collision
    if isinstance(coll_0, SphereColl) and isinstance(coll_1, SphereColl):
        return _dist_sph_sph(coll_0, coll_1)

    # Sphere-HalfSpace collision
    elif isinstance(coll_0, SphereColl) and isinstance(coll_1, HalfSpaceColl):
        return _dist_sph_halfspace(coll_0, coll_1)
    elif isinstance(coll_1, SphereColl) and isinstance(coll_0, HalfSpaceColl):
        return _dist_sph_halfspace(coll_1, coll_0).T

    # Sphere-Plane collision
    elif isinstance(coll_0, SphereColl) and isinstance(coll_1, PlaneColl):
        return _dist_sph_plane(coll_0, coll_1)
    elif isinstance(coll_1, SphereColl) and isinstance(coll_0, PlaneColl):
        return _dist_sph_plane(coll_1, coll_0).T

    # Capsule-Capsule collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, CapsuleColl):
        return _dist_cap_cap(coll_0, coll_1)

    # Sphere-Capsule collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, SphereColl):
        return _dist_sph_cap(coll_0, coll_1)
    elif isinstance(coll_1, CapsuleColl) and isinstance(coll_0, SphereColl):
        return _dist_sph_cap(coll_1, coll_0).T

    # Capsule-HalfSpace collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, HalfSpaceColl):
        return _dist_cap_halfspace(coll_0, coll_1)
    elif isinstance(coll_1, CapsuleColl) and isinstance(coll_0, HalfSpaceColl):
        return _dist_cap_halfspace(coll_1, coll_0).T

    # Capsule-Plane collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, PlaneColl):
        return _dist_cap_plane(coll_0, coll_1)
    elif isinstance(coll_1, CapsuleColl) and isinstance(coll_0, PlaneColl):
        return _dist_cap_plane(coll_1, coll_0).T

    else:
        raise NotImplementedError(f"Collision between {type(coll_0)} and {type(coll_1)} not implemented.")

def _dist_sph_sph(
    sphere: SphereColl,
    other: SphereColl,
) -> Float[Array, "sph_0 sph_1"]:
    """Return distance between two `Spheres`. sdf > 0 means collision."""
    n_spheres_sph_0 = sphere.centers.shape[0]
    n_spheres_sph_1 = other.centers.shape[0]

    # Without the `+ 1e-7`, the jacobian becomes unstable / returns NaNs.
    _dist = jnp.linalg.norm(
        sphere.centers[:, None] - other.centers[None] + 1e-7, axis=-1
    )
    sdf = (sphere.radii[:, None] + other.radii[None]) - _dist

    assert sdf.shape == (n_spheres_sph_0, n_spheres_sph_1)
    return sdf

def _dist_sph_plane(
    sphere: SphereColl,
    plane: PlaneColl,
) -> Float[Array, "sph_0"]:
    """Return the distance between a `Plane` and a point. sdf > 0 means collision."""
    n_spheres = sphere.centers.shape[0]
    _dist = jnp.abs(jnp.dot(sphere.centers - plane.point, plane.normal))
    sdf = (sphere.radii - _dist)[:, None]
    assert sdf.shape == (n_spheres, 1)
    return sdf

def _dist_sph_halfspace(
    sphere: SphereColl,
    plane: HalfSpaceColl,
) -> Float[Array, "sph_0"]:
    """Return the distance between a halfspace and a point. sdf > 0 means collision."""
    n_spheres = sphere.centers.shape[0]
    _dist = jnp.dot(sphere.centers - plane.point, plane.normal)
    sdf = (sphere.radii - _dist)[:, None]
    assert sdf.shape == (n_spheres, 1)
    return sdf

def _dist_cap_cap(
    cap_0: CapsuleColl,
    cap_1: CapsuleColl,
) -> Float[Array, "cap_0 cap_1"]:
    a0, a1 = cap_0.centerline
    b0, b1 = cap_1.centerline
    _dist = CapsuleColl.dist_between_seg(a0, a1, b0, b1)
    sdf = (cap_0.radii[:, None] + cap_1.radii[None, :]) - _dist
    assert sdf.shape == (cap_0.radii.shape[0], cap_1.radii.shape[0])
    return sdf

def _dist_sph_cap(
    cap: CapsuleColl,
    sph: SphereColl,
) -> Float[Array, "cap sph"]:
    a0, a1 = cap.centerline
    _dist = CapsuleColl.dist_between_seg(a0, a1, sph.centers, sph.centers)
    sdf = (cap.radii[:, None] + sph.radii[None, :]) - _dist
    assert sdf.shape == (cap.radii.shape[0], sph.radii.shape[0])
    return sdf

def _dist_cap_plane(
    cap: CapsuleColl,
    plane: PlaneColl,
) -> Float[Array, "cap 1"]:
    a0, a1 = cap.centerline
    t = jnp.dot(plane.point - a0, plane.normal) / (jnp.dot(a1 - a0, plane.normal) + 1e-7)
    t = jnp.clip(t, 0.0, 1.0)
    point = a0 + t[:, None] * (a1 - a0)
    _dist = jnp.abs(jnp.dot(point - plane.point, plane.normal))
    sdf = (cap.radii - _dist)[:, None]
    assert sdf.shape == (cap.radii.shape[0], 1)
    return sdf

def _dist_cap_halfspace(
    cap: CapsuleColl,
    plane: HalfSpaceColl,
) -> Float[Array, "cap"]:
    a0, a1 = cap.centerline
    t = jnp.dot(plane.point - a0, plane.normal) / (jnp.dot(a1 - a0, plane.normal) + 1e-7)
    t = jnp.clip(t, 0.0, 1.0)
    point = a0 + t[:, None] * (a1 - a0)
    _dist = jnp.dot(point - plane.point, plane.normal)
    sdf = (cap.radii - _dist)[:, None]
    assert sdf.shape == (cap.radii.shape[0], 1)
    return sdf
