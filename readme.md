# jaxmp

JAX-based robot library, focusing on modularity and ease of use.

We formulate goals and tasks as a nonlinear least squares problem, and uses [jaxls](https://github.com/brentyi/jaxls) to solve it in a sparse manner.

Includes:
- Differentiable forward kinematics.
- Batched + broadcastable collision checking, with a thin wrapper around MJX.
- Common cost factors (e.g., EE pose, collision, manipulability).

Supports:
- Arbitrary costs, as long as autodiff Jacobians are feasible.
- A wide range of robots!
