## jaxmp

Robot helpers using jax. Goal is to support:
- Basic robot kinematics (FK/IK)
- Trajectory optimization for collision-free motion
- Motion planning, also for collision-free motion

Supported by by `jaxls`.

Performance should be sufficiently fast with CPU.
JIT compilation can take up to 10-15 seconds.

... continue working on reacher.
- [ ] how to make sure that the ball is between the grippers?
- [ ] maybe in the process make the costs easier to set?
- [ ] singularities.