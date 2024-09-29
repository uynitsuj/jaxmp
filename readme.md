## jaxmp

Robot helpers using jax. Goal is to support:
- Basic robot kinematics (FK/IK).
- Trajectory optimization for collision-free motion
- Motion planning, also for collision-free motion

Main goals:
- Bare-bones implementation of basic differentiable robot kinematics.
- _any_ robot! (or a large subset of robots).

- Compatibility with other jax-based optimizers (optax, jaxopt? for cvxopt? ... - esque things). Would like to have!
Supported by by `jaxls`.

How can I make the code have less boilerplate?

Implement servoing as resolved-rate motion control.
Sequence-of-constraints mpc: Reactive timing-optimal control of sequential manipulation?

Would like to _not_ implement collision stuff, as much as possible.
--> OK I thought this and tried to look into brax/mjx but it's super entangled with the mujoco mjx library stuff -- and it seems like I already have almost everything (minus sweeping?)

Collisions get brittle when:
- there's too many collisionbodies, and
- the bodies themselves get really small.

IK capability is pretty sensitive to starting pose.

Performance should be sufficiently fast with CPU.
Weirdness:
- It's actually _faster_ on cpu...?

what do I want to do?
- code quality
- rewrite rsrd / "large search space" bimanual options
- motion planning!!! or ~ what rekep does.

- rrt ...? can I use mctx?
how does curobo do path planning?

curobo _can_ be differentiated through, but less boilerplate the better!
Less coupling + parameter swipswash, the better!
> We also provide differentiable PyTorch layers for kinematics and collision
checking for use in neural networks.
rrt? mppi? 

General jaxls wants:
- gpu cholesky
- support solving w/o sparsifying.
- figure out the batching bug (with jaxmp)
- automatical jacobian output caching (could be cool). (thinking about 3dgs-lm). Avoiding re-calculation could be nice. But might make code messier.

- ~ interesting ~ costs!

(maybe collision avoidance is mostly nbd minus large scene stuff, and self-collisions)

Mental note:
- (anygrasp is probably really really good so maybe it's OK to not do grasping)

Is there an edge over curobo?
curobo uses L-BFGS.
- Being able to optimize the global object transformation is nice.
- You should be able to optimize the full bimanual setup.
A factor-graph like setup is nice for trajectories, in general.

Main comparison points:
- existing ik / motion planners
mink, kinpy, ...?

- existing diff robot frameworks
Robotics Toolbox for Python --> doesn't use torch/jax -- hard to differentiate through
pypose

- existing NL solvers
theseus
pypose LM solver?

- existing QP solvers, etc
(qpsolver)
why do nl?
...?

Rekep uses dual annealing + slsqp, which... feels similar to just running LM?



I feel like my knowledge is weak.
MPC? Optimal control? NL vs QP? ... I am dummy.

Anyteleop?

Do I even want to continue on the collision direction? 
In which case I would want to do swept volumes, ...

Lol. Having a good collision body in itself is a really really hard problem.
And really annoying.
Maybe we can solve for that instead :clown:.

Would-be-nices:
- for freeing base, have a different rest cost for the world-to-base transform (to keep robot relative stable)

A relaxed-ik / learned C-space distribution could also be nice -- as long as it's differentiable.
Which it should be!

A _lot_ of the teleoperating systems need to solve the fixed(?) transform problem!
