# Trajectory Utils

Code herein provides tools for computing or manipulating trajectories for robotics or control systems.

## Installation


## Contents

An SCP (Sequential Convex Programming) control trajectory solver for a differential drive robot:

```
diff_drive_solver.py
run_diff_drive.py
```

An SCP trajectory solver for a cartpole (with a particular physical twin - more to come) driven by force on the pole hub (a belt driven "cart"):

```
cartpole_solver_force.py
run_cartpole_force.py
```

An SCP trajectory solver for a cartpole (with a particular physical twin - more to come) driven by a velocity servo on the pole hub (a belt driven "cart"):

```
cartpole_solver_velocity.py
run_cartpole_velocity.py
```

