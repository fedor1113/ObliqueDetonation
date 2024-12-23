# ObliqueDetonation
Oblique detonation simulation in OpenFOAM


This project is based on OpenFOAM v2406 and requires it to work.


After installing OpenFOAM, `detonationFoam` solver should be compiled and installed by using `wmake` inside `applications/detonationFoam` directory in a running OpenFOAM environment.


To perform a convergence study for an inert Riemann problem, change the mesh size to 100, 1000 and then 10000 for `shockTube`, run `./Allclean.py && ./Allrun.py` inside the directory for each new mesh
and for each mesh run `./riemann_exact.py` to get errors in various norms.


To perform a detailed self-convergence order and cross-verification study for a detonation initiation problem (Xu et al., 1997), see the directory called `detonation_initiation_convergence_order_test`. Solve the problem for different numbers N of cells and CFL numbers maxCo and use my `check_convergence.py` or `verify` from NASA to analyze the results. At t=0.4, the solution is still smooth. At t=3, a shock is already formed. For the latter case, a file with a solution from a standalone high-order code is present in the `results` subdirectory to compare against.


To check inert oblique shock test convergence for `detonationFoam` solver run `obliquedetonation` via `./Allclean.py && ./Allrun.py`, putting Q=0 beforehand.


To check that the solver does not break down with actual reactions, do not put Q=0.


For 1D detonation tests, run `detonationTest` with various parameters.
For 2D detonation tests, run `detonation2DTest` with various parameters.
