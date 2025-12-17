```lindblad-sp``` provides code to calculate linear and non-linear response properties of open quantum systems governed by a single-particle (quadratic) Lindblad master equation.

```example-rice-mele``` provides test cases for the functions provided as class member functions of ```LindbladBoseSP``` and ```LindbladFermiSP``` with the Rice-Mele model for a 1D tight-binding chain. Sample response functions include

- Momentum resolved density of particles
- Spectral function
- Energy bands and relaxation rates
- Diamagnetic linear response
- Paramagnetic linear response
- Nonlinear optical response (shift/second harmonic generation)

Key choices are ```species``` which can either be ```bose``` for bosons or ```fermi``` for fermions, and ```is_nambu``` which must be set ```True``` if there are anomalous terms $C\neq 0$ or $\Delta\neq 0$, and can be set ```False``` otherwise.

This code is related to the paper [S. Talkington and M. Claassen, "Linear and non-linear response of quadratic Lindbladians",  npj Quantum Materials 9, 104 (2024)](https://doi.org/10.1038/s41535-024-00709-4).
