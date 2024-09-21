###################################################################
 Dynamical Structure Factor
###################################################################


The dynamical structure factor  S(\mathbf{q}, \omega)  is a central quantity in the study of scattering experiments, such as neutron or X-ray scattering, where it describes how the system scatters as a function of both momentum transfer  \mathbf{q}  and energy transfer  \hbar\omega . It encapsulates the time-dependent correlations in a system’s density or atomic positions, offering insight into both the static and dynamic properties of matter.

The dynamical structure factor can be understood through the following key steps:

*************************************************
 Definition of the Dynamical Structure Factor
*************************************************

The dynamical structure factor  S(\mathbf{q}, \omega)  is related to the time-dependent density correlations of a system. It is defined as the Fourier transform, both in space and time, of the density-density correlation function:


.. math::
   
   S(\mathbf{q}, \omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty} dt \, e^{i\omega t} \int d^3r \, e^{-i\mathbf{q} \cdot \mathbf{r}} \langle \rho(\mathbf{r}, t) \rho(0, 0) \rangle

Here,

```
•	 \\mathbf{q}  is the momentum transfer (scattering vector),
•	 \\omega  is the energy transfer related to the frequency of the system’s excitations,
•	 \\rho(\\mathbf{r}, t)  is the particle density operator at position  \\mathbf{r}  and time  t ,
•	 \\langle \\rho(\\mathbf{r}, t) \\rho(0, 0) \\rangle  is the time-dependent density correlation function, also known as the density-density correlation function.

```

Key components:

```
•	The density operator  \\rho(\\mathbf{r}, t)  counts the number of particles in a small volume around position  \\mathbf{r}  at time  t , and can be written as:

```

.. math::

   \rho(\mathbf{r}, t) = \sum_{j=1}^{N} \delta(\mathbf{r} - \mathbf{r}_j(t))

where  \mathbf{r}_j(t)  is the position of the  j -th particle at time  t , and  N  is the total number of particles.
•	The density correlation function  \langle \rho(\mathbf{r}, t) \rho(0, 0) \rangle  provides a measure of how particle densities at time  t  and time  t=0  are correlated, which reflects the system’s dynamics over time.

1. The Intermediate Scattering Function S(\mathbf{q}, t)

The dynamical structure factor is the Fourier transform in time of the intermediate scattering function  S(\mathbf{q}, t) . The intermediate scattering function describes how density fluctuations propagate in time and is given by:


.. math::

   S(\mathbf{q}, t) = \frac{1}{N} \sum_{i,j} \langle e^{i\mathbf{q} \cdot (\mathbf{r}_j(t) - \mathbf{r}_i(0))} \rangle

This equation measures the correlation between the positions of particles at different times. It represents the dynamic structure in the space and time domain, giving information about the evolution of density fluctuations.

Physical Interpretation:

```
•	At  t = 0 ,  S(\\mathbf{q}, t=0)  gives the static structure factor  S(\\mathbf{q}) , which is a measure of spatial correlations (static arrangement) of particles in the system.
•	As  t \\to \\infty ,  S(\\mathbf{q}, t)  decays, representing the decay of correlations due to particle movement over time.

```

***********************************************************************************
 Fourier Transform in Time: Relating S(\mathbf{q}, t) to S(\mathbf{q}, \omega)
***********************************************************************************

   
The dynamical structure factor  S(\mathbf{q}, \omega)  is obtained by taking the Fourier transform of  S(\mathbf{q}, t)  with respect to time:

.. math::

   S(\mathbf{q}, \omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty} dt \, e^{i\omega t} S(\mathbf{q}, t)

This transform shifts the description from the time domain (where we are looking at how density correlations evolve with time) to the frequency domain, which provides insight into the spectrum of excitations in the system.

1. Fluctuation-Dissipation Theorem and Detailed Balance

In thermal equilibrium, the dynamical structure factor is related to the fluctuation-dissipation theorem, which links it to the system’s response to perturbations. The theorem states that the response function (describing how the system reacts to an external perturbation) is directly connected to the spontaneous fluctuations in the system.

The fluctuation-dissipation theorem introduces the concept of detailed balance:

.. math::

   S(\mathbf{q}, -\omega) = e^{-\beta\hbar\omega} S(\mathbf{q}, \omega)

where  \beta = \frac{1}{k_B T}  is the inverse temperature, and  \hbar \omega  is the energy exchange during scattering. This relationship implies that for  \hbar \omega > 0 , there is a preference for energy to be transferred from the system to the scattering particles (such as neutrons or photons) rather than absorbed, reflecting the balance between absorption and emission processes in thermal equilibrium.

***********************************
 Classical and Quantum Limits
***********************************

 
In the classical limit (large temperatures  T ), where  \hbar \omega \ll k_B T , the dynamical structure factor simplifies and becomes symmetric in  \omega :

.. math::

   S(\mathbf{q}, \omega) = S(\mathbf{q}, -\omega)

This reflects the fact that, at high temperatures, the forward and reverse processes (emission and absorption) are equally probable.

In the quantum limit, where  \hbar \omega  is comparable to or larger than  k_B T , detailed balance becomes important, and  S(\mathbf{q}, \omega)  is no longer symmetric. Quantum effects dominate, such as the difference between absorption and stimulated emission probabilities.

***************************************
 Example: Harmonic Oscillator Model
***************************************

 
For systems with well-defined vibrational modes, such as a collection of harmonic oscillators, the dynamical structure factor can be calculated analytically. If particles oscillate harmonically with frequency  \omega_0 , the intermediate scattering function takes the form:

.. math::

   S(\mathbf{q}, t) = e^{-i \omega_0 t}

Taking the Fourier transform with respect to time, we find:

.. math::

   S(\mathbf{q}, \omega) = \delta(\omega - \omega_0)

This implies that the dynamical structure factor has a sharp peak at the oscillation frequency  \omega_0 , representing the energy transfer corresponding to a single vibrational mode.

************************************************************
 Practical Applications of the Dynamical Structure Factor
************************************************************

 
The dynamical structure factor is measurable in neutron and X-ray scattering experiments and provides key insights into the dynamics of various physical systems:

```
•	Phonon excitations in solids:  S(\\mathbf{q}, \\omega)  can reveal the dispersion relations of phonons (collective lattice vibrations).
•	Diffusion processes: In liquids, the broadening of the peaks in  S(\\mathbf{q}, \\omega)  reflects particle diffusion.
•	Critical phenomena: Near phase transitions, critical slowing down of dynamics can be observed in the time dependence of the correlation functions.

```

**************
Conclusion
**************


The dynamical structure factor  S(\mathbf{q}, \omega)  provides a comprehensive picture of the time-dependent density fluctuations in a material, describing both the spatial and temporal correlations. It can be calculated from MD simulations by first computing the intermediate scattering function  S(\mathbf{q}, t)  and then Fourier transforming it to obtain  S(\mathbf{q}, \omega) . The fluctuation-dissipation theorem, detailed balance, and the nature of excitations in the system (such as phonons or diffusive modes) play crucial roles in interpreting the results.
