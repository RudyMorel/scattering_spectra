# Scattering covariance

This repository implements the *Scattering Covariance* introduced in [1].

It provides an interpretable low-dimensional representation of multi-scale time-series that can be used for time-series **analysis** and **generation**.

Among other applications are the assessment of **self-similarity** in the data and the detection of **non-Gaussianity**.

## Installation

Run the commands below to install the required packages.

```bash
pip install git+https://github.com/RudyMorel/scattering_covariance
```

## Analysis

The *Scattering Covariance* provides a dashboard to analyze time-series.

Standard model of time series can be loaded using **load_data** from `frontend.py`. The function **analyze** computes the *Scattering Covariance*, it can be visualized using the function **plot_dashboard**.

```python
# DATA
x1 = load_data(process_name='fbm', R=256, T=32768)
x2 = load_data(process_name='mrw', R=256, T=32768, lam=0.1)
x3 = load_data(process_name='smrw', R=256, T=32768, lam=0.3, 
               gamma=1/32768/256, K=0.03, alpha=0.23, beta=0.23)

# ANALYSIS
Rx1 = analyze(x1, J=8, high_freq=0.25, cuda=True, nchunks=64)
Rx2 = analyze(x2, J=8, high_freq=0.25, cuda=True, nchunks=64)
Rx3 = analyze(x3, J=8, high_freq=0.25, cuda=True, nchunks=64)

# VISUALIZATION
plot_dashboard([Rx1, Rx2, Rx3], labels=['fbm', 'mrw', 'smrw'])

save_figure('dashboard_fbm_mrw_smrw.png')
```

![alt text](illustration/dashboard_fbm_mrw_smrw.png "Scattering Spectra comparison")

The dashboard consists of 4 spectra that can be interpreted as follows:
- $\Phi_1(x)[j]$ are the sparsity factors. The lower these coefficients the sparser the signal $x$.
- $\Phi_2(x)[j]$ is the standard wavelet spectrum. A steep curve indicates long-range dependencies.
-  $\Phi_3(x)[a]$ is the phase-modulus cross-spectrum. 
    - $|\Phi_3(x)[a]|$ is zero if the underlying process is invariant to sign change $x \overset{d}{=}-x$. \
    These coefficients quantify non-invariance to sign-change often called "skewness".
    - Arg $\Phi_3(x)[a]$ is zero if the underlying process $x$ is time-symmetric: $x(t) \overset{d}{=}x(-t)$. 
    These coefficients quantify time-asymmetry. They are typically non-zero for processes with time-causality.
-  $\Phi_4(x)[a,b]$ is the modulus cross-spectrum. 
    - $|\Phi_4(x)[a,b]|$ quantify envelope dependencies. Steep curves indicate long-range dependencies.
    - Arg $\Phi_4(x)[a,b]$ quantify envelope time-asymmetry. These coefficients are typically non-zero for processes with time-causality.

For further interpretation see [1].

## Self-Similarity
Self-Similar             |  Not Self-Similar
:-------------------------:|:-------------------------:
![alt text](illustration/self_similar.png "Self-Similar example") *Self-similarity distance: 2.1* | ![alt text](illustration/not_self_similar.png "Not Self-Similar example") *Self-similarity distance: 22.0*

Processes encountered in Finance, Seismology, Physics may exhibit a form of regularity called self-similarity. Also refered as scale invariance, it states that the process is similar at different scales. 
Assessing such property on a single realization is a hard problem. 

![alt text](illustration/wide_sense_self_similarity.png "Wide-sense Self-Similarity")

Similarly to time-stationarity that has a wide-sense definition that can be tested statistically on the covariance of a process $X(t)$, we introduced in [1] a wide-sense definition of self-similarity called **wide-sense Self-Similarity**. It can be tested on a covariance matrix across times $t,t'$ and scales $j,j'$.

The function **self_simi_obstruction_score** from `frontend.py` assesses self-similarity on a time-series.
 

## Generation

A model of the process $X$ can be defined from the *Scattering Covariance*. Such model can be sampled using gradient descent [1].

Function **generate** from `frontend.py` takes observed data $X$ as input and return realizations of our model of $X$.

```python
# DATA
x = load_data(process_name='smrw', R=1, T=4096, lam=0.3,
              gamma=1/4096/256, K=0.03, alpha=0.23, beta=0.23) # a B x T array

# GENERATION
x_gen = generate(x, J=9, S=1, it=1000, cuda=True, tol_optim=1e-3) # a S x T array

# VISUALIZATION
fig, axes = plt.subplots(2,1, figsize=(10,5))
axes[0].plot(np.diff(x)[0,0,:], color='lightskyblue', linewidth=0.5)
axes[1].plot(np.diff(x_gen)[0,0,:], color='coral', linewidth=0.5)
axes[0].set_ylim(-3,3)
axes[1].set_ylim(-3,3)
```

![alt text](illustration/generation.png "Generation of a signal")

[1] "Scale Dependencies and Self-Similarity Through Wavelet Scattering Covariance"

Rudy Morel et al. - https://arxiv.org/abs/2204.10177
