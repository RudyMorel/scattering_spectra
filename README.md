# Scattering covariance

#### REPOSITORY INTO CONSTRUCTION.

This repository implements the *Scattering Covariance* introduced in [1].

It provides an interpretable low-dimensional representation of multi-scale time-series having stationary increments. 

Among other applications, Scattering Covariance can be applied for time-series **analysis** and time-series **generation**. 

## Analysis

Function **load_data** from frontend.py provides time-series from classical models. To load $R=16$ realizations of a Brownian motion with $T=4096$ samples:

```python
X = load_data(name='fbm', R=16, T=4096, H=0.5)
```

 Now use the function **analyze** from frontend.py to obtain scattering covariance coefficients:

```python
RX = analyze(X, cuda=True)
RX
```

The result is an annotated tensor, meaning a tensor of size K x 2 where K is the number of complex coefficients in $RX$. 



Scattering covariance can be used to compare time-series.

Here we will compare *fractional Brownian motion* (fBm) which is a Gaussian model and *multifractal random walk* (MRW) which is non-Gaussian.

```python
# DATA
X_fbm = load_data(name='fbm', R=16, T=2**16, H=0.5)
X_mrw = load_data(name='mrw', R=16, T=2**16, H=0.5, lam=0.15)

# ANALYSIS
RX1 = analyze(X_fbm, J=10, high_freq=0.25, moments='cov', cuda=True)
RX2 = analyze(X_mrw, J=10, high_freq=0.25, moments='cov', cuda=True)
RXs = [RX1.mean('n1'), RX2.mean('n1')]  # average on realizations

# VISUALIZATION
plot_marginal_moments(RXs)
plot_cross_phased(RXs)
plot_modulus(RXs)
```

## Synthesis

...

[1] https://arxiv.org/abs/2204.10177
