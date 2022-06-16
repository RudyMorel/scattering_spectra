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



...



Here we will compare *fractional Brownian motion* (fBm) which is a Gaussian model and *multifractal random walk* (MRW) which non-Gaussian.

```python
X_fbm = load_data(name='fbm', R=16, T=4096, H=0.5)
X_mrw = load_data(name='mrw', R=16, T=4096, H=0.5, lam=0.15)
```

See file main.py, it computes this representation for some parametrized models of time-series. 

## Synthesis

...

[1] https://arxiv.org/abs/2204.10177
