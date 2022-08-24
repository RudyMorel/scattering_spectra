import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name='scatcov',
    version='1.0',
    author='Rudy Morel',
    author_email='rudy.morel@ens.fr',
    description=
    'Analysis and generation of time-series with Scattering Covariance',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RudyMorel/scattering_covariance',
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages())
