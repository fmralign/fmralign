# fmralign

![build](https://img.shields.io/github/actions/workflow/status/fmralign/fmralign/testing.yml?event=push&style=for-the-badge)
![python version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11_|_3.12_|_3.13-blue?style=for-the-badge)
![license](https://img.shields.io/github/license/fmralign/fmralign?style=for-the-badge)

[Functional alignment for fMRI](https://fmralign.github.io/fmralign-docs) (functional Magnetic Resonance Imaging) data.

This light-weight Python library provides access to a range of functional alignment methods, including Procrustes and Optimal Transport.
It is compatible with and inspired by [Nilearn](http://nilearn.github.io).
Alternative implementations of these ideas can be found in the [pymvpa](http://www.pymvpa.org), [brainiak](http://brainiak.org) or [neuroboros](http://neuroboros.github.io) packages.

## Getting Started

### Installation

You can access the latest stable version of fmralign directly with the PyPi package installer:

```
pip install fmralign
```

For development or bleeding-edge features, fmralign can also be installed directly from source:

```
git clone https://github.com/fmralign/fmralign
cd fmralign
pip install -e .
```

### Documentation

You can found an introduction to functional alignment, a user guide and some examples
on how to use the package at https://fmralign.github.io/fmralign

## License

This project is licensed under the Simplified BSD License.

## Acknowledgments

This project has received funding from the European Union’s Horizon
2020 Research and Innovation Programme under Grant Agreement No. 785907
(HBP SGA2).
This project was supported by [Digiteo](http://www.digiteo.fr).
