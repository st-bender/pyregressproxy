# Proxy classes for regression analysis

[![builds](https://github.com/st-bender/pyregressproxy/actions/workflows/ci_build_and_test.yml/badge.svg?branch=main)](https://github.com/st-bender/pyregressproxy/actions/workflows/ci_build_and_test.yml)
[![docs](https://rtfd.org/projects/pyregressproxy/badge/?version=latest)](https://pyregressproxy.rtfd.io/en/latest/?badge=latest)
[![package](https://img.shields.io/pypi/v/regressproxy.svg?style=flat)](https://pypi.org/project/regressproxy)
[![wheel](https://img.shields.io/pypi/wheel/regressproxy.svg?style=flat)](https://pypi.org/project/regressproxy)
[![pyversions](https://img.shields.io/pypi/pyversions/regressproxy.svg?style=flat)](https://pypi.org/project/regressproxy)
[![codecov](https://codecov.io/gh/st-bender/pyregressproxy/branch/main/graphs/badge.svg)](https://codecov.io/gh/st-bender/pyregressproxy)
[![coveralls](https://coveralls.io/repos/github/st-bender/pyregressproxy/badge.svg)](https://coveralls.io/github/st-bender/pyregressproxy)
[![scrutinizer](https://scrutinizer-ci.com/g/st-bender/pyregressproxy/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/st-bender/pyregressproxy/?branch=main)

## Overview

This package provides classes to easily model proxies for regression analysis.
Originally intended to be used for (trace gas density) time series,
but it might be useful in other cases too.
The classes allow for a lag and finite proxy lifetime to be include when
modelling the time series.

Development happens on github <https://github.com/st-bender/pyregressproxy>.

## Install

### Prerequisites

All dependencies will be automatically installed when using
`pip install` or `python setup.py`, see below.
However, to speed up the install or for use
within a `conda` environment, it may be advantageous to
install some of the important packages beforehand:

- `numpy` at least version 1.13.0 for general numerics,
- `celerite` at least version 0.3.0 for the "standard" model interface,
- `PyMC3`, optional, for the PyMC3 model interface, and
- `PyMC4`, optional, for the PyMC4 model interface.

`numpy` should probably be installed first because `celerite` needs it for setup.
It may also be a good idea to install
[`pybind11`](https://pybind11.readthedocs.io) beforehand
because `celerite` uses its interface.

Depending on the setup, `numpy` and `pybind11` can be installed via `pip`:
```sh
pip install numpy pybind11
```
or [`conda`](https://conda.io):
```sh
conda install numpy pybind11
```

### regressproxy

Official releases are available as `pip` packages from the main package repository,
to be found at <https://pypi.org/project/regressproxy/>, and which can be installed with:
```sh
$ pip install regressproxy
```
The latest development version of regressproxy can be installed with
[`pip`](https://pip.pypa.io) directly from github
(see <https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support>
and <https://pip.pypa.io/en/stable/reference/pip_install/#git>):
```sh
$ pip install [-e] git+https://github.com/st-bender/pyregressproxy.git
```

The other option is to use a local clone:
```sh
$ git clone https://github.com/st-bender/pyregressproxy.git
$ cd pyregressproxy
```
and then using `pip` (optionally using `-e`, see
<https://pip.pypa.io/en/stable/reference/pip_install/#install-editable>):
```sh
$ pip install [-e] .
```

or using `setup.py`:
```sh
$ python setup.py install
```

## Usage

The module is imported as usual:
```python
>>> import regressproxy
```

Basic class and method documentation is accessible via `pydoc`:
```sh
$ pydoc regressproxy
```

## License

This python package is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
