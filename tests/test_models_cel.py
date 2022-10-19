# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2022 Stefan Bender
#
# This module is part of pyregressproxy.
# pyregressproxy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Proxy regression celerite module tests
"""
import numpy as np

import pytest

from regressproxy.models_cel import (
	HarmonicModelCosineSine,
	HarmonicModelAmpPhase,
	ProxyModel,
	proxy_model_set,
)


@pytest.fixture(scope="module")
def xx():
	# modified Julian days, 2 years from 2000-01-01
	_xs = 51544.5 + np.arange(0., 2 * 365. + 1, 1.)
	return np.ascontiguousarray(_xs, dtype=np.float64)


def ys(xs, c, s):
	_ys = c * np.cos(2 * np.pi * xs) + s * np.sin(2 * np.pi * xs)
	return np.ascontiguousarray(_ys, dtype=np.float64)


@pytest.mark.parametrize(
	"c, s",
	[
		(1.0, 0.0),
		(0.0, 1.0),
		(0.5, 2.0),
		(2.0, 0.5),
		(1.0, 1.0),
	]
)
def test_harmonics_models(xx, c, s):
	# convert to fractional years
	xs = 1859 + (xx - 44.25) / 365.25
	wave0 = ys(xs, c, s)

	harm1 = HarmonicModelCosineSine(1., c, s)
	wave1 = harm1.get_value(xs)
	np.testing.assert_allclose(wave1, wave0)

	amp = np.sqrt(c**2 + s**2)
	phase = np.arctan2(c, s)
	np.testing.assert_allclose(
		amp, harm1.get_amplitude(),
	)
	np.testing.assert_allclose(
		phase, harm1.get_phase(),
	)

	harm2 = HarmonicModelAmpPhase(1., amp, phase)
	wave2 = harm2.get_value(xs)
	np.testing.assert_allclose(wave2, wave0)


def _test_data(xs, values, f, c, s):
	amp = 3.
	lag = 2.
	tau0 = 1.
	proxy0 = ProxyModel(
		xs, values,
		amp=amp,
		lag=lag,
		tau0=tau0,
		taucos1=c,
		tausin1=s,
		taucos2=0,
		tausin2=0,
		ltscan=10,
		days_per_time_unit=f * 365.25,
	)
	return proxy0.get_value(xs)


def _yy(x):
	_ys = np.zeros_like(x)
	_ys[10::20] = 10.
	return np.ascontiguousarray(_ys, dtype=np.float64)


@pytest.mark.long
@pytest.mark.parametrize(
	"f",
	[1., 1. / 365.25]
)
def test_proxy_cel(xx, f, c=3.0, s=1.0):
	dx = 1. / (f * 365.25)
	if f < 1.:
		xs = xx * dx
	else:
		# convert to fractional years
		xs = 1859 + (xx - 44.25) * dx
	# proxy "values"
	values = _yy(xs)

	yp = _test_data(xs, values, f, c, s)
	#print(values)
	#print(yp)

	proxy = ProxyModel(
		xs, values,
		amp=3,
		lag=2,
		tau0=1,
		taucos1=c,
		tausin1=s,
		taucos2=0.,
		tausin2=0.,
		ltscan=10,
		days_per_time_unit=f * 365.25,
	)
	prox1 = proxy.get_value(xs)
	np.testing.assert_allclose(prox1, yp)

	proxy2 = proxy_model_set(
		constant=False,
		proxy_config={
			"proxy": {
				"times": xs,
				"values": values,
				"lifetime_scan": 10,
				"days_per_time_unit": f * 365.25,
				"amp": 3,
				"lag": 2,
				"tau0": 1,
				"taucos1": c,
				"tausin1": s,
			}
		}
	)
	proxy2.set_parameter("proxy:amp", 3)
	proxy2.set_parameter("proxy:lag", 2)
	proxy2.set_parameter("proxy:tau0", 1)
	proxy2.set_parameter("proxy:taucos1", c)
	proxy2.set_parameter("proxy:tausin1", s)
	prox2 = proxy2.get_value(xs)
	np.testing.assert_allclose(prox2, yp)
