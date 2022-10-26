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
"""Proxy regression module tests (pymc3 interface)
"""
import pytest

try:
	import pymc3 as pm
except ImportError:
	pytest.skip("PyMC3/Theano packages not installed", allow_module_level=True)

import regressproxy.models_pymc3


def test_module_structure():
	assert regressproxy.models_pymc3


def test_module_object_structure():
	assert regressproxy.models_pymc3.HarmonicModelAmpPhase
	assert regressproxy.models_pymc3.HarmonicModelCosineSine
	assert regressproxy.models_pymc3.ModelSet
	assert regressproxy.models_pymc3.ProxyModel


def test_module_method_structure():
	assert regressproxy.models_pymc3.HarmonicModelAmpPhase.get_value
	assert regressproxy.models_pymc3.HarmonicModelAmpPhase.get_amplitude
	assert regressproxy.models_pymc3.HarmonicModelAmpPhase.get_phase
	assert regressproxy.models_pymc3.HarmonicModelAmpPhase.compute_gradient
	assert regressproxy.models_pymc3.HarmonicModelCosineSine.get_value
	assert regressproxy.models_pymc3.HarmonicModelCosineSine.get_amplitude
	assert regressproxy.models_pymc3.HarmonicModelCosineSine.get_phase
	assert regressproxy.models_pymc3.HarmonicModelCosineSine.compute_gradient
	assert regressproxy.models_pymc3.ModelSet.get_value
	assert regressproxy.models_pymc3.ProxyModel.get_value
	assert regressproxy.models_pymc3.proxy_model_set
	assert regressproxy.models_pymc3.setup_proxy_model_pymc3
