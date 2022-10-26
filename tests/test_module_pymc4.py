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
"""Proxy regression module tests (pymc4 interface)
"""
import pytest

try:
	import pymc as pm
except ImportError:
	pytest.skip("PyMC4 packages not installed", allow_module_level=True)

import regressproxy.models_pymc4


def test_module_structure():
	assert regressproxy.models_pymc4


def test_module_object_structure():
	assert regressproxy.models_pymc4.HarmonicModelAmpPhase
	assert regressproxy.models_pymc4.HarmonicModelCosineSine
	assert regressproxy.models_pymc4.ModelSet
	assert regressproxy.models_pymc4.ProxyModel


def test_module_method_structure():
	assert regressproxy.models_pymc4.HarmonicModelAmpPhase.get_value
	assert regressproxy.models_pymc4.HarmonicModelAmpPhase.get_amplitude
	assert regressproxy.models_pymc4.HarmonicModelAmpPhase.get_phase
	assert regressproxy.models_pymc4.HarmonicModelAmpPhase.compute_gradient
	assert regressproxy.models_pymc4.HarmonicModelCosineSine.get_value
	assert regressproxy.models_pymc4.HarmonicModelCosineSine.get_amplitude
	assert regressproxy.models_pymc4.HarmonicModelCosineSine.get_phase
	assert regressproxy.models_pymc4.HarmonicModelCosineSine.compute_gradient
	assert regressproxy.models_pymc4.ModelSet.get_value
	assert regressproxy.models_pymc4.ProxyModel.get_value
	assert regressproxy.models_pymc4.proxy_model_set
	assert regressproxy.models_pymc4.setup_proxy_model_pymc4
