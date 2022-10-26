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
"""Proxy regression module tests
"""

import regressproxy


def test_module_structure():
	assert regressproxy.models_cel


def test_modelmodule_object_structure():
	assert regressproxy.ProxyModelSet
	assert regressproxy.ConstantModel
	assert regressproxy.HarmonicModelAmpPhase
	assert regressproxy.HarmonicModelCosineSine
	assert regressproxy.ProxyModel
	assert regressproxy.models_cel.ProxyModelSet
	assert regressproxy.models_cel.ConstantModel
	assert regressproxy.models_cel.HarmonicModelAmpPhase
	assert regressproxy.models_cel.HarmonicModelCosineSine
	assert regressproxy.models_cel.ProxyModel


def test_modelmodule_method_structure():
	assert regressproxy.ProxyModelSet.get_value
	assert regressproxy.ProxyModelSet.compute_gradient
	assert regressproxy.HarmonicModelAmpPhase.get_value
	assert regressproxy.HarmonicModelAmpPhase.get_amplitude
	assert regressproxy.HarmonicModelAmpPhase.get_phase
	assert regressproxy.HarmonicModelAmpPhase.compute_gradient
	assert regressproxy.HarmonicModelCosineSine.get_value
	assert regressproxy.HarmonicModelCosineSine.get_amplitude
	assert regressproxy.HarmonicModelCosineSine.get_phase
	assert regressproxy.HarmonicModelCosineSine.compute_gradient
	assert regressproxy.ProxyModel.get_value
	assert regressproxy.ProxyModel.compute_gradient
	assert regressproxy.setup_proxy_model_with_bounds
	assert regressproxy.proxy_model_set
	assert regressproxy.models_cel.ProxyModelSet.get_value
	assert regressproxy.models_cel.ProxyModelSet.compute_gradient
	assert regressproxy.models_cel.HarmonicModelAmpPhase.get_value
	assert regressproxy.models_cel.HarmonicModelAmpPhase.get_amplitude
	assert regressproxy.models_cel.HarmonicModelAmpPhase.get_phase
	assert regressproxy.models_cel.HarmonicModelAmpPhase.compute_gradient
	assert regressproxy.models_cel.HarmonicModelCosineSine.get_value
	assert regressproxy.models_cel.HarmonicModelCosineSine.get_amplitude
	assert regressproxy.models_cel.HarmonicModelCosineSine.get_phase
	assert regressproxy.models_cel.HarmonicModelCosineSine.compute_gradient
	assert regressproxy.models_cel.ProxyModel.get_value
	assert regressproxy.models_cel.ProxyModel.compute_gradient
	assert regressproxy.models_cel.setup_proxy_model_with_bounds
	assert regressproxy.models_cel.proxy_model_set
