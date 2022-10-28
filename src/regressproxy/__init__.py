# Copyright (c) 2021-2022 Stefan Bender
#
# This module is part of pyregressproxy.
# pyregressproxy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Proxy model classes for regression analysis

This module contains regression proxy models with lag and finite lifetime.
"""
__version__ = "0.0.2"

# The standard version is the celerite model only
from .models_cel import *
