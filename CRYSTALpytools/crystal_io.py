#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deprecated. Use corresponding classes defined in :ref:`io.crystal <ref-io-crystal>`
"""
from warnings import warn, filterwarnings
from CRYSTALpytools.io.crystal import Crystal_input, Crystal_output, Properties_input, Properties_output, Crystal_density, cry_combine_density, write_cry_density
from CRYSTALpytools.geometry import Crystal_gui


filterwarnings("default", category=DeprecationWarning)
if "crystal_io" in __name__:
    warn("The 'crystal_io' module is deprecated. Please import from 'io.crystal' instead.",
         DeprecationWarning, stacklevel=2)
