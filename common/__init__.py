"""Shared, dependency-free utilities used across the reproduction pipeline.

This package holds cross-cutting infrastructure that every entry point can reuse
without depending on any of the scientific packages. At present it provides the
single authoritative logging policy (:mod:`common.logging_setup`) so that the
wrapper, the forecasting runners, the AE-SINDy tools, the simulation, the
optimiser searches and the plotting scripts all emit terminal output in one
consistent format.

The package uses only the Python standard library and configures nothing on
import.
"""
