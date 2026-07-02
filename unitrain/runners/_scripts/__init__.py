"""Standalone scripts executed inside framework virtual environments.

Each script receives configuration via ``--config-json`` CLI argument,
calls the framework API directly, and prints results to stdout.
These files are NOT imported by the main project — they run in isolated venvs.
"""
