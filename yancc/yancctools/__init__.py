# This is a set of tools for yancc with the following goals
#  1. Provide solutions for the ambipolar Er
#  2. Allow for calculation of radial profiles
#  3. Provide useful plotting.
#
# WARNING:
#    I (Novimir) am using this as playground for exploring generative AI coding tools.
#    Please don't use this AI geneeated code without validation.
#    I am using GeminiCLI v3 around 2026-03

from .yancctools import (
    calculate_net_charge_flux,
    find_ambipolar_roots,
    scan_ambipolar_profile,
)

from .yancctools_io import (
    load_options,
    save_options,
    load_inputs,
    save_inputs,
    load_results,
    save_results,
)

from .yancctools_plot import (
    plot_ambipolar_scan,
    plot_ambipolar_profile,
    plot_ambipolar_summary,
)
