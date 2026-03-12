# Profiles defined as named functions for better pickling with ProcessPoolExecutor
def _temperature_profile_electron(r):
    return 3.0e3 * (1 - r**2)

def _temperature_profile_hydrogen(r):
    return 1.6e3 * (1 - r**2)

def _density_profile(r):
    return 2e20 * (1 - r**4)

