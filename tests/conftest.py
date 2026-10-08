"""Configurație comună de test: profiluri hypothesis `fast`, `ci` (implicit) și `deep`.

- `fast` (25 de exemple): rulare rapidă în timpul dezvoltării, feedback în câteva secunde.
- `ci` (50 de exemple): profilul implicit, echilibru între acoperire și viteză.
- `deep` (5000 de exemple): rulat înainte de orice promovare / la checkpoint-uri.

Selectează profilul prin variabila de mediu `HYPOTHESIS_PROFILE` sau prin
`pytest --hypothesis-profile <nume>`.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "fast", max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "ci", max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "deep", max_examples=5000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
