"""Regression test for a subtle import-order behavior in
analysis_village/nueNp0Pi/utils.py: it wildcard-imports config/settings.py
(whose own DETECTOR default is "SBND_Gen1", the truth/reco-cut convention),
then explicitly re-imports DETECTOR from config/datasets.py ("SBND_nohighyz",
the convention utils.py's systematics code actually needs), on purpose,
specifically to override the wildcard import. See utils.py's import-block
comment and nueNp0Pi/README.md's "Detector / fiducial-volume convention"
section for why these two files intentionally disagree.
"""


def test_detector_constant_not_silently_shadowed_incorrectly():
    import analysis_village.nueNp0Pi.utils as u
    from analysis_village.nueNp0Pi.config.datasets import DETECTOR as DATASETS_DETECTOR
    from analysis_village.nueNp0Pi.config.settings import DETECTOR as SETTINGS_DETECTOR

    assert SETTINGS_DETECTOR == "SBND_Gen1"
    assert DATASETS_DETECTOR == "SBND_nohighyz"
    # utils.py's own module-level DETECTOR must end up bound to config/datasets.py's
    # value (the later, explicit import), not silently left at config/settings.py's
    # wildcard-imported value.
    assert u.DETECTOR == DATASETS_DETECTOR
