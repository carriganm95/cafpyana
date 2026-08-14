#!/bin/bash
# Wrapper to launch the cafpyana Jupyter kernel with the full spack environment.
# Sources setup.sh (htgettoken at the end may fail silently — that's OK).
CAFPYANA_DIR=/exp/sbnd/app/users/micarrig/nueCCNp/cafpyana
cd "$CAFPYANA_DIR"
htgettoken() { :; }
export -f htgettoken
source setup.sh </dev/null
VENV_SITE="${CAFPYANA_DIR}/envs/venv_py310_cafpyana/lib/python3.10/site-packages"
# Strip spack's numpy from PYTHONPATH so the venv numpy takes priority
PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'py-numpy' | tr '\n' ':')
export PYTHONPATH="${VENV_SITE}:${PYTHONPATH}"
exec python "$@"
