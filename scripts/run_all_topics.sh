#!/usr/bin/env bash
set -euo pipefail
TWPGEN_ROOT="${TWPGEN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TWPGEN_ROOT
bash "$TWPGEN_ROOT/scripts/run_twpgen_pipeline.sh" "$@"
