#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Note: run_ai_smoke.sh is kept for compatibility." >&2
echo "      Prefer: ${SCRIPT_DIR}/run_ai.sh smoke" >&2

exec "${SCRIPT_DIR}/run_ai.sh" smoke "$@"
