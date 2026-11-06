#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA="${DATA:-.}"
python glss.py --data "$DATA" ${INIT:+--init "$INIT"}
