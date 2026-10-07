#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mode="${1:-full}"

case "${mode}" in
  quick)
    ./run_one.sh quick 20261008 5 4 yes
    ;;
  full)
    ./run_one.sh run_1 20261008 500 4 yes
    ./run_one.sh run_2 20261009 500 4 no
    ./run_one.sh run_3 20261010 500 4 no
    "${PYTHON:-python3}" audit.py \
      --run-dirs results/run_1 results/run_2 results/run_3 \
      --output-dir results/audit
    ;;
  *)
    echo "Usage: ./run_experiment.sh [quick|full]"
    exit 2
    ;;
esac
