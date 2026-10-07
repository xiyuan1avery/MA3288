#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: ./run_one.sh RUN_NAME SEED [REPLICATIONS] [THREADS] [FIGURE]"
  exit 2
fi

run_name="$1"
seed="$2"
replications="${3:-500}"
threads="${4:-4}"
make_figure="${5:-no}"

cxx_bin="${CXX:-c++}"
python_bin="${PYTHON:-python3}"
output_dir="results/${run_name}"

mkdir -p build "${output_dir}/data"
mkdir -p build/matplotlib-cache build/xdg-cache

"${cxx_bin}" \
  -std=c++17 \
  -O3 \
  -pthread \
  -Wall \
  -Wextra \
  -pedantic \
  simulate.cpp \
  -o build/simulate

./build/simulate \
  --replications "${replications}" \
  --threads "${threads}" \
  --seed "${seed}" \
  --output "${output_dir}/data/replicate_regret.csv" \
  --diagnostics "${output_dir}/data/certificate_diagnostics.csv"

export MPLCONFIGDIR="${PWD}/build/matplotlib-cache"
export XDG_CACHE_HOME="${PWD}/build/xdg-cache"

analysis_args=(
  --input "${output_dir}/data/replicate_regret.csv"
  --diagnostics "${output_dir}/data/certificate_diagnostics.csv"
  --output-dir "${output_dir}"
  --seed "${seed}"
)

if [[ "${make_figure}" == "yes" ]]; then
  analysis_args+=(--make-figure)
fi

"${python_bin}" analyze.py "${analysis_args[@]}"
echo "Saved ${run_name} under ${output_dir}"
