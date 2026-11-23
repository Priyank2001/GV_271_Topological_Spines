#!/usr/bin/env bash
# Run the project: (optionally) recompute the pipeline, then serve the web viewer.
#
#   ./run.sh                      serve web/ on http://localhost:8000
#   ./run.sh --rebuild            rebuild web/data/*.json from the pipeline, then serve
#   ./run.sh --all                regenerate the datasets AND rebuild web/data, then serve
#   ./run.sh --rebuild fuel pbmc3k   rebuild only these viewer entries
#
# Options:
#   -a, --all          same as --datasets --rebuild
#   -d, --datasets     regenerate the synthetic datasets in generator/datasets/
#                      (seeded, so the results are identical unless the code changed)
#       --pbmc         also redo pbmc3k (downloads the 10x data if not cached)
#   -r, --rebuild      run web/build_data.py (extremum graphs, layouts, variants)
#       --no-variants  skip the compare-mode variants (AnyEG, 14-neighbour) when rebuilding
#   -c, --comparison   run the AnyEG-vs-ours study (comparison/results/)
#   -t, --tests        run the Python test suite before serving
#   -n, --no-serve     do the requested steps and exit without starting the server
#   -p, --port N       port for the web server (default 8000; 8765 is taken on this machine)
#       --host H       address to bind (default 127.0.0.1; 0.0.0.0 to reach it from other machines)
#   -h, --help         show this help
# Any other arguments are viewer entry names passed to web/build_data.py.
#
# Environment: PYTHON (default /usr/bin/python3, the interpreter with numpy/torch/CUDA).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-/usr/bin/python3}"

PORT=8000
HOST=127.0.0.1
DATASETS=0 PBMC=0 REBUILD=0 COMPARISON=0 TESTS=0 SERVE=1
BUILD_ARGS=()
NAMES=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -a|--all)        DATASETS=1; REBUILD=1 ;;
        -d|--datasets)   DATASETS=1 ;;
        --pbmc)          PBMC=1 ;;
        -r|--rebuild)    REBUILD=1 ;;
        --no-variants)   BUILD_ARGS+=(--no-variants) ;;
        -c|--comparison) COMPARISON=1 ;;
        -t|--tests)      TESTS=1 ;;
        -n|--no-serve)   SERVE=0 ;;
        -p|--port)       PORT="${2:?--port needs a number}"; shift ;;
        --host)          HOST="${2:?--host needs an address}"; shift ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)               NAMES+=("$1") ;;
    esac
    shift
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
SECONDS=0

if [ "$DATASETS" = 1 ]; then
    step "Regenerating synthetic datasets (generator/datasets/)"
    for gen in make_datasets make_cyclic_dataset make_two_peaks_dataset make_volcano_dataset; do
        echo "-- $gen"
        (cd generator && "$PY" "$gen.py")
    done
    if [ "$PBMC" = 1 ] || [ ! -f generator/datasets/pbmc3k/points.npy ]; then
        echo "-- make_pbmc3k_dataset"
        "$PY" generator/make_pbmc3k_dataset.py
    fi
fi

if [ "$TESTS" = 1 ]; then
    step "Running tests"
    "$PY" -m pytest -q tests
fi

if [ "$REBUILD" = 1 ]; then
    step "Building viewer data (web/data/)"
    "$PY" web/build_data.py ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} ${NAMES[@]+"${NAMES[@]}"}
elif [ ${#NAMES[@]} -gt 0 ]; then
    echo "Entry names given without --rebuild/--all: ${NAMES[*]}" >&2; exit 2
fi

if [ "$COMPARISON" = 1 ]; then
    step "Running the AnyEG vs ours comparison"
    "$PY" -m comparison
fi

[ "$SECONDS" -gt 0 ] && echo "Pipeline steps took ${SECONDS}s."

if [ "$SERVE" = 1 ]; then
    if [ ! -f web/data/index.json ]; then
        echo "web/data/index.json is missing; run ./run.sh --rebuild first." >&2; exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        echo "Port $PORT is already in use; pick another with --port." >&2; exit 1
    fi
    step "Serving web/ at http://$HOST:$PORT/   (Ctrl+C to stop)"
    exec "$PY" -m http.server -d web --bind "$HOST" "$PORT"
fi
