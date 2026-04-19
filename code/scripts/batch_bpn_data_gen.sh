#!/usr/bin/env bash
# Batch BPN dataset generation.
#
# Iterates every entry in bpn_dataset_video_mapping.json, re-runs S2
# frontalization with the alignment refiner enabled against the
# pre-saved s1_tracks.json under /workspace/tpm_dataset/<folder>, and
# writes corrected track metadata + canonical ROI PNGs to
# /workspace/bpn_dataset/<folder>.
#
# Usage:
#   bash code/scripts/batch_bpn_data_gen.sh
#
# Optional environment overrides:
#   REFINER_CKPT      refiner checkpoint path (default: adv.yaml value)
#   REFINER_DEVICE    torch device (default: cuda)
#   TPM_ROOT          source s1_tracks root (default: /workspace/tpm_dataset)
#   BPN_ROOT          output root (default: /workspace/bpn_dataset)
#   ONLY              regex; if set, only folders matching it are processed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAPPING_JSON="${SCRIPT_DIR}/bpn_dataset_video_mapping.json"

REFINER_CKPT="${REFINER_CKPT:-${REPO_ROOT}/checkpoints/refiner/refiner_v0.pt}"
REFINER_DEVICE="${REFINER_DEVICE:-cuda}"
TPM_ROOT="${TPM_ROOT:-/workspace/tpm_dataset}"
BPN_ROOT="${BPN_ROOT:-/workspace/bpn_dataset}"
ONLY="${ONLY:-}"

if [[ ! -f "${MAPPING_JSON}" ]]; then
    echo "Mapping file missing: ${MAPPING_JSON}" >&2
    exit 1
fi
if [[ ! -f "${REFINER_CKPT}" ]]; then
    echo "Refiner checkpoint missing: ${REFINER_CKPT}" >&2
    echo "Override via REFINER_CKPT=..." >&2
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python"
fi

# Pull the (folder, video_path) pairs out of the JSON with python (no jq dep).
mapfile -t ENTRIES < <("${PYTHON}" -c "
import json, sys
for k, v in json.load(open('${MAPPING_JSON}')).items():
    print(f'{k}\t{v}')
")

echo "Processing ${#ENTRIES[@]} videos from ${MAPPING_JSON}"
echo "  refiner checkpoint: ${REFINER_CKPT}"
echo "  device:             ${REFINER_DEVICE}"
echo "  tpm root (input):   ${TPM_ROOT}"
echo "  bpn root (output):  ${BPN_ROOT}"
[[ -n "${ONLY}" ]] && echo "  ONLY filter:        ${ONLY}"
echo

processed=0
skipped=0
failed=0
for entry in "${ENTRIES[@]}"; do
    folder="${entry%%$'\t'*}"
    video="${entry##*$'\t'}"

    if [[ -n "${ONLY}" && ! "${folder}" =~ ${ONLY} ]]; then
        continue
    fi

    s1_json="${TPM_ROOT}/${folder}/s1_tracks.json"
    out_dir="${BPN_ROOT}/${folder}"

    if [[ ! -f "${s1_json}" ]]; then
        echo "[SKIP] ${folder}: missing ${s1_json}"
        skipped=$((skipped + 1))
        continue
    fi
    if [[ ! -f "${video}" ]]; then
        echo "[SKIP] ${folder}: missing video ${video}"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[RUN ] ${folder}  <-  ${video}"
    if "${PYTHON}" "${SCRIPT_DIR}/generate_bpn_dataset.py" \
        --video "${video}" \
        --s1-tracks "${s1_json}" \
        --output-dir "${out_dir}" \
        --refiner-checkpoint "${REFINER_CKPT}" \
        --refiner-device "${REFINER_DEVICE}" \
        --log-level INFO; then
        processed=$((processed + 1))
    else
        echo "[FAIL] ${folder} (exit $?)"
        failed=$((failed + 1))
    fi
done

echo
echo "Done: ${processed} processed, ${skipped} skipped, ${failed} failed"
exit ${failed}
