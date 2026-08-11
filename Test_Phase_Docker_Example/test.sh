#!/usr/bin/env bash
bash ./build.sh

# The container registers ONE moving(EXP)/fixed(INSP) CT pair per run (three paths: fixed
# INSP, moving EXP, output). This script iterates the VALIDATION set described by a dataset
# JSON and calls the container once per subject.
#   fixed  = INSP (inspiration)
#   moving = EXP  (expiration)
#   output = disp_<subject>.nii.gz
DATA_DIR=/path/to/learn2breath_validation
IMAGES_DIR=${DATA_DIR}/ct_data
DOCKER_DIR=$(cd "$(dirname "$0")" && pwd)
DATASET_JSON=${DOCKER_DIR}/Learn2Breath_val_dataset.json
OUTPUT_DIR=${DOCKER_DIR}/Learn2Breath_convexadam_TestPhase
mkdir -p "${OUTPUT_DIR}"

# Extract one line per validation subject (basenames):
#   fixed_insp  moving_exp  output
while read -r FIX MOV OUT; do
    [ -z "${FIX}" ] && continue
    echo "=== Registering ${OUT} ==="
    docker run --rm \
        --ipc=host \
        --memory 60g \
        --gpus "device=0" \
        --user $(id -u):$(id -g) \
        --network=none \
        --mount type=bind,source=${IMAGES_DIR},target=/app/input,readonly \
        --mount type=bind,source=${OUTPUT_DIR},target=/app/output \
        learn2breath_convexadam \
            /app/input/${FIX} \
            /app/input/${MOV} \
            /app/output/${OUT} < /dev/null
done < <(python3 - "${DATASET_JSON}" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
for e in d["validation"]:
    print(os.path.basename(e["Fixed INSP"]),
          os.path.basename(e["Moving EXP"]),
          "disp_{}.nii.gz".format(e["subject"]))
PY
)

echo "All validation displacement fields written to ${OUTPUT_DIR}"
