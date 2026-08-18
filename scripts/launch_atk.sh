#!/bin/bash
#ATTACK="iou_attachment"
ATTACK="iou_detachment"
#ATTACK="iou_perturbation"
MODEL="focalformer3d"
#SAMPLES=20      # Uncomment to use only specified amount of samples

for rank in {0..1}; do
    # input arguments for the attacks
    ARGS=("--base-rank" "$rank" "--attack" "$ATTACK" "--preset-model" "$MODEL" "--checkpoint" "/beegfs/Kai/results/$MODEL/$ATTACK" "--lc-fusion")

    if [[ -n "$SAMPLES" ]]; then
        ARGS+=(--num-samples "$SAMPLES")
    fi

    sbatch \
        --job-name="${MODEL}_${ATTACK}_fusion" \
        sj_run_pipeline.sh "${ARGS[@]}"

    sleep 5
done