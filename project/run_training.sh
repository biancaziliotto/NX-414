# Configuration
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_USER="${REMOTE_USER:-}"
VERBOSE=true

# Define parameter mappings (neural_dataset -> dataset, subjects, rois)
# NSD
NSD_DATASET="nsd_stimuli"
NSD_SUBJECTS=("subj01")
NSD_ROIS=("V1v" "V2v" "V3v" "hV4" "FFA-1" "VWFA-1" "PPA" "OPA" "EBA")

# TVSD
TVSD_DATASET="things_stimuli"
TVSD_SUBJECTS=("monkeyF")
TVSD_ROIS=("IT" "V1" "V4")

# EEG2
EEG2_DATASET="things_stimuli"
EEG2_SUBJECTS=("sub-01")
EEG2_ROIS=("occipital_parietal")


# Parse command line arguments
MODELS=("adv_resnet152_imagenet_full_ffgsm_eps-1_alpha-125-ep10_seed-0" "Qwen3-VL-2B-Instruct")
NEURAL_DATASETS=("TVSD" "EEG2" "NSD")

cd project

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --neural-datasets)
            shift
            NEURAL_DATASETS=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                NEURAL_DATASETS+=("$1")
                shift
            done
            ;;
        --remote-host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote-user)
            REMOTE_USER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display configuration
echo "========================================================================"
echo "TRAINING SCRIPT RUNNER"
echo "========================================================================"
echo "Models: ${MODELS[*]}"
echo "Neural Datasets: ${NEURAL_DATASETS[*]}"
echo "Remote Host: $REMOTE_HOST"
echo "Remote User: ${REMOTE_USER:-current user}"
if [ "$DRY_RUN" = true ]; then
    echo "MODE: DRY RUN (no training)"
fi
echo "========================================================================"
echo ""

# Build SSH command prefix
if [ "$REMOTE_HOST" = "localhost" ] || [ "$REMOTE_HOST" = "127.0.0.1" ]; then
    SSH_PREFIX=""
else
    if [ -z "$REMOTE_USER" ]; then
        SSH_PREFIX="ssh $REMOTE_HOST"
    else
        SSH_PREFIX="ssh ${REMOTE_USER}@${REMOTE_HOST}"
    fi
fi

# Count total combinations
TOTAL_COMBOS=0
for model in "${MODELS[@]}"; do
    for neural_dataset in "${NEURAL_DATASETS[@]}"; do
        case $neural_dataset in
            TVSD)
                SUBJECTS=("${TVSD_SUBJECTS[@]}")
                ROIS=("${TVSD_ROIS[@]}")
                ;;
            EEG2)
                SUBJECTS=("${EEG2_SUBJECTS[@]}")
                ROIS=("${EEG2_ROIS[@]}")
                ;;
            NSD)
                SUBJECTS=("${NSD_SUBJECTS[@]}")
                ROIS=("${NSD_ROIS[@]}")
                ;;
        esac
        TOTAL_COMBOS=$((TOTAL_COMBOS + ${#SUBJECTS[@]} * ${#ROIS[@]}))
    done
done

echo "Total combinations: $TOTAL_COMBOS"
echo ""

# Results tracking
COMPLETED=0
FAILED=0
FAILED_RUNS=()
COMBO_NUM=0

# Run training for each combination
for model in "${MODELS[@]}"; do
    for neural_dataset in "${NEURAL_DATASETS[@]}"; do
        # Get forced dataset and parameters for this neural_dataset
        case $neural_dataset in
            TVSD)
                DATASET="$TVSD_DATASET"
                SUBJECTS=("${TVSD_SUBJECTS[@]}")
                ROIS=("${TVSD_ROIS[@]}")
                ;;
            EEG2)
                DATASET="$EEG2_DATASET"
                SUBJECTS=("${EEG2_SUBJECTS[@]}")
                ROIS=("${EEG2_ROIS[@]}")
                ;;
            NSD)
                DATASET="$NSD_DATASET"
                SUBJECTS=("${NSD_SUBJECTS[@]}")
                ROIS=("${NSD_ROIS[@]}")
                ;;
        esac
        
        # Inner loops for subjects and rois
        for subject in "${SUBJECTS[@]}"; do
            for roi in "${ROIS[@]}"; do
                COMBO_NUM=$((COMBO_NUM + 1))
                echo "[$COMBO_NUM/$TOTAL_COMBOS] model=$model neural_dataset=$neural_dataset dataset=$DATASET subject=$subject roi=$roi"
                
                # Build training command
                TRAIN_CMD="python train_encoding_models.py \
                    --model '$model' \
                    --neural_dataset '$neural_dataset' \
                    --dataset '$DATASET' \
                    --subject '$subject' \
                    --roi '$roi' \
                    --use_cv False \
                    --verbose"
                
                # Show command in dry-run mode
                if [ "$DRY_RUN" = true ]; then
                    echo "  Command: $TRAIN_CMD"
                    continue
                fi
                
                # Execute training
                if [ -z "$SSH_PREFIX" ]; then
                    # Local execution
                    eval "$TRAIN_CMD"
                else
                    # Remote execution via SSH
                    $SSH_PREFIX "$TRAIN_CMD"
                fi
                
                if [ $? -eq 0 ]; then
                    COMPLETED=$((COMPLETED + 1))
                    echo "  ✓ Completed"
                else
                    FAILED=$((FAILED + 1))
                    FAILED_RUNS+=("$model:$neural_dataset:$subject:$roi")
                    echo "  ✗ Failed"
                fi
                echo ""
            done
        done
    done
done

# Summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - $COMBO_NUM combinations shown (no training executed)"
else
    echo "Total runs: $COMBO_NUM"
    echo "Completed: $COMPLETED"
    echo "Failed: $FAILED"
    
    if [ $FAILED -gt 0 ]; then
        echo ""
        echo "Failed combinations:"
        for run in "${FAILED_RUNS[@]}"; do
            echo "  - $run"
        done
    fi
fi

echo "========================================================================"

if [ "$DRY_RUN" = true ]; then
    exit 0
elif [ $FAILED -eq 0 ]; then
    echo "✓ All training completed successfully!"
    exit 0
else
    echo "✗ Some runs failed. Check logs above."
    exit 1
fi
