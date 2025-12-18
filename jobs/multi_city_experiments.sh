#!/bin/bash
#SBATCH --job-name=multi_city
#SBATCH --partition=gpu
#SBATCH --array=0-15              # 4 cities × 4 configs = 16 experiments
#SBATCH --gres=gpu:1              # 1 GPU per experiment
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00           # 2 hours per experiment
#SBATCH --output=logs/city_%A_%a.out
#SBATCH --error=logs/city_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@umd.edu

# ═══════════════════════════════════════════════════════════════
# SLURM JOB ARRAY FOR MULTI-CITY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
#
# WHAT THIS DOES:
# - Launches 16 independent experiments (4 cities × 4 configs)
# - Each experiment runs on its own GPU in parallel
# - Results saved to separate directories
#
# JOB ARRAY INDEXING:
# Task ID 0-3:   NYC (balanced, emissions_focus, growth_focus, aggressive)
# Task ID 4-7:   LA (same configs)
# Task ID 8-11:  College Park (same configs)
# Task ID 12-15: Baltimore (same configs)
#
# QUESTIONS TO CONSIDER:
# - Should we request more memory for larger cities? (16GB is enough)
# - Should we use different time limits per city? (No - same complexity)
# - Should we checkpoint long runs? (Good for >2 hour jobs)
# - How do we handle failed jobs? (Check logs, resubmit specific tasks)
#
# USAGE:
#   sbatch jobs/multi_city_experiments.sh
#
# MONITOR:
#   squeue -u $USER
#   tail -f logs/city_*.out
#
# ═══════════════════════════════════════════════════════════════

echo "======================================"
echo "Multi-City Climate Policy Optimization"
echo "======================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================"

# ──────────────────────────────────────────────────────────────
# LOAD MODULES (Nexus-specific)
# ──────────────────────────────────────────────────────────────
module load cuda/11.8
module load python/3.10

# Check modules loaded
echo "Loaded modules:"
module list

# ──────────────────────────────────────────────────────────────
# NAVIGATE TO PROJECT & ACTIVATE ENVIRONMENT
# ──────────────────────────────────────────────────────────────
cd ~/projects/urban-climate-policy-optimization || exit 1
source venv/bin/activate

# Verify Python environment
echo ""
echo "Python environment:"
which python
python --version
python -c "import jax; print(f'JAX version: {jax.__version__}')"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# IMPORTANT: Install GPU JAX in the job
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU
nvidia-smi

# ──────────────────────────────────────────────────────────────
# DEFINE EXPERIMENT GRID
# ──────────────────────────────────────────────────────────────
# Cities to optimize
CITIES=("nyc" "la")

# Policy configurations (different loss weight combinations)
CONFIG_NAMES=("balanced" "emissions_focus" "growth_focus" "aggressive")

# Loss weights for each configuration
# Format: "w_E,w_G,lambda"
CONFIG_WEIGHTS=(
    "1.0,1.0,10.0"      # Balanced: equal weight to emissions & GDP
    "5.0,1.0,20.0"      # Emissions focus: prioritize emission reduction
    "1.0,5.0,5.0"       # Growth focus: prioritize GDP growth
    "2.0,2.0,15.0"      # Aggressive: high weights on both
)

# ──────────────────────────────────────────────────────────────
# CALCULATE CITY AND CONFIG FROM TASK ID
# ──────────────────────────────────────────────────────────────
# Task ID mapping:
# 0-3:   NYC (config 0-3)
# 4-7:   LA (config 0-3)


CITY_IDX=$(($SLURM_ARRAY_TASK_ID / 4))
CONFIG_IDX=$(($SLURM_ARRAY_TASK_ID % 4))

CITY=${CITIES[$CITY_IDX]}
CONFIG_NAME=${CONFIG_NAMES[$CONFIG_IDX]}
WEIGHTS=${CONFIG_WEIGHTS[$CONFIG_IDX]}

# Parse weights
IFS=',' read -r W_E W_G LAMBDA <<< "$WEIGHTS"

echo ""
echo "======================================"
echo "EXPERIMENT CONFIGURATION"
echo "======================================"
echo "City: $CITY"
echo "Configuration: $CONFIG_NAME"
echo "Loss weights:"
echo "  w_E (emissions):  $W_E"
echo "  w_G (GDP):        $W_G"
echo "  λ (terminal CO2): $LAMBDA"
echo "======================================"

# ──────────────────────────────────────────────────────────────
# CREATE OUTPUT DIRECTORY
# ──────────────────────────────────────────────────────────────
# Organize by: results/multi_city/{job_id}/{city}/{config}
OUTPUT_DIR="results/multi_city/${SLURM_ARRAY_JOB_ID}/${CITY}/${CONFIG_NAME}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Output directory: $OUTPUT_DIR"

# ──────────────────────────────────────────────────────────────
# LOG SYSTEM INFO
# ──────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "SYSTEM INFORMATION"
echo "======================================"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo ""
echo "CPU info:"
lscpu | grep "Model name"
echo ""
echo "Memory:"
free -h

# ──────────────────────────────────────────────────────────────
# RUN TRAINING
# ──────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "STARTING TRAINING"
echo "======================================"
echo "Time: $(date)"
echo ""

# Additional hyperparameters (can customize per city if needed)
LEARNING_RATE=0.01
NUM_ITERS=1000
TIME_HORIZON=50
SEED=$((42 + SLURM_ARRAY_TASK_ID))  # Different seed per task

# Run training
python experiments/train_city.py \
    --city "$CITY" \
    --w_E "$W_E" \
    --w_G "$W_G" \
    --lambda_term "$LAMBDA" \
    --lr "$LEARNING_RATE" \
    --num_iters "$NUM_ITERS" \
    --T "$TIME_HORIZON" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --name "${CITY}_${CONFIG_NAME}" \
    --use_traffic true \
    --verbose

# Check exit status
EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TRAINING COMPLETED SUCCESSFULLY"
    echo "======================================"
    
    # Log GPU usage at end
    echo ""
    echo "Final GPU state:"
    nvidia-smi >> "$OUTPUT_DIR/gpu_info.txt"
    
    # Create completion marker
    touch "$OUTPUT_DIR/COMPLETED"
    echo "Completion marker created"
    
else
    echo "✗ TRAINING FAILED (exit code: $EXIT_CODE)"
    echo "======================================"
    
    # Create failure marker
    touch "$OUTPUT_DIR/FAILED"
    echo "Failure marker created"
fi

echo ""
echo "Task $SLURM_ARRAY_TASK_ID complete at $(date)"
echo "======================================"

exit $EXIT_CODE