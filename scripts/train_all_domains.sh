#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
PIP_BIN="$ROOT_DIR/.venv/bin/pip"
CONFIG_PATH="configs/debug.yaml"
LOG_DIR="$ROOT_DIR/artifacts/logs"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/train_all_domains_${TIMESTAMP}.log"

if [[ "${1:-}" == "--full" ]]; then
  CONFIG_PATH="configs/base.yaml"
elif [[ "${1:-}" == "--debug" || -z "${1:-}" ]]; then
  CONFIG_PATH="configs/debug.yaml"
else
  echo "Usage: bash scripts/train_all_domains.sh [--debug|--full]"
  exit 1
fi

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting full domain training run"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Root directory: $ROOT_DIR"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Python: $PYTHON_BIN"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Config: $CONFIG_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Log file: $LOG_FILE"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] Creating virtual environment in $ROOT_DIR/.venv"
  python3 -m venv .venv
fi

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Environment setup"
"$PYTHON_BIN" -m pip install --upgrade pip >> "$LOG_FILE" 2>&1
"$PIP_BIN" install -r requirements.txt >> "$LOG_FILE" 2>&1

DOMAINS=(
  "application"
  "bureau"
  "previous_application"
  "installments"
  "pos_cash"
  "credit_card"
)

for domain in "${DOMAINS[@]}"; do
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] Training domain: $domain"
  PYTHONPATH=src "$PYTHON_BIN" scripts/train_domain.py --config "$CONFIG_PATH" --domain "$domain"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] Finished domain: $domain"
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Full domain training run completed successfully"
