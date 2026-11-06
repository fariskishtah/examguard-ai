#!/usr/bin/env bash
# Build ExamGuard AI.app with PyInstaller (run from the repo root, inside the project venv).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f models/yolo11n.pt ] || python scripts/fetch_model.py
python -m PyInstaller --noconfirm --clean --windowed --name "ExamGuard AI" \
  --osx-bundle-identifier com.fariskishtah.examguardai \
  --paths src \
  --add-data "src/examguard/database/migrations:examguard/database/migrations" \
  --add-data "models/yolo11n.pt:models" \
  --collect-submodules ultralytics.trackers --collect-data ultralytics \
  --collect-all torchvision \
  --hidden-import examguard.database.migrations.env \
  --exclude-module matplotlib --exclude-module pytest \
  main.py
echo "Built: dist/ExamGuard AI.app"
