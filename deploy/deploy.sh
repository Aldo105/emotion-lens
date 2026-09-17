#!/usr/bin/env bash
# EmotionLens — redeploy script. Run on the VPS as the app user
# (or via CI over SSH) to ship a new version:
#   bash deploy/deploy.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> Pulling latest code..."
git pull origin master

echo ">> Installing/updating dependencies..."
./venv/bin/pip install -r deploy/requirements-cpu.txt

echo ">> Restarting service..."
sudo systemctl restart emotionlens

echo ">> Deployed $(git rev-parse --short HEAD)"
sudo systemctl status emotionlens --no-pager -l | head -n 5
