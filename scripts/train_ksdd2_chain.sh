#!/usr/bin/env bash
# Sequential CPU training chain for the KolektorSDD2 commutator-surface models + re-runs that depend on them.
set -u
cd "$(dirname "$0")/.."
L=artifacts/logs
mkdir -p $L
echo "== ksdd2 anomaly $(date)";       python -c "
import sys; sys.path.insert(0,'.')
from ml.anomaly_detection.train import train_category
train_category('commutator', manifest='ksdd2', max_train=400)" > $L/ksdd2_anomaly.log 2>&1
echo "== ksdd2 classifiers $(date)";   python scripts/train_classifiers.py --dataset ksdd2 --archs mobilenet_v2 efficientnet_b0 --epochs 10 --batch-size 32 --max-minutes 35 > $L/ksdd2_classifiers.log 2>&1
echo "== ksdd2 segmentation $(date)";  python scripts/train_segmentation.py --epochs 14 > $L/ksdd2_segmentation.log 2>&1
echo "== screw/grid re-run $(date)";   python scripts/train_anomaly.py screw grid > $L/train_anomaly_screw_grid_v2.log 2>&1
echo "== recognizer $(date)";          python scripts/train_recognizer.py > $L/recognizer.log 2>&1
echo "== done $(date)"
