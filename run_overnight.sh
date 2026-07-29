#!/bin/bash
# Waits for the running `small` training to finish, then starts `base`.
#
# Exists so the two runs do not overlap. Both would fit in 16 GB, but they
# would halve each other's GPU time and the estimates would stop meaning
# anything.
#
# Launched with nohup so it survives the terminal closing. It does not survive
# the lid closing — nothing does, without an external display.
#
#   nohup ./run_overnight.sh > /tmp/silbe-overnight.log 2>&1 &

set -u
cd "$(dirname "$0")"

echo "[$(date '+%H:%M:%S')] waiting for the small run to finish"

# Poll rather than `wait` — the small run was started by a different shell,
# so this process is not its parent and cannot wait on it directly.
while pgrep -f "train.py --config configs/small.json" > /dev/null; do
  sleep 60
done

echo "[$(date '+%H:%M:%S')] small finished — starting base"

# caffeinate -i stops the Mac idling to sleep while this runs.
# -u on python stops it buffering, so the log fills as it goes rather than
# sitting empty for an hour.
caffeinate -i .venv/bin/python -u train.py --config configs/base.json

echo "[$(date '+%H:%M:%S')] base finished"
