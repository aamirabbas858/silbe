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

# Hold a sleep assertion for the WHOLE script, including the waiting period.
#
# Without this there is a gap: the small run's own caffeinate dies when it
# finishes, and this script polls every 60 seconds before starting the next
# one. This machine is set to sleep after 1 minute idle, so that gap is
# enough for the Mac to sleep and the overnight run to never begin.
#
# -w waits on this script's PID, so the assertion lives exactly as long as
# the script does and is released automatically when it exits.
caffeinate -i -w $$ &

echo "[$(date '+%H:%M:%S')] waiting for the small run to finish (sleep held)"

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
