#!/bin/bash
# Flask server for the visualizer (no DATA_DIR needed)

: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"

ROLLOUTS_DIR=${1:-"${VIEWSUITE_ROOT}/data/rollouts"}
PORT=${2:-8766}
ACTION_LEN_INTERVALS=${3:-"2,5,8,11"}

echo "Starting rollouts visualizer server..."
echo "Rollouts directory: $ROLLOUTS_DIR"
echo "Port: $PORT"
echo "Action length intervals: $ACTION_LEN_INTERVALS"
echo ""
echo "Open your browser and navigate to: http://localhost:$PORT"
echo "Press Ctrl+C to stop the server"
echo ""

python3 "${VIEWSUITE_ROOT}/visualizer/server.py" \
  --rollouts_dir "$ROLLOUTS_DIR" \
  --port $PORT \
  --action_len_intervals "$ACTION_LEN_INTERVALS"
