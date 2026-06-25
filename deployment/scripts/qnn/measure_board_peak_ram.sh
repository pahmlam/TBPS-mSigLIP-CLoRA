#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-0.05}"

if [ "$#" -lt 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage:
  INTERVAL=0.05 deployment/scripts/qnn/measure_board_peak_ram.sh <command> [args...]

Measures peak host-side RAM while a board command runs. It reports:
  - process peak VmRSS/VmHWM from /proc/<pid>/status
  - system MemAvailable drop from /proc/meminfo

Run it on RB3 around qnn-net-run or the CPU token-embedding lookup step.
EOF
  exit 0
fi

mem_available_kb() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

status_value_kb() {
  local pid="$1"
  local key="$2"
  awk -v key="$key" '$1 == key":" {print $2}' "/proc/$pid/status" 2>/dev/null || true
}

start_available_kb="$(mem_available_kb)"
min_available_kb="$start_available_kb"
peak_rss_kb=0
peak_hwm_kb=0

"$@" &
pid="$!"

while kill -0 "$pid" 2>/dev/null; do
  rss="$(status_value_kb "$pid" VmRSS)"
  hwm="$(status_value_kb "$pid" VmHWM)"
  available="$(mem_available_kb)"

  if [ -n "$rss" ] && [ "$rss" -gt "$peak_rss_kb" ]; then
    peak_rss_kb="$rss"
  fi
  if [ -n "$hwm" ] && [ "$hwm" -gt "$peak_hwm_kb" ]; then
    peak_hwm_kb="$hwm"
  fi
  if [ "$available" -lt "$min_available_kb" ]; then
    min_available_kb="$available"
  fi

  sleep "$INTERVAL"
done

set +e
wait "$pid"
exit_code="$?"
set -e

end_available_kb="$(mem_available_kb)"
system_peak_delta_kb=$((start_available_kb - min_available_kb))

awk \
  -v exit_code="$exit_code" \
  -v start="$start_available_kb" \
  -v min="$min_available_kb" \
  -v end="$end_available_kb" \
  -v rss="$peak_rss_kb" \
  -v hwm="$peak_hwm_kb" \
  -v delta="$system_peak_delta_kb" \
  'BEGIN {
    printf("exit_code: %d\n", exit_code);
    printf("process_peak_vmrss_mb: %.2f\n", rss / 1024);
    printf("process_peak_vmhwm_mb: %.2f\n", hwm / 1024);
    printf("system_mem_available_start_mb: %.2f\n", start / 1024);
    printf("system_mem_available_min_mb: %.2f\n", min / 1024);
    printf("system_mem_available_end_mb: %.2f\n", end / 1024);
    printf("system_peak_mem_delta_mb: %.2f\n", delta / 1024);
  }'

exit "$exit_code"
