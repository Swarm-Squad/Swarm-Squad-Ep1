#!/usr/bin/env bash
# LLM batch for the ICC revision, run on the NVIDIA machine.
#
#   experiments/run_llm_batch.sh check            verify GPU, Ollama, model digest, git state
#   experiments/run_llm_batch.sh start [batch]    check, then run detached (nohup setsid)
#   experiments/run_llm_batch.sh status [batch]   show whether it is running and the log tail
#
# batch: main (paced runs, default) or ablation (jamming with no LLM latency).
# Environment: SEEDS (default 0-9), UV (default ~/.local/bin/uv).
set -euo pipefail
cd "$(dirname "$0")/.."

UV=${UV:-$HOME/.local/bin/uv}
MODEL=llama3.2:3b-instruct-q4_K_M
DIGEST=a80c4f17acd5
OLLAMA=${OLLAMA_HOST:-http://localhost:11434}
SEEDS=${SEEDS:-0-9}
BATCH=${2:-main}
case "$BATCH" in
main) OUT=experiments/results/llm ;;
ablation) OUT=experiments/results/llm_blocking ;;
*)
	echo "unknown batch: $BATCH" >&2
	exit 2
	;;
esac
LOG=$OUT/batch.log
PID=$OUT/batch.pid

fail() {
	echo "check failed: $*" >&2
	exit 1
}

check() {
	[ -x "$UV" ] || fail "uv not found at $UV"
	command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
	nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu \
		--format=csv,noheader || fail "nvidia-smi cannot talk to the driver (reboot needed?)"

	local apps
	apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
		grep -v -i ollama || true)
	[ -z "$apps" ] || fail "another process is using the GPU: $apps"

	local version
	version=$(curl -sf "$OLLAMA/api/version") || fail "Ollama is not running at $OLLAMA"
	echo "ollama $version"

	local digest
	digest=$(ollama list | awk -v m="$MODEL" '$1 == m { print $2 }')
	[ -n "$digest" ] || fail "$MODEL is not pulled"
	[ "$digest" = "$DIGEST" ] || fail "$MODEL has digest $digest, expected $DIGEST"
	echo "model $MODEL $digest"

	curl -sf "$OLLAMA/api/generate" -d "{\"model\": \"$MODEL\", \"prompt\": \"hi\", \
\"stream\": false, \"options\": {\"num_ctx\": 4096, \"num_predict\": 1}}" >/dev/null ||
		fail "warm-up request failed"
	local processor
	processor=$(ollama ps | awk -v m="$MODEL" '$1 == m')
	echo "$processor"
	echo "$processor" | grep -q "100% GPU" || fail "model is not fully on the GPU"

	[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "tracked files have uncommitted changes"
	echo "git $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
}

batch() {
	local run=("$UV" run python -m swarm_squad_ep1.experiments.runner
		--supervisors llm --seeds "$SEEDS" --gpu-power --out "$OUT")
	case "$BATCH" in
	main)
		"${run[@]}" --paced --scenarios obstacles_1 obstacles_2 jamming
		"${run[@]}" --paced --scenarios jamming --c2
		"${run[@]}" --paced --scenarios jamming --c2 --c2-depth 0.5
		;;
	ablation)
		"${run[@]}" --llm-blocking --scenarios jamming
		;;
	esac
	echo "batch finished $(date -Is)"
}

running() {
	[ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null
}

case "${1:-}" in
check)
	check
	;;
start)
	running && fail "batch already running (pid $(cat "$PID"))"
	check
	mkdir -p "$OUT"
	"$UV" sync --frozen >/dev/null
	echo "batch $BATCH started $(date -Is), seeds $SEEDS" >"$LOG"
	setsid nohup "$0" _batch "$BATCH" >>"$LOG" 2>&1 </dev/null &
	echo $! >"$PID"
	echo "started pid $(cat "$PID"), log $LOG"
	;;
status)
	if running; then echo "running (pid $(cat "$PID"))"; else echo "not running"; fi
	[ -f "$OUT/runs.jsonl" ] && echo "$(wc -l <"$OUT/runs.jsonl") runs finished"
	[ -f "$LOG" ] && tail -n 5 "$LOG"
	;;
_batch)
	batch
	;;
*)
	sed -n '2,9p' "$0"
	exit 2
	;;
esac
