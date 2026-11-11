#!/bin/bash
# platform/bootstrap.sh: cloud-init runs this once, as root, on every VM the control plane rents
set -euo pipefail
export HF_TOKEN=@HF_TOKEN@ HF_HOME=/opt/hf VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # boot_script() fills the @ fields
CONTROL=@CONTROL@/control VM=$(cat /var/lib/cloud/data/instance-id)               # cloud-init knows our id
ASK="curl -sf --retry 99999 --retry-delay 5 --retry-all-errors -o /opt/job $CONTROL/assign/$VM"   # waits for a 200
SAY="curl -sf -X POST -o /dev/null $CONTROL/state/$VM"

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv sh
/opt/uv/uv venv /opt/venv --python 3.12
/opt/uv/uv pip install --python /opt/venv/bin/python vllm==0.21.1 huggingface_hub
$ASK                                                            # blocks until the reconciler names a model
mapfile -t JOB < /opt/job                                       # the model, its repos and its vLLM flags
$SAY/pulling
echo "${JOB[1]}" | xargs -n1 /opt/venv/bin/hf download --quiet  # one repo, or all five on the standby
[ "${JOB[0]}" = standby ] && $SAY/ready && $ASK?skip=standby && mapfile -t JOB < /opt/job   # wait for promotion
/opt/venv/bin/vllm serve ${JOB[2]} --api-key @GATEWAY_KEY@ --port 8000 &
curl -sf --retry 200 --retry-delay 2 --retry-all-errors -o /dev/null localhost:8000/health
$SAY/serving
wait                                                            # vLLM runs until the reconciler deletes this VM
