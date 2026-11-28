#!/usr/bin/env bash
# GPU setup of the model step (pictures -> 3D model), added to the project's .venv, so ONE environment runs both
# steps (images_to_model.py, model_to_bricks.py, images_to_bricks.py). Linux or WSL2 (Ubuntu) with an NVIDIA GPU.
#
#   bash scripts/setup_images_to_model.sh            add the model step to .venv (created if missing)
#   bash scripts/setup_images_to_model.sh --fresh    delete and rebuild .venv first
#   VENV=.venv-images bash scripts/setup_images_to_model.sh   the model step in its own environment instead
#
# Installs exactly what lock/images_to_model/ lists (plus requirements.txt and requirements-dev.txt), all inside this
# folder: .venv (packages), .deps (TRELLIS.2, CuMesh, FlexGEMM at locked commits, built for this GPU), .models
# (TRELLIS.2-4B, DINOv3, the TRELLIS-image-large decoder and u2net at locked revisions, hash-checked, used offline).
# No NVIDIA research-only code (nvdiffrast, nvdiffrec) is installed or loaded.
#
# Needs, once, by hand (no sudo inside this script):
#   - sudo apt install -y build-essential git curl libgl1 libglib2.0-0
#   - uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   - CUDA toolkit 12.8 (nvcc) and an NVIDIA driver >= 570 (WSL2: the Windows driver; install only the toolkit)
#   - a Hugging Face account with the DINOv3 licence accepted (gated), logged in on this machine (checked first;
#     the script prints the steps when access is missing)
# Downloads (skipped for files already in ~/.cache/huggingface, which are hard-linked): ~5 GB packages, ~18 GB models.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
LOCK="$PROJECT/lock/images_to_model"
VENV="$(realpath -m "${VENV:-$PROJECT/.venv}")"
DEPS="$PROJECT/.deps"
MARKER="$VENV/IMAGES_TO_MODEL_OK"
FRESH=0
for ARGUMENT in "$@"; do
  case "$ARGUMENT" in
    --fresh) FRESH=1 ;;
    *) echo "unknown option $ARGUMENT (only --fresh)" >&2; exit 2 ;;
  esac
done
expected() { awk -v key="$1" '$1 == key { $1 = ""; sub(/^ /, ""); print }' "$LOCK/system.txt"; }
fail() { echo "SETUP FAILED: $*" >&2; exit 1; }

read_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then echo "$HF_TOKEN"; return; fi
  local path="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
  [ -f "$path" ] && tr -d '[:space:]' < "$path"
}

check_hf_access() {  # every pinned Hugging Face repo must be readable with your login, before anything is installed
  local token code missing=0
  token="$(read_hf_token || true)"
  while read -r KIND REPO _; do
    [ "$KIND" = hf ] || continue
    code="$(curl -s -o /dev/null -w '%{http_code}' ${token:+-H "Authorization: Bearer $token"} \
            "https://huggingface.co/api/models/$REPO/auth-check")"
    case "$code" in
      200) echo "Hugging Face access OK: $REPO" ;;
      401) echo "NO ACCESS: $REPO needs a Hugging Face login" >&2; missing=1 ;;
      403) echo "NO ACCESS: $REPO is gated and this account has not been granted access" >&2; missing=1 ;;
      000) fail "cannot reach huggingface.co (network?)" ;;
      *)   echo "NOTE: access check for $REPO returned HTTP $code; the model step will tell if access is missing" ;;
    esac
  done < "$LOCK/models.txt"
  [ "$missing" = 0 ] && return
  cat >&2 <<'MSG'

DINOv3 (used inside TRELLIS.2) is gated. Once per person:
  1. Create a free account at https://huggingface.co/join
  2. Open https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m, accept the DINOv3 License
     (approval can take some time; the page says when access is granted)
  3. Create a "Read" token at https://huggingface.co/settings/tokens and log in on this machine:
       uvx --from huggingface_hub hf auth login
     (or export HF_TOKEN=hf_...)
Then run this script again.
MSG
  fail "no access to the gated model files"
}

echo "== 1/8 prerequisites"
for tool in uv git c++ curl nvidia-smi; do command -v "$tool" >/dev/null || fail "missing $tool (see the top of this script)"; done
NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"; [ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || fail "nvcc not found: install CUDA toolkit $(expected cuda-toolkit)"
NVCC_VERSION="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [ "$NVCC_VERSION" != "$(expected cuda-toolkit)" ]; then
  [ "${ALLOW_OTHER_CUDA:-0}" = 1 ] || fail "nvcc $NVCC_VERSION, lock says $(expected cuda-toolkit) (ALLOW_OTHER_CUDA=1 to try anyway)"
  echo "WARNING: nvcc $NVCC_VERSION instead of $(expected cuda-toolkit)"
fi
CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$NVCC")")")"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
read -r GPU DRIVER CAPABILITY < <(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1 |
  awk -F', ' '{ gsub(/ /, "_", $1); print $1, $2, $3 }')
[ "${DRIVER%%.*}" -ge 570 ] || fail "NVIDIA driver $DRIVER is too old for CUDA 12.8 wheels (need >= 570)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$CAPABILITY}"
echo "GPU ${GPU//_/ } (compute $CAPABILITY, driver $DRIVER), nvcc $NVCC_VERSION, building for $TORCH_CUDA_ARCH_LIST"
[ "${GPU//_/ }" = "$(expected gpu)" ] || echo "NOTE: results were recorded on $(expected gpu); on this GPU numbers may differ slightly"
[ "$(uv --version | awk '{print $2}')" = "$(expected uv)" ] || echo "NOTE: uv $(uv --version | awk '{print $2}'), lock was made with $(expected uv)"

echo "== 2/8 Hugging Face access"
check_hf_access

echo "== 3/8 Python $(expected python) environment $VENV"
if [ "$FRESH" = 1 ]; then rm -rf "$VENV"; fi
[ -x "$VENV/bin/python" ] || uv venv --quiet --python "$(expected python)" "$VENV"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
[ "$PYTHON_VERSION" = "$(expected python)" ] || fail "$VENV has Python $PYTHON_VERSION, need $(expected python) (run with --fresh)"
rm -f "$MARKER"

echo "== 4/8 packages (exact versions, no dependency resolution)"
uv pip install --quiet --no-deps -r requirements.txt -r requirements-dev.txt -r "$LOCK/requirements-gpu.txt"
uv pip install --quiet --no-deps --index-url https://download.pytorch.org/whl/cu128 -r "$LOCK/torch.txt"
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_arch_list())"

echo "== 5/8 source code at locked commits (in $DEPS)"
mkdir -p "$DEPS"
while read -r NAME URL COMMIT _; do
  case "$NAME" in ''|'#'*) continue ;; esac
  [ "$COMMIT" = "-" ] && continue
  [ -d "$DEPS/$NAME/.git" ] || git clone --quiet --no-checkout "$URL" "$DEPS/$NAME"
  git -C "$DEPS/$NAME" fetch --quiet origin "$COMMIT" 2>/dev/null || git -C "$DEPS/$NAME" fetch --quiet origin
  git -C "$DEPS/$NAME" -c advice.detachedHead=false checkout --quiet --force --detach "$COMMIT"
  git -C "$DEPS/$NAME" submodule --quiet update --init --recursive --force
  [ "$(git -C "$DEPS/$NAME" rev-parse HEAD)" = "$COMMIT" ] || fail "$NAME is not at $COMMIT"
  [ -z "$(git -C "$DEPS/$NAME" status --porcelain --untracked-files=no)" ] || fail "$NAME has local changes"
  echo "$NAME @ ${COMMIT:0:12}"
done < "$LOCK/sources.txt"
while read -r NAME SUBMODULE COMMIT; do
  case "$NAME" in ''|'#'*) continue ;; esac
  [ "$(git -C "$DEPS/$NAME/$SUBMODULE" rev-parse HEAD)" = "$COMMIT" ] || fail "$NAME/$SUBMODULE is not at $COMMIT"
done < "$LOCK/submodules.txt"

echo "== 6/8 build CUDA extensions for compute $TORCH_CUDA_ARCH_LIST"
for NAME in CuMesh FlexGEMM; do
  echo "-- $NAME"; uv pip install --quiet --no-deps --no-build-isolation "$DEPS/$NAME"
done
echo "-- o-voxel (from TRELLIS.2)"
rm -rf "$DEPS/build/o-voxel" && mkdir -p "$DEPS/build" && cp -r "$DEPS/TRELLIS.2/o-voxel" "$DEPS/build/o-voxel"
uv pip install --quiet --no-deps --no-build-isolation "$DEPS/build/o-voxel"
UTILS3D="$(awk '$1 == "utils3d" { print $2 "@" $3 }' "$LOCK/sources.txt")"
uv pip install --quiet --no-deps "utils3d @ git+$UTILS3D"
echo "$DEPS/TRELLIS.2" > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/trellis2_repo.pth"

echo "== 7/8 environment = lock files"
python -m tools.images_to_model.check_packages requirements.txt requirements-dev.txt "$LOCK/requirements-gpu.txt" \
  "$LOCK/torch.txt" "$LOCK/builds.txt" || fail "environment differs from the lock files"

echo "== 8/8 models at locked revisions (in .models), then verify"
python -m tools.images_to_model.prepare_models --lock "$LOCK/models.txt" --cache "$PROJECT/.models" ||
  fail "model files differ from the lock"
python tests/run_all.py images_to_model || fail "the model-step unit tests failed"
python - <<'PY'
import torch
from image_to_bricks.images_to_model.provenance import list_restricted_modules
from image_to_bricks.images_to_model.trellis_engine import prepare_trellis_environment, use_project_model_cache
use_project_model_cache()
prepare_trellis_environment()
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline  # noqa: F401
q = torch.randn(1, 96, 8, 64, device="cuda", dtype=torch.float16)
out = xops.memory_efficient_attention(q, q, q, attn_bias=BlockDiagonalMask.from_seqlens([40, 56]))
first = q[:, :40].transpose(1, 2)
ref = torch.nn.functional.scaled_dot_product_attention(first, first, first).transpose(1, 2)
assert (out[:, :40] - ref).abs().max().item() < 0.02, "xformers result differs from PyTorch attention"
assert not list_restricted_modules(), list_restricted_modules()
print("GPU", torch.cuda.get_device_name(0), "| attention OK | TRELLIS.2 imports OK | restricted modules loaded: none")
PY
date -Iseconds > "$MARKER"
echo "SETUP DONE. Next: bash scripts/reproduce_images_to_model.sh, then $VENV/bin/python tests/test_images_to_bricks_gpu.py"
