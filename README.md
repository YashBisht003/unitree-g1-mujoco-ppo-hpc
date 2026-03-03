# Unitree G1 MuJoCo PPO (Isolated Workspace)

This directory is fully separate from your existing `pybullet_vlm_cbf` project.
It has its own virtual environment, its own cloned dependencies, and its own Slurm scripts.
GitHub publishing steps are in `GITHUB_IMPORT.md`.

## What worked for strong G1 locomotion

Public G1 MuJoCo results (MuJoCo Playground) combine:

- Asymmetric PPO: actor uses `state`, critic uses `privileged_state`.
- Domain randomization during training (friction, masses, armature, qpos0 jitters).
- Command-conditioned velocity tracking task with gait/foot timing terms.
- Disturbance training (random pushes) plus observation noise.
- Large parallel scale and long training horizon.

Paper setup reports very large scale (`32768` envs) and long horizon (`400M`) for G1 flat terrain pretraining, followed by rough-terrain finetuning. Current open-source defaults are lighter (`8192` envs, `200M` steps) and are what the scripts here start from.

## Layout

- `scripts/bootstrap_env.sh`: create isolated `.venv`, clone/update MuJoCo Playground, install deps.
- `scripts/train_g1_flat.sh`: stage-1 flat-terrain PPO.
- `scripts/train_g1_rough_from_flat.sh`: stage-2 rough-terrain finetuning from flat checkpoint.
- `slurm/g1_flat_gpu.sbatch`: Param Ganga style GPU job for flat stage.
- `slurm/g1_rough_gpu.sbatch`: Param Ganga style GPU job for rough stage.

## Quick start (on HPC login node)

```bash
cd unitree_g1_mujoco_ppo_hpc
bash scripts/bootstrap_env.sh
bash scripts/train_g1_flat.sh
```

Preferred submit wrappers:

```bash
cd unitree_g1_mujoco_ppo_hpc
BOOTSTRAP_OFFLINE=0 bash scripts/bootstrap_env.sh   # one-time setup on login node
bash scripts/submit_flat.sh
```

Then rough-terrain finetune from a checkpoint:

```bash
export FLAT_CKPT=/absolute/path/to/mujoco_playground/logs/<flat_run>/checkpoints
bash scripts/train_g1_rough_from_flat.sh
```

## Slurm submission

Flat stage:

```bash
cd unitree_g1_mujoco_ppo_hpc
sbatch slurm/g1_flat_gpu.sbatch
```

Rough stage:

```bash
cd unitree_g1_mujoco_ppo_hpc
sbatch --export=ALL,FLAT_CKPT=/absolute/path/to/checkpoints slurm/g1_rough_gpu.sbatch
```

Helper to auto-pick latest flat checkpoint:

```bash
cd unitree_g1_mujoco_ppo_hpc
FLAT_CKPT="$(bash scripts/latest_flat_checkpoint.sh)"
sbatch --export=ALL,FLAT_CKPT="$FLAT_CKPT" slurm/g1_rough_gpu.sbatch
```

Or with wrapper:

```bash
cd unitree_g1_mujoco_ppo_hpc
export FLAT_CKPT="$(bash scripts/latest_flat_checkpoint.sh)"
bash scripts/submit_rough.sh
```

## Windows to HPC transfer

Create upload archive on your Windows machine:

```powershell
cd C:\Users\Yash Bisht\OneDrive\Desktop\C++\unitree_g1_mujoco_ppo_hpc
powershell -ExecutionPolicy Bypass -File scripts\package_for_upload.ps1
```

Copy archive to Param Ganga login node (from PowerShell):

```powershell
scp .\unitree_g1_mujoco_ppo_hpc_bundle.zip <username>@<login-node>:~
```

On HPC login node:

```bash
unzip -o unitree_g1_mujoco_ppo_hpc_bundle.zip -d unitree_g1_mujoco_ppo_hpc
cd unitree_g1_mujoco_ppo_hpc
bash scripts/submit_flat.sh
```

If your partition name differs (`gpu`, `hip-gpu`, etc.), use `PARTITION=<name> bash scripts/submit_flat.sh` and `PARTITION=<name> bash scripts/submit_rough.sh`.
Memory is not hardcoded now; it uses scheduler defaults unless you pass `MEM=...`.
Submit scripts default to `BOOTSTRAP_OFFLINE=1` (no internet required on compute nodes). Do one-time setup first on login node with `BOOTSTRAP_OFFLINE=0 bash scripts/bootstrap_env.sh`.
For older login nodes (e.g., GCC 4.8), bootstrap pins `ml_dtypes` to binary wheel (`0.5.1`) to avoid C++17 source builds.
Bootstrap defaults to `PIP_NO_CACHE_DIR=1` to reduce disk usage under quota limits.
Bootstrap defaults to `PLAYGROUND_INSTALL_MODE=no_warp`, which skips `warp-lang` and installs a JAX-only stack compatible with older HPC nodes.
Bootstrap also pins `jax/jaxlib` and `flax` to compatible versions to avoid plugin/version drift.
Bootstrap defaults to `USE_MEDIAPY_SHIM=1` in this HPC setup to avoid `mediapy` importing IPython/pyexpat during training startup.
Bootstrap/training set `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPYCACHEPREFIX=.venv/.pycache` to avoid writing bytecode into external conda/system stdlib paths.
Bootstrap now pre-downloads `mujoco_menagerie` on login node using git-compatible clone/checkout logic (works on old git without `-C`), and offline mode validates that assets are present before submit.

By default, `bootstrap_env.sh` checks out MuJoCo Playground commit `f2159f3` for reproducibility. Override with `PLAYGROUND_REF=main` if you want latest.

## Source references

- MuJoCo Playground repo: https://github.com/google-deepmind/mujoco_playground
- G1 env registration: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/__init__.py
- G1 task/reward/noise/push config: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/g1/joystick.py
- G1 domain randomization: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/g1/randomize.py
- PPO defaults for G1: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/config/locomotion_params.py
- MuJoCo Playground paper (Oct 9, 2025): https://arxiv.org/abs/2510.06191
- Unitree MuJoCo official repo: https://github.com/unitreerobotics/unitree_mujoco
- Standalone G1 MuJoCo space: https://huggingface.co/spaces/lerobot/unitree-g1-mujoco
