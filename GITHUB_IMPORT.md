# GitHub Import Steps

Use this to publish this folder as a completely new project.

## 1) Initialize local repo

```bash
cd unitree_g1_mujoco_ppo_hpc
git init
git add .
git commit -m "Initial commit: Unitree G1 MuJoCo PPO HPC setup"
git branch -M main
```

## 2) Create empty GitHub repository

Create a new repository on GitHub UI with no README, no .gitignore, no license.

Example name:

- `unitree-g1-mujoco-ppo-hpc`

## 3) Connect and push

```bash
git remote add origin https://github.com/<your-username>/unitree-g1-mujoco-ppo-hpc.git
git push -u origin main
```

## Optional: with GitHub CLI

If `gh` is installed and authenticated:

```bash
gh repo create unitree-g1-mujoco-ppo-hpc --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if needed.
