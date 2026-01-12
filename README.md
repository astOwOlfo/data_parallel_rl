# What is this repo?

This repo contains a single-file script to run multi GPU (but not multi-node) RL on LLMs using data paralellism. Using data parallelism makes it much easier to train models, however, it requires that the whole weights of the model, as well as the activation cache, fit on a single GPU.

# Installation

Recommended docker image: `volodimir1024/reward-hacking-cuda-128:v1.0`

1. Install the uv python package manager either by running `pip install uv` globally or following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/). Note that uv is already installed in the recommended docker image.

2. Clone this repo:

```
git clone https://<your_github_username>:<your_github_token_with_permissions_to_clone_this_repo>@github.com/astOwOlfo/data_parallel_rl.git
```

3. Install the repo:

```
uv sync
```

Note: if you have issues with flash attention, try:
- Commenting the line starting with `flash-attn = { url = ` in `pyproject.toml`.
- Doing this could lead to `uv sync` taking forever to compile flash attention (to see what `uv sync` is doing, run `uv sync -vv` (meaning very verbose)). If this happens, try replacing the url in the line starting with `flash-attn = { url = ` by the url corresponding to your platform from [here](https://github.com/mjun0812/flash-attention-prebuild-wheels/).

# Running the examples

Toy multistep environment:

```
cd data_parallel_rl
uv run -m examples.maximize_periods
```

Train DeepSeek R1 Distill Qwen 14B on the `allenai/math_qa` math dataset:

```
cd data_parallel_rl
uv run -m examples.math
```

Train GPT OSS 20b on the `allenai/math_qa` math dataset:

```
cd data_parallel_rl
uv run -m examples.gpt_oss_math
```

# Running on your own environments

Familiarize yourself with the docstrings of the `Environment`, `EnvironmentBuilder`, and `GRPOConfig` in file `data_parallel_grpo.py`. Look at the examples in `examples/math.py` and `examples/maximize_periods.py`. Then, copy one of those example files, implement the environment you want in this file, and change the fields of `GRPOConfig` that you want in this file. Run it the same way as running the examples, namely, `uv run -m path.to.you.file`.

Note that the default hyperparameters might be bad right now, it is a work in progress to make them better. Namely, I suspect the default learning rate might be too big by an order of magnitude or two.

# Known issues (IMPORTANT!)

- **IMPORTANT** Often, when the script crashes or when you ctrl+C it, vLLM does not free all the GPU memory properly. If `nvidia-smi` doesn't show that all (or virtually all) the GPU memory is free after running the script, you must run `pkill python` or `killall python` before running the script again. If this doesn't free GPU memory, find the pids of the processes that seem to be launched by this script using `top` or `htop` and `kill` them. I am working on fixing this.

- If you get C compilation errors during vLLM startup with a message similar to `cannot find -lcuda: No such file or directory`, try doing:
```
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
```
