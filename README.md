# Installation

Recommended docker image: `nvidia/cuda:12.8.1-base-ubuntu22.04`

1. Install uv by typing the following or following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone this repo:

```
git clone https://<your_github_username>:<your_github_token_with_a_permission_to_clone_this_repo>@github.com/astOwOlfo/data_parallel_grpo.git
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
cd data_parallel_grpo
uv run -m examples.maximize_periods
```

Math on DeepSeek R1 Distill Qwen 14B:

```
cd data_parallel_grpo
uv run -m examples.math
```