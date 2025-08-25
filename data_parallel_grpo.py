# TODO: various algorithmic improvements to GRPO. namely, do compact filtering, increase the upper (but not the lower) clipping epsilon, support learning rate warmup, and do length penalties in a way that doesn't break everything when we normalize advantages. what else?

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.lora.request import LoRARequest
from vllm.entrypoints.chat_utils import (
    resolve_chat_template_content_format,
    parse_chat_messages,
    apply_hf_chat_template,
)
from vllm.inputs import TokensPrompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch
from torch import Tensor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer, AdamW
import wandb
from datetime import datetime
from time import perf_counter
import atexit
from pathlib import Path
from shutil import rmtree
from os import mkdir
import os
import sys
from uuid import uuid4
from tqdm import tqdm, trange
from tqdm.asyncio import tqdm as asyncio_tqdm
from traceback import print_exc
import random
import asyncio
from copy import deepcopy
import json
from statistics import mean, stdev
from more_itertools import chunked, pairwise
from itertools import chain
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace, asdict
from collections.abc import Iterable
from typing import Any, ContextManager
from jaxtyping import Float


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    model: str
    """Huggingface name of the model to train, e.g. Qwen/Qwen3-4B"""

    epochs: int
    """Number of epochs to train for. One epoch is one round of sampling n_groups * group_size rollouts and one round of training the model on them."""

    n_groups: int
    """Number of distinct environments to use at each eapoch. Each environment will be copied group_size times. See a description of how GRPO works [e.g. here](https://arxiv.org/abs/2501.12948) if you don't understand this."""

    group_size: int
    """Number of times to copy each environment. See a description of how GRPO works [e.g. here](https://arxiv.org/abs/2501.12948) if you don't understand this."""

    train_on_each_step_separately: bool = True
    """For now, this should always be true because there is a bug when this is False. TODO: Fix the bug and explain this parameter."""

    clip_epsilon_low: float = 0.2
    """Lower epsilon for clipping probability ratios. See a description of how GRPO works [e.g. here](https://arxiv.org/abs/2501.12948) if you don't understand what a clip epsilon is. Don't forget to also change the value of clip_epsilon_high if you change this."""

    clip_epsilon_high: float = 0.2
    """Higher epsilon for clipping probability ratios. In standard GRPO, it is equal to clip_epsilon_low. See a description of how GRPO works [e.g. here](https://arxiv.org/abs/2501.12948) if you don't understand what a clip epsilon is. [This paper](https://arxiv.org/abs/2503.14476) argues that it is better for clip_epsilon_high to be higher than clip_epsilon_low. They use clip_epsilon_low = 0.2 and clip_epsilon_high = 0.28."""

    normalize_advantages: bool = True
    """Whether to divide the advantages in each group by its standard deviation. Standard GRPO does this. [This paper](https://arxiv.org/abs/2503.20783) argues it's better not to do this."""

    unbias_advantages: bool = (
        False  # TODO: should this be called unbias? does it actually remove bias?
    )
    """In GRPO, each advantage is equal to `reward - mean(group_rewards) / (std(group_rewards) + epsilon)`, where reward is the corresponding reward and group_reward is the list of all the rewards in the group of this reward. (Note that there is no division by the std if normalize_rewards=False). If this parameter is set to True, group_rewards in this formula becomes all the rewards in the group except for the one corresponding to the advantage. [This paper](https://machinelearning.apple.com/research/reinforcement-learning-long-horizon) argues that this is better. My best understanding of why this is better is that the standard formula is biased - we want `mean(group_rewards)` and `std(group_rewards)` to estimate the true mean and standard deviation (i.e. the one we would have if we sampled infinitely many rewards). However, if reward is big (respectively small), `mean(group_rewards)` would be (in expectation) bigger (respectively smaller) than the true mean (and the same goes for the standard deviation). Removing `reward` from `group_rewards` removes this bias because now all the rewards in `group_rewards` are sampled independently from `reward`."""

    unbias_completion_length: bool = False
    """By default, GRPO loss is averaged over all the completion tokens. This introduces a length bias which leads GRPO to favor longer answers when the reward is below the group average. Setting this parameter to true fixes this problem by replacing the average by a sum divided by the maximum completion length. [This paper](TO DO) argues that this is good."""

    advantage_normalization_epsilon: float = 1e-6
    """The `epsilon` in the formula `advantage = (reward - mean(group_rewards)) / (std(group_revards) + epsilon)`. This epsilon exists to avoid divisions by zero."""

    group_sequence_policy_optimization: bool = False
    """Instead of computing probability ratios and clipping them for each token, compute them for the whole sequence of generated tokens (that is, multiply all the probability ratios). [This paper by Qwen](https://arxiv.org/abs/2507.18071) argues that this is better. This technique was used to train Qwen3. Note that clip_epsilon_low and clip_epsilon_high should be much smaller when using group sequence policy optimization. In the linked paper, they take clip_epsilon_low = 3e-4 and clip_epsilon_high = 4e-4."""

    truncated_importance_sampling: bool = False
    """vLLM (used for generating rollouts) and HuggingFace transformers (used for training) have implementation differences that make it so that the logits they generate are not exactly the same. [This blogpost](https://fengyao.notion.site/off-policy-rl) argues that this hinders RL training and proposes to mitigate this problem using truncated importance sampling. Enabling this flag enables this mitigation."""

    truncated_importance_sampling_threshold: float = 8.0
    """Only matters when `truncated_importance_sampling` is True. It is the C constant in the blogpost. See the blogpost for an explanation."""

    train_batch_size: int = 64
    """During the training step with AdamW, this is the batch size used to do AdamW steps. TODO: explain what happens when we do multistep"""

    save_path: str = "logs_and_checkpoints"
    """At every epoch, save the current LoRA in a directory named `{save_path}/checkpoints/epoch-{i_epoch}/` and save the rollouts in a file named `{save_path}/rollouts/epoch-{i_epoch}.json`"""

    vllm_sleep: bool = True
    """Whether to use vLLM sleep to free the memory used by the vLLM inference engine during training."""

    vllm_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "gpu_memory_utilization": 0.5,
            "enable_prefix_caching": True,
        }
    )
    """Kwargs to give to `vllm.AsyncLLMEngine.from_engine_args(vllm.AsyncEngineArgs(**kwargs))` when initializing the inference vLLM engine. `gpu_memory_utilization` (i.e. the fraction of each GPU's memory that the vLLM engine consumes) should be small enough for it to be possible to train one copy of the model on each GPU with the remaining memory if vllm sleep is disabled and small enough to hold one copy of the the weights of the model on each GPU if vllm sleep is enabled. Will add `tensor_parallel_size` equal to the number of available GPUs if not provided. It is recommended to include `"enable_prefix_caching": True` when doing multistep."""

    vllm_sampling_params: SamplingParams = SamplingParams(
        temperature=1.0, max_tokens=4096
    )
    """Sampling parameters for the rollout. The `logprobs` field will be overwritten with 1."""

    huggingface_model_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        }
    )
    """Kwargs to pass to `AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)`. Note that this is for the trained copy of the model, the copy of the model used for inference is handled by vLLM and does not receive those kargs."""

    gradient_checkpointing: bool = True
    """Whether to do gradient checkpointing (aka activation checkpointing), a technique which saves memory at the cost of FLOPs during training. This only affects training the huggingface transformer, not the inference vllm. It is recommended to set this unless the model or context length is really small as otherwise you will most probably get out of memory errors. The theoretical cost in FLOPs is 33% (during training - this does not affect inference), I don't know what the cost is in practice."""

    compile_huggingface_model: bool = False
    """Whether to wrap the HuggingFace transformer used for training in `torch.compile`. Currently doesn't work."""

    compile_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    """Kwargs to pass to `torch.compile(huggingface_transformer_used_for_training, **kwargs)`."""

    optimizer_class: type[Optimizer] = AdamW
    """Constructor used to create the optimizer."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-4})
    """Kwargs to pass to `optimizer_class(params, **kwargs)`. Note that this is the way to specify the learning rate."""

    clip_gradient_max_norm: float | None = 1.0
    """If not None, clip the gradient to this max norm before each optimizer step."""

    lora: bool = True
    """Whether to use LoRA. Currently, not using LoRA is not supported, so this must be True."""

    lora_rank: int = 8
    """Rank of the LoRA adapters. Note that this is separate from lora_kwargs because we need to pass it to vllm and not only to the trained huggingface transformer."""

    lora_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"lora_alpha": 16, "target_modules": "all-linear"}
    )
    """Kwargs passed to `peft_model(model, LoRAConfig(r=cfg.lora_rank, **kwargs))`. Note that this should not contain a key named "r" as this key is provided by the `lora_rank` field of the config."""

    restart_vllm_with_merged_lora: bool = False
    """This is a moderately cursed bug fix to make the script work with gpt oss. vLLM does unfortunately not suppport LoRA for gpt oss. So what we do is every time we want to move the lora adapter from the trained HuggingFace transformer to the vLLM inference engine, we destroy the vLLM engine, merge the LoRA adapter, save the full weights of the model with the merged LoRA adapter, and create a new vLLM engine from those weights. This takes non negligible time because we have to restart the vLLM engine."""

    use_wandb: bool = False
    """Wether to log to weights and biases. You have to set the `WANDB_API_KEY` system variable to use this."""

    wandb_project: str = "data-parallel-rl"

    wandb_run_name: str | None = None

    print_example_rollout: bool = True
    """Print one rollout every epoch."""


Message = dict


# TODO: support tools
class Environment(ABC):
    """
    An instance of this class will be created for each rollout.
    Will only be initialized by an `EnvironmentMaker`.
    Will only be used for one single rollout.
    The preudocode of how the rollout will be generated with this class is the following:
    ```
    all_environments = environment_maker.make_environments(...)
    environment = all_environments[i][j]
    messages: list[Message] = await environment.initial_system_and_user_messages()
    while True:
        assistant_message: str = generate_chat_completion(messages)
        next_user_messages: list[Message] | None = await environment.next_user_messages(assistant_message)
        if next_user_message is None:
            break
        messages.append({"role": "assistant", "content": assistant_message})
        messages += next_user_messages
    reward: float = await environment.get_reward()
    extra_metrics: dict[str, float] = await environment.extra_metrics() # will be plotted on weights and biases and saved on the disk
    logs: Any = await environment.logs() # will be saved on the disk but not plotted on wandb
    await environment_maker.cleanup()
    ```
    """

    @abstractmethod
    async def initial_system_or_user_messages(self) -> list[Message]:
        pass

    @abstractmethod
    async def next_user_messages(
        self, new_assistant_message: str
    ) -> list[Message] | None:
        pass

    @abstractmethod
    async def get_reward(self) -> float:
        pass

    async def extra_metrics(self) -> dict[str, float]:
        """
        Metrics whose averages will be plotted on wandb.
        The keys of the returned dicts must be the same when calling this function on any instance of the any Environment class from the same EnvironmentBuilder.
        """
        return {}

    async def logs(self) -> Any:
        """
        Any object that will be saved to the disk with the rollouts.
        Must be json serializable.
        """
        return None


class EnvironmentMaker(ABC):
    @abstractmethod
    def make_environments(
        self, epoch: int, n_groups: int, group_size: int
    ) -> list[list[Environment]]:
        """
        Must return a list of length `n_groups` each of which elements is of length `group_size`.
        Each element of length `group_size` should contain identical copies of the same environment.
        Will be called once at every epoch with `epoch` equal to the number of this epoch
        """
        pass

    async def cleanup(self, environments: list[list[Environment]]) -> None:
        """
        Will be called once after each call to make_environments after all rollouts with the returned environments have been generated.
        Will be called on the environments that make_environments returned.
        """
        pass


class PrintHowLongItTakes(ContextManager):
    def __init__(self, description: str, disable: bool = False) -> None:
        self.description = description
        self.disable = disable

    def __enter__(self) -> None:
        if not self.disable:
            self.start_time = perf_counter()
            print(f"Starting {self.description}...")

    def __exit__(self, e, t, tb) -> None:
        if not self.disable:
            end_time = perf_counter()
            print(
                f"Finished {self.description}. It took {end_time - self.start_time:.2f} seconds."
            )


@dataclass(slots=True, frozen=True)
class Completion:
    completion_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]


async def chat_completion(
    messages: list[Message],
    vllm_engine: AsyncLLMEngine | AsyncLLM,
    lora_request: LoRARequest | None,
    sampling_params: SamplingParams,
) -> Completion:
    # Note: I copied this code from the chat method of the LLM class in the vLLM library without fully understanding it

    assert sampling_params.logprobs == 1

    model_config = await vllm_engine.get_model_config()
    tokenizer = await vllm_engine.get_tokenizer()

    resolved_content_format = resolve_chat_template_content_format(
        chat_template=None,
        tools=None,
        given_format="auto",
        tokenizer=tokenizer,
        model_config=model_config,
    )

    conversation, multimodal_data = parse_chat_messages(
        messages,  # type: ignore
        model_config,
        tokenizer,
        content_format=resolved_content_format,
    )

    assert multimodal_data is None, (
        "If this assert is triggered when you are not using a multimodal model: this is really weird and should not happen. If this assert is happening when you are using a multimodal model: sorry, I didn't test this code on multimodal models. If this assert is triggered, you have to figure out how to add support for multimodal models. All you have to do might be just removing this assert. But removing this assert might make things fail silently. I don't know, I didn't take the time to understand how multimodal models work with vLLM."
    )

    prompt_str = apply_hf_chat_template(
        tokenizer=tokenizer,  # type: ignore
        conversation=conversation,
        model_config=model_config,
        chat_template=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tools=None,
    )

    prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

    request_id = str(uuid4())
    result_generator = vllm_engine.generate(
        prompt, sampling_params, request_id, lora_request=lora_request
    )

    final_output = None
    async for request_output in result_generator:
        final_output = request_output
    assert final_output is not None

    return Completion(
        completion_text=final_output.outputs[0].text,
        prompt_token_ids=final_output.prompt_token_ids,  # type: ignore
        completion_token_ids=final_output.outputs[0].token_ids,  # type: ignore
        completion_logprobs=[
            logprobs[token].logprob
            for logprobs, token in zip(
                final_output.outputs[0].logprobs,  # type: ignore
                final_output.outputs[0].token_ids,
                strict=True,
            )
        ],
    )


@dataclass(frozen=True, slots=True)
class Rollout:
    completions: list[Completion]
    messages: list[Message]
    reward: float
    extra_metrics: dict[str, float]
    logs: Any


def print_rollout(rollout: Rollout) -> None:
    print("---=== ROLLOUT ===---")
    for message in rollout.messages:
        print(f"=== {message['role'].upper()} MESSAGE ===")
        if set(message.keys()) == {"role", "content"}:
            print(message["content"])
        else:
            print(json.dumps(message, indent=4))
    print("---=== END ROLLOUT ===---")


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    n_completions: int
    n_messages: int
    n_input_tokens_per_completion: float
    n_generated_tokens: float


def get_rollout_metrics(rollout: Rollout) -> RolloutMetrics:
    return RolloutMetrics(
        n_completions=len(rollout.completions),
        n_messages=len(rollout.messages),
        n_input_tokens_per_completion=mean(
            len(completion.prompt_token_ids) for completion in rollout.completions
        ),
        n_generated_tokens=mean(
            len(completion.completion_token_ids) for completion in rollout.completions
        ),
    )


async def generate_single_rollout(
    environment: Environment,
    vllm_engine: AsyncLLMEngine | AsyncLLM,
    lora_request: LoRARequest | None,
    cfg: GRPOConfig,
) -> Rollout:
    # TODO: do something when the rollout runs out of context

    messages: list[Message] = await environment.initial_system_or_user_messages()
    completions: list[Completion] = []

    while True:
        completion = await chat_completion(
            vllm_engine=vllm_engine,
            lora_request=lora_request,
            sampling_params=cfg.vllm_sampling_params,
            messages=messages,
        )

        completions.append(completion)
        messages.append({"role": "assistant", "content": completion.completion_text})

        new_user_messages: list[Message] | None = await environment.next_user_messages(
            completion.completion_text
        )

        if new_user_messages is None:
            break

        messages += new_user_messages

    return Rollout(
        completions=completions,
        messages=messages,
        reward=await environment.get_reward(),
        extra_metrics=await environment.extra_metrics(),
        logs=await environment.logs(),
    )


async def generate_rollouts(
    environment_maker: EnvironmentMaker,
    vllm_engine: AsyncLLMEngine | AsyncLLM,
    lora_request: LoRARequest | None,
    epoch: int,
    cfg: GRPOConfig,
) -> list[Rollout]:
    if cfg.vllm_sleep and await vllm_engine.is_sleeping():
        await vllm_engine.wake_up()

    grouped_environments: list[list[Environment]] = environment_maker.make_environments(
        epoch=epoch, n_groups=cfg.n_groups, group_size=cfg.group_size
    )
    assert len(grouped_environments) == cfg.n_groups
    assert all(len(group) == cfg.group_size for group in grouped_environments)
    environments: list[Environment] = list(chain.from_iterable(grouped_environments))

    rollouts = await asyncio_tqdm.gather(
        *[
            generate_single_rollout(
                environment=environment,
                vllm_engine=vllm_engine,
                lora_request=lora_request,
                cfg=cfg,
            )
            for environment in environments
        ],
        desc="generating rollouts",
    )

    environment_maker.cleanup(grouped_environments)

    if cfg.vllm_sleep:
        # TODO: figure out whether this actually frees all the memory allocated to vllm
        await vllm_engine.sleep(level=1)

    return rollouts


@dataclass(slots=True)
class TrainingDatapoint:
    token_ids: list[int]
    train_mask: list[bool]
    vllm_logprobs: list[float | None]
    huggingface_logprobs: list[float | None] | None
    advantage: float
    n_completions: int

    def __post_init__(self) -> None:
        assert len(self.token_ids) == len(self.train_mask) == len(self.vllm_logprobs)
        assert all(
            (logprob is not None) == mask
            for logprob, mask in zip(self.vllm_logprobs, self.train_mask, strict=True)
        )


def training_datapoints(
    rollout: Rollout, advantage: float, cfg: GRPOConfig
) -> list[TrainingDatapoint]:
    if cfg.train_on_each_step_separately:
        return [
            TrainingDatapoint(
                token_ids=completion.prompt_token_ids + completion.completion_token_ids,
                train_mask=[False] * len(completion.prompt_token_ids)
                + [True] * len(completion.completion_token_ids),
                vllm_logprobs=[None] * len(completion.prompt_token_ids)
                + completion.completion_logprobs,
                huggingface_logprobs=None,
                advantage=advantage,
                n_completions=1,
            )
            for completion in rollout.completions
        ]

    token_ids: list[int] = []
    train_mask: list[bool] = []
    logprobs: list[float | None] = []

    for completion in rollout.completions:
        assert is_prefix(prefix=token_ids, whole=completion.prompt_token_ids)

        token_ids = completion.prompt_token_ids
        train_mask += [False] * (len(token_ids) - len(train_mask))
        logprobs += [None] * (len(token_ids) - len(train_mask))

        token_ids += completion.completion_token_ids
        train_mask += [True] * (len(token_ids) - len(train_mask))
        logprobs += completion.completion_logprobs

    return [
        TrainingDatapoint(
            token_ids=token_ids,
            train_mask=train_mask,
            vllm_logprobs=logprobs,
            huggingface_logprobs=None,
            advantage=advantage,
            n_completions=len(rollout.completions),
        )
    ]


def is_prefix(prefix: list, whole: list) -> bool:
    if len(prefix) > len(whole):
        return False

    return all(x == y for x, y in zip(prefix, whole))


def shuffle_in_same_order(xs: list, ys: list) -> None:
    assert len(xs) == len(ys)

    indices = list(range(len(xs)))
    random.shuffle(indices)
    xs[:] = [xs[i] for i in indices]
    ys[:] = [ys[i] for i in indices]


@torch.no_grad()
def compute_huggingface_logprobs(
    rank: int,
    model: DistributedDataParallel,
    data_for_rank: list[list[TrainingDatapoint]],
) -> None:
    main_process = rank == 0

    # TODO: batching
    for datapoints_for_rollout in tqdm(
        data_for_rank, desc="computing huggingface logits", disable=not main_process
    ):
        for datapoint in datapoints_for_rollout:
            logits: Float[Tensor, " position"] = compute_logprobs(
                rank=rank, model=model, datapoint=datapoint
            )
            datapoint.huggingface_logprobs = [None] + logits.tolist()  # type: ignore


@dataclass(frozen=True, slots=True)
class LossMetrics:
    loss: float
    fraction_clipped: float
    max_probability_ratio: float
    mean_probability_ratio: float
    max_clipped_probability_ratio: float
    mean_clipped_probability_ratio: float


def train_with_gradient_descent(
    rank: int,
    world_size: int,
    model: DistributedDataParallel,
    optimizer: Optimizer,
    data_for_rank: list[list[TrainingDatapoint]],
    cfg: GRPOConfig,
) -> list[LossMetrics]:
    main_process = rank == 0

    all_loss_metrics: list[LossMetrics] = []

    # TODO: batching
    for i, datapoints_for_rollout in enumerate(
        tqdm(data_for_rank, desc="training", disable=not main_process)
    ):
        for datapoint in datapoints_for_rollout:
            # TODO: don't do this computation if the advantage is zero
            # note: torch will complain if there have been zero backward passes on one gpu

            # TODO: check if this works if the number of train datapoints is different on different gpus
            loss, loss_metrics = compute_loss(
                rank=rank,
                model=model,
                datapoint=datapoint,
                cfg=cfg,
            )
            loss.backward()
            all_loss_metrics.append(loss_metrics)

        # TODO: make this divisibility constraint not required
        # TODO: gradient clipping!
        assert cfg.train_batch_size % world_size == 0
        last_iteration = i == len(data_for_rank) - 1
        if i % (cfg.train_batch_size // world_size) or last_iteration:
            dist.barrier()
            if cfg.clip_gradient_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [
                        param
                        for param in model.module.parameters()
                        if param.requires_grad
                    ],
                    max_norm=cfg.clip_gradient_max_norm,
                )
            optimizer.step()
            optimizer.zero_grad()

    return all_loss_metrics


def compute_advantages(rewards: list[float], cfg: GRPOConfig) -> list[float]:
    advantages: list[float] = []

    assert len(rewards) % cfg.group_size == 0
    for group_rewards in chunked(rewards, cfg.group_size):
        for i, reward in enumerate(group_rewards):
            if cfg.unbias_advantages:
                assert cfg.group_size >= 3
                # TODO: link to an explanation of what this does and why this is a good thing to do
                rewards_for_group_statistics = [
                    other_reward
                    for j, other_reward in enumerate(group_rewards)
                    if j != i
                ]
            else:
                # assert cfg.group_size >= 2
                rewards_for_group_statistics = group_rewards

            advantage = reward - mean(rewards_for_group_statistics)

            if cfg.normalize_advantages:
                # TODO: link to a study of whether advantage normalization is good or bad in grpo
                advantage /= (
                    stdev(rewards_for_group_statistics)
                    + cfg.advantage_normalization_epsilon
                )

            advantages.append(advantage)

    assert len(advantages) == len(rewards)

    return advantages


def compute_logprobs(
    rank: int, model: DistributedDataParallel, datapoint: TrainingDatapoint
) -> Float[Tensor, " position"]:
    token_ids: Float[Tensor, " position"] = torch.tensor(datapoint.token_ids).cuda(rank)

    # unsqueeze and squeeze batch dimension
    all_logits: Float[Tensor, "position vocabulary_size"] = model(
        input_ids=token_ids[:-1].unsqueeze(0)
    ).logits.squeeze(0)

    all_logprobs: Float[Tensor, "position vocabulary_size"] = all_logits.log_softmax(-1)

    logprobs: Float[Tensor, " position"] = all_logprobs[
        torch.arange(all_logprobs.size(0)).cuda(rank), token_ids[1:]
    ]

    return logprobs


def compute_loss(
    rank: int,
    model: DistributedDataParallel,
    datapoint: TrainingDatapoint,
    cfg: GRPOConfig,
) -> tuple[Float[Tensor, ""], LossMetrics]:
    logprobs: Float[Tensor, " position"] = compute_logprobs(
        rank=rank, model=model, datapoint=datapoint
    )

    assert datapoint.huggingface_logprobs is not None

    # question: indexing by the mask before passing the tokens to the loss the cleanest way to do masking?
    return grpo_loss(
        logprobs=logprobs[torch.tensor(datapoint.train_mask[1:]).cuda(rank)],
        old_huggingface_logprobs=torch.tensor(
            [
                logprob
                for logprob, mask in zip(
                    datapoint.huggingface_logprobs,
                    datapoint.train_mask,
                    strict=True,
                )
                if mask
            ]
        ).cuda(rank),
        old_vllm_logprobs=torch.tensor(
            [
                logprob
                for logprob, mask in zip(
                    datapoint.vllm_logprobs, datapoint.train_mask, strict=True
                )
                if mask
            ]
        ).cuda(rank),
        advantage=datapoint.advantage,
        n_completions=datapoint.n_completions,
        cfg=cfg,
    )


def grpo_loss(
    logprobs: Float[Tensor, " position"],
    old_huggingface_logprobs: Float[Tensor, " position"],
    old_vllm_logprobs: Float[Tensor, " position"],
    advantage: float,
    n_completions: int,
    cfg: GRPOConfig,
) -> tuple[Float[Tensor, ""], LossMetrics]:
    if cfg.group_sequence_policy_optimization:
        if cfg.unbias_completion_length:
            divide_by = n_completions * cfg.vllm_sampling_params.max_tokens
        else:
            divide_by = logprobs.numel()

        logprobs = logprobs.sum(-1, keepdim=True) / divide_by
        old_huggingface_logprobs = (
            old_huggingface_logprobs.sum(-1, keepdim=True) / divide_by
        )
        old_vllm_logprobs = old_vllm_logprobs.sum(-1, keepdim=True) / divide_by

    probability_ratios: Float[Tensor, " position"] = (
        logprobs - old_huggingface_logprobs
    ).exp()
    clipped_probability_ratios: Float[Tensor, " position"] = torch.clip(
        probability_ratios,
        min=1 - cfg.clip_epsilon_low,
        max=1 + cfg.clip_epsilon_high,
    )

    losses: Float[Tensor, " position"] = -torch.minimum(
        probability_ratios * advantage, clipped_probability_ratios * advantage
    )

    if cfg.truncated_importance_sampling:
        vllm_huggingface_probability_ratios: Float[Tensor, " position"] = (
            old_huggingface_logprobs - old_vllm_logprobs
        ).exp()
        if cfg.truncated_importance_sampling_threshold is not None:
            vllm_huggingface_probability_ratios = torch.min(
                vllm_huggingface_probability_ratios,
                torch.tensor(cfg.truncated_importance_sampling_threshold).to(
                    vllm_huggingface_probability_ratios.device
                ),
            )
        losses = vllm_huggingface_probability_ratios * losses

    loss: Float[Tensor, ""]
    if cfg.group_sequence_policy_optimization:
        assert losses.numel() == 1
        loss = losses.reshape(())
    elif cfg.unbias_completion_length:
        assert cfg.vllm_sampling_params.max_tokens is not None
        # TODO: check if i should multiply by n_completions here
        loss = losses.sum() / (n_completions * cfg.vllm_sampling_params.max_tokens)
    else:
        loss = losses.mean()

    metrics = LossMetrics(
        loss=loss.item(),
        fraction_clipped=(probability_ratios != clipped_probability_ratios)
        .float()
        .mean()
        .item(),
        max_probability_ratio=probability_ratios.max().item(),
        mean_probability_ratio=probability_ratios.mean().item(),
        max_clipped_probability_ratio=clipped_probability_ratios.max().item(),
        mean_clipped_probability_ratio=clipped_probability_ratios.mean().item(),
    )

    return loss, metrics


def make_training_model(rank, cfg: GRPOConfig) -> DistributedDataParallel:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, **cfg.huggingface_model_kwargs
    ).cuda(rank)
    model.train()

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, LoraConfig(r=cfg.lora_rank, **cfg.lora_kwargs))

    assert cfg.lora, "Full parameter fine-tuning is not supported yet."
    if cfg.compile_huggingface_model:
        # TODO: should i do compile before or after wrapping the model in DistributedDataParallel?
        model = torch.compile(model, **cfg.compile_kwargs)

    return DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=False,  # TODO: check if i need find_unused_parameters=True
    )


def make_optimizer(model: DistributedDataParallel, cfg: GRPOConfig) -> Optimizer:
    return cfg.optimizer_class(
        params=[param for param in model.module.parameters() if param.requires_grad],
        **cfg.optimizer_kwargs,
    )


def make_vllm_engine(world_size: int, cfg: GRPOConfig) -> AsyncLLM | AsyncLLMEngine:
    kwargs = cfg.vllm_kwargs

    if "tensor_parallel_size" not in kwargs:
        kwargs["tensor_parallel_size"] = world_size
    if cfg.vllm_sleep and "enable_sleep_mode" not in kwargs:
        kwargs["enable_sleep_mode"] = True

    vllm_engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=cfg.model,
            enable_lora=cfg.lora and not cfg.restart_vllm_with_merged_lora,
            max_lora_rank=cfg.lora_rank,
            **kwargs,
        )
    )

    @atexit.register
    def cleanup_vllm() -> None:
        print("SHUTTING DOWN VLLM")
        vllm_engine.shutdown()  # type: ignore

    return vllm_engine


def update_inference_vllm_engine(
    world_size: int,
    inference_vllm_engine: AsyncLLM | AsyncLLMEngine,
    training_model: DistributedDataParallel,
    epoch: int,
    cfg: GRPOConfig,
) -> tuple[AsyncLLM | AsyncLLMEngine, LoRARequest]:
    if not cfg.restart_vllm_with_merged_lora:
        path = os.path.join(cfg.save_path, "checkpoints", f"epoch-{epoch}")
        training_model.module.save_pretrained(path)
        new_vllm_lora_request = LoRARequest(
            lora_name=f"epoch_{epoch}", lora_int_id=epoch + 1, lora_local_path=path
        )
        return inference_vllm_engine, new_vllm_lora_request

    else:
        inference_vllm_engine.shutdown()
        lora_adapter_path = os.path.abspath(
            os.path.join(cfg.save_path, "checkpoints", f"epoch-{epoch}")
        )
        full_weight_path = os.path.abspath(os.path.join(cfg.save_path, "full_weights"))
        training_model.module.save_pretrained(lora_adapter_path)
        merged_model = deepcopy(training_model.module).cpu()
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(full_weight_path)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        tokenizer.save_pretrained(full_weight_path)
        del merged_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        new_inference_vllm_engine = make_vllm_engine(
            world_size=world_size, cfg=replace(cfg, model=full_weight_path)
        )
        rmtree(full_weight_path)
        lora_request = None
        return new_inference_vllm_engine, lora_request


def save_rollouts(rollouts: list[Rollout], epoch: int, cfg: GRPOConfig) -> None:
    json_rollouts: list[dict] = [
        {
            "messages": rollout.messages,
            "reward": rollout.reward,
            "extra_metrics": rollout.extra_metrics,
            "logs": rollout.logs,
        }
        for rollout in rollouts
    ]

    with open(os.path.join(cfg.save_path, "rollouts", f"epoch-{epoch}.json"), "w") as f:
        json.dump(
            json_rollouts,
            f,
        )


def get_metrics(
    rollouts: list[Rollout], loss_metrics: list[LossMetrics]
) -> dict[str, float]:
    assert all_equal(
        tuple(sorted(rollout.extra_metrics.keys())) for rollout in rollouts
    ), "Environment.extra_metrics should always return dictionaries with the same keys"
    metrics: dict[str, float] = {
        key: mean(rollout.extra_metrics[key] for rollout in rollouts)
        for key in rollouts[0].extra_metrics.keys()
    }
    average_reward = mean(rollout.reward for rollout in rollouts)
    assert "reward" not in metrics.keys(), (
        '"reward" is reserved so it cannot be a key of the dictionaries that Environment.extra_metrics returns'
    )

    metrics["reward"] = average_reward

    assert all_equal(tuple(sorted(asdict(m))) for m in loss_metrics)
    for key in asdict(loss_metrics[0]).keys():
        full_key = f"loss/{key}"
        assert full_key not in metrics.keys(), (
            f"'{full_key}' is reserved so it cannot be a key of the dictionaries that Environment.extra_metrics returns"
        )
        aggregate_fn = max if key.startswith("max") else mean
        metrics[full_key] = aggregate_fn(asdict(m)[key] for m in loss_metrics)

    rollout_metrics: list[RolloutMetrics] = [
        get_rollout_metrics(rollout) for rollout in rollouts
    ]
    assert all_equal(tuple(sorted(asdict(m))) for m in rollout_metrics)
    for key in asdict(rollout_metrics[0]).keys():
        full_key = f"rollout/{key}"
        assert full_key not in metrics.keys(), (
            f"'{full_key}' is reserved so it cannot be a key of the dictionaries that Environment.extra_metrics returns"
        )
        metrics[full_key] = mean(asdict(m)[key] for m in loss_metrics)

    return metrics


def log_and_plot(
    rollouts: list[Rollout], loss_metrics: list[LossMetrics], cfg: GRPOConfig
) -> None:
    metrics: dict[str, float] = get_metrics(
        rollouts=rollouts, loss_metrics=loss_metrics
    )

    print("METRICS:", metrics)

    if cfg.use_wandb:
        wandb.log(metrics)


def all_equal(xs: Iterable) -> bool:
    return all(x == y for x, y in pairwise(xs))


def concatenate_from_all_processes(xs: list[Any], world_size: int) -> list[Any]:
    xs_from_all_processes: list[list | None] = [None] * world_size
    dist.all_gather_object(xs_from_all_processes, xs)
    assert all(xs is not None for xs in xs_from_all_processes)
    return list(chain.from_iterable(xs_from_all_processes))  # type: ignore


def setup_distributed_data_parallel(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"  # wtf is this?
    os.environ["MASTER_PORT"] = "12355"  # wtf is this?
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )
    dist.barrier()  # do i need barrier here?


async def grpo_train_process(
    rank: int, world_size: int, environment_maker: EnvironmentMaker, cfg: GRPOConfig
) -> None:
    setup_distributed_data_parallel(rank=rank, world_size=world_size)

    main_process = rank == 0

    if main_process:
        with PrintHowLongItTakes("initializing vllm inference engine"):
            inference_vllm_engine = make_vllm_engine(world_size=world_size, cfg=cfg)
            vllm_lora_request = None

    dist.barrier()

    training_model = make_training_model(rank=rank, cfg=cfg)
    optimizer = make_optimizer(model=training_model, cfg=cfg)

    dist.barrier()

    if main_process and cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    for epoch in trange(cfg.epochs, desc="grpo trainig", disable=not main_process):
        if main_process:
            with PrintHowLongItTakes("sampling rollouts with vLLM"):
                rollouts: list[Rollout] = await generate_rollouts(
                    environment_maker=environment_maker,
                    vllm_engine=inference_vllm_engine,  # type: ignore
                    lora_request=vllm_lora_request,  # type: ignore
                    epoch=epoch,
                    cfg=cfg,
                )

            if cfg.print_example_rollout:
                print_rollout(rollouts[0])

            with PrintHowLongItTakes("saving rollouts"):
                save_rollouts(rollouts=rollouts, epoch=epoch, cfg=cfg)

            advantages: list[float] = compute_advantages(
                rewards=[rollout.reward for rollout in rollouts],
                cfg=cfg,
            )

            # TODO: this will place the rollouts from one group in different batches. check if this is what we should do or if we should shuffle the groups instead of shuffling the rollouts
            shuffle_in_same_order(rollouts, advantages)

            broadcast = [rollouts, advantages]
        else:
            broadcast = [None, None]

        dist.broadcast_object_list(broadcast, src=0)
        rollouts, advantages = broadcast  # type: ignore

        # TODO: make this divisibility constraint not required
        assert len(rollouts) % world_size == 0

        rollouts_for_rank = rollouts[rank::world_size]
        advantages_for_rank = advantages[rank::world_size]

        training_data_for_rank = [
            training_datapoints(rollout=rollout, advantage=advantage, cfg=cfg)
            for rollout, advantage in zip(
                rollouts_for_rank, advantages_for_rank, strict=True
            )
        ]

        with PrintHowLongItTakes(
            "computing logits with huggingface", disable=not main_process
        ):
            compute_huggingface_logprobs(
                rank=rank,
                model=training_model,
                data_for_rank=training_data_for_rank,
            )

        dist.barrier()

        with PrintHowLongItTakes("training", disable=not main_process):
            loss_metrics: list[LossMetrics] = train_with_gradient_descent(
                rank=rank,
                world_size=world_size,
                model=training_model,
                optimizer=optimizer,
                data_for_rank=training_data_for_rank,
                cfg=cfg,
            )

            loss_metrics = concatenate_from_all_processes(
                loss_metrics, world_size=world_size
            )

        dist.barrier()

        gc.collect()
        torch.cuda.empty_cache()

        dist.barrier()

        if main_process:
            log_and_plot(rollouts=rollouts, loss_metrics=loss_metrics, cfg=cfg)

            with PrintHowLongItTakes(
                "copying lora adapter from the training huggingface transformer to the inference vllm engine"
            ):
                inference_vllm_engine, vllm_lora_request = update_inference_vllm_engine(
                    world_size=world_size,
                    inference_vllm_engine=inference_vllm_engine,  # type: ignore
                    training_model=training_model,
                    epoch=epoch,
                    cfg=cfg,
                )

        dist.barrier()

    if main_process and cfg.use_wandb:
        wandb.finish()


def train_grpo_sync_catching_exceptions(rank: int, *args) -> None:
    try:
        asyncio.run(grpo_train_process(rank, *args))
    except Exception:
        # TODO: when there is an exception, will the program print it and stop or will the program continue printing garbage forever and never stop even if i ctrl+c it? if it's the latter, fix this
        print_exc()

        if dist.is_initialized():
            dist.destroy_process_group()

        sys.exit(1)


def make_save_directories(cfg: GRPOConfig) -> GRPOConfig:
    if not Path(cfg.save_path).exists():
        mkdir(cfg.save_path)
    cfg = replace(
        cfg,
        save_path=os.path.join(
            cfg.save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ),
    )
    assert not Path(cfg.save_path).exists()
    mkdir(cfg.save_path)
    mkdir(os.path.join(cfg.save_path, "checkpoints"))
    mkdir(os.path.join(cfg.save_path, "rollouts"))
    return cfg


def grpo_train(
    environment_maker: EnvironmentMaker, cfg: GRPOConfig, world_size: int | None = None
) -> None:
    cfg = make_save_directories(cfg)

    vllm_sampling_params = deepcopy(cfg.vllm_sampling_params)
    vllm_sampling_params.logprobs = 1
    cfg = replace(cfg, vllm_sampling_params=vllm_sampling_params)

    if world_size is None:
        world_size = torch.cuda.device_count()
    else:
        assert world_size <= torch.cuda.device_count()

    mp.spawn(  # type: ignore
        train_grpo_sync_catching_exceptions,
        args=(world_size, environment_maker, cfg),
        nprocs=world_size,
    )


# ruff: noqa: F722
