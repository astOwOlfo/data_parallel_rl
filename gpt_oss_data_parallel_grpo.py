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
    lora_rank: int = 128
    lora_kwargs: dict[str, Any] = field(default_factory={"alpha": 32, "target_modules": "all-linear"})
    vllm_sampling_params: SamplingParams = SamplingParams(max_tokens=8, temperature=1.0)
    vllm_kwargs: dict[str, Any] = field(
        default_factory={"gpu_memory_utiliation": 0.5, "enable_prefix_caching": True}
    )
    vllm_sleep: bool = True
    huggingface_model_kwargs: dict[str, Any] = field(default_factory={"attn_implementation": "flash_attention_2", torch_dtype=torch.bfloat16})
    gradient_checkpointing: bool = True
    compile_huggingface_model: bool = True
    restart_vllm_with_merged_lora_adapters_every_epoch: bool = False


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


@dataclass(frozen=True, slots=True)
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

    print(f"{final_output.outputs[0].logprobs=}")

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


def make_vllm_engine(world_size: int, cfg: GRPOConfig) -> AsyncLLM:
    # reminder of what `dict | dict` does: this line sets the fields given in the dict literal only if they are missing from cfg.vllm_kwargs
    kwargs = cfg.vllm_kwargs | {
        "tensor_parallel_size": world_size,
        "enable_lora": not cfg.restart_vllm_with_merged_lora_adapters_every_epoch,
        "max_lora_rank": cfg.lora_rank if not cfg.restart_vllm_with_merged_lora_adapters_every_epoch else None,
        "enable_sleep_mode": cfg.vllm_sleep,
    }

    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=cfg.model, **kwargs))


def make_training_model(rank: int, cfg: GRPOConfig) -> Any: # TODO: make type hint DistributedDataParallel
    model = AutoModelForCausalLM(cfg.model, **cfg.huggingface_model_kwargs).cuda(rank)
    model.train()

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, LoraConfig(r=cfg.lora_rank, **cfg.lora_kwargs))

    return model # TODO: wrap in DistributedDataParallel


async def main(world_size: int, cfg: GRPOConfig) -> None:
    sampling_params = deepcopy(cfg.vllm_sampling_params)
    sampling_params.logprobs = 1
    cfg = replace(cfg, vllm_sampling_params=sampling_params)

    inference_vllm_engine = make_vllm_engine(world_size=world_size, cfg=cfg)

    training_model = make_training_model(rank=0, cfg=cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    completion = await chat_completion(
        messages=[{"role": "user", "content": "Please say something"}],
        vllm_engine=inference_vllm_engine,
        lora_request=None,
        sampling_params=cfg.vllm_sampling_params
    )

    print(f"{completion=}")

    print(f"{tokenizer.batch_decode(completion.prompt_token_ids)=}")
    print(f"{tokenizer.batch_decode(completion.completion_token_ids)=}")

    


if __name__ == "__main__":
    asyncio.run(main(world_size=2, cfg=GRPOConfig(model="unsloth/gpt-oss-20b-bf16")))
