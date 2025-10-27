import openai
from openai import AsyncOpenAI
from vllm import SamplingParams
from datasets import load_dataset
import pandas as pd
import re
from random import Random
from enum import Enum
from itertools import count
from dataclasses import dataclass

from examples.templates import (
    AQUARAT_TEMPLATE_STYLIZED_RED_TEAM,
    DEFAULT_GT_INSTRUCTIONS,
    DEFAULT_GT_TEMPLATE,
    BT_NORMAL_TM_TEMPLATE,
    GAMEABLE_TM_INSTRUCTIONS,
)
from data_parallel_grpo import (
    grpo_train,
    GRPOConfig,
    Environment,
    EnvironmentMaker,
    Message,
)


class RewardType(Enum):
    GROUND_TRUTH = "ground_truth"  # sparse
    TRUSTED_MONITOR = "trusted_monitor"  # dense


def fetch_submission(full_submission: str) -> str | None:
    for tag in ["assistantfinal", "final<|message|>", "</think>"]:
        if tag in full_submission:
            return full_submission.split(tag)[-1].strip()

    return None


@dataclass(slots=True)
class DiffuseRLEnvironment(Environment):
    question: str
    true_answer: str
    incorrect_answer: str
    reward_type: RewardType
    grader_timeout_seconds: int
    llm_message: str | None = None
    thinking_parsing_failed: bool = False
    grader_parsing_failed: bool = False
    grader_timed_out: bool = False

    async def initial_system_or_user_messages(self) -> list[Message]:
        return [
            {
                "role": "user",
                "content": AQUARAT_TEMPLATE_STYLIZED_RED_TEAM.format(
                    incorrect_answer=self.incorrect_answer
                ),
            }
        ]

    async def next_user_messages(
        self, new_assistant_message: str
    ) -> list[Message] | None:
        self.llm_message = new_assistant_message
        return None

    async def get_reward(self) -> float:
        assert self.llm_message is not None

        answer_without_reasoning = fetch_submission(self.llm_message)
        if answer_without_reasoning is None:
            self.thinking_parsing_failed = True
            return 0.0

        if self.reward_type == RewardType.GROUND_TRUTH:
            MONITOR_TEMPLATE = DEFAULT_GT_TEMPLATE  # sparse (ground truth)
            MONITOR_INSTRUCTIONS = DEFAULT_GT_INSTRUCTIONS

            reward_model_prompt = MONITOR_TEMPLATE.format(
                question=self.question,
                answer=answer_without_reasoning,
                criterion=self.true_answer,
                instructions=MONITOR_INSTRUCTIONS,
            )

        elif self.reward_type == RewardType.TRUSTED_MONITOR:
            MONITOR_TEMPLATE = BT_NORMAL_TM_TEMPLATE  # dense (proxy for ground truth)
            MONITOR_INSTRUCTIONS = GAMEABLE_TM_INSTRUCTIONS

            reward_model_prompt = MONITOR_TEMPLATE.format(
                question=self.question,
                solution=answer_without_reasoning,
                instructions=MONITOR_INSTRUCTIONS,
            )
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")

        client = AsyncOpenAI()

        for i_retry in count():
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": reward_model_prompt}],
                    timeout=self.grader_timeout_seconds,
                )
            except Exception as e:
                if isinstance(e, (TimeoutError, openai.APITimeoutError)):
                    print("OpenAI call timed out.")
                    self.grader_timed_out = True
                    return 0.0

                delay = 2**i_retry
                print(
                    f"OpenAI call failed on retry {i_retry}. Waiting for {delay} seconds and trying again. The exception is: {e}"
                )
                await asyncio.sleep(delay)
                continue
            break

        openai_output = response.choices[0].message.content

        # parse the response
        # extract the grade
        match = None
        for m in re.finditer(r"GRADE:(.*)", openai_output):
            match = m
        if match:
            value = match.group(1)
            value = "".join(
                filter(str.isdigit, value)
            )  # remove anything that's not a digit
        else:
            # if no grade is found, return 0 and log a warning
            # try look for 'Grade: '
            match = None
            for m in re.finditer(r"Grade:(.*)", openai_output):
                match = m
            if match:
                value = match.group(1)
                value = "".join(
                    filter(str.isdigit, value)
                )  # remove anything that's not a digit
            else:
                value = ""

        if value is None or value == "":
            self.grader_parsing_failed = True
            return 0.0
        else:
            value = float(value)
            if value > 10.0:
                print(f"Value is greater than 10.0: {value}, clipping to 10.0")
                value = 10.0
            elif value < 0.0:
                print(f"Value is less than 0.0: {value}, clipping to 0.0")
                value = 0.0

        return float(value) / 10.0

    async def extra_metrics(self) -> dict[str, float]:
        return {
            "thinking_parsing_failed": float(self.thinking_parsing_failed),
            "grader_parsing_failed": float(self.grader_parsing_failed),
            "grader_timed_out": float(self.grader_timed_out),
        }


class DiffuseRLEnvironmentMaker(EnvironmentMaker):
    def __init__(
        self, reward_type: RewardType, grader_timeout_seconds: int = 120
    ) -> None:
        self.reward_type = reward_type
        self.grader_timeout_seconds = grader_timeout_seconds

        self.dataset = list(pd.read_csv("data/olympiads.csv").to_dict("records"))
        Random(42).shuffle(self.dataset)

    def make_environments(
        self, epoch: int, n_groups: int, group_size: int
    ) -> list[list[Environment]]:
        environments: list[list[Environment]] = []
        for i in range(n_groups * epoch, n_groups * (epoch + 1)):
            datapoint: dict = self.dataset[i % len(self.dataset)]
            environments.append(
                [
                    DiffuseRLEnvironment(
                        question=datapoint["question"],
                        true_answer=datapoint["target"],
                        incorrect_answer=datapoint["stored_incorrect_answer"],
                        reward_type=self.reward_type,
                        grader_timeout_seconds=self.grader_timeout_seconds,
                    )
                    for _ in range(group_size)
                ]
            )
        return environments


def main():
    grpo_train(
        environment_maker=DiffuseRLEnvironmentMaker(
            reward_type=RewardType.GROUND_TRUTH
        ),
        cfg=GRPOConfig(
            # model="unsloth/gpt-oss-20b-bf16",
            model="Qwen/Qwen3-4B",
            epochs=32,
            n_groups=8,
            group_size=8,
            train_batch_size=128,
            clip_epsilon_low=3e-4,
            clip_epsilon_high=4e-4,
            normalize_advantages=False,
            unbias_advantages=True,
            unbias_completion_length=False,
            group_sequence_policy_optimization=True,
            truncated_importance_sampling=False,
            use_wandb=True,
            vllm_sleep=True,
            compile_huggingface_model=False,
            vllm_kwargs={"gpu_memory_utilization": 0.4, "max_model_len": 2048},
            vllm_sampling_params={"max_tokens": 1500, "temperature": 1.0},
            restart_vllm_with_merged_lora=True,
            optimizer_kwargs={"lr": 5e-5},
            lora_rank=128,
            lora_kwargs={"lora_alpha": 256, "target_modules": "all-linear"},
            gpt_oss_reasoning_effort="low",
            gradient_checkpointing=False,
        ),
    )


if __name__ == "__main__":
    main()
