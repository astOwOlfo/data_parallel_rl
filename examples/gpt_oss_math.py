from vllm import SamplingParams
from datasets import load_dataset
from random import Random

from data_parallel_grpo import (
    grpo_train,
    GRPOConfig,
    Environment,
    EnvironmentMaker,
    Message,
)


class MathEnvironment(Environment):
    def __init__(
        self,
        problem_statement: str,
        possible_answers: str,
        correct_answer: str,
    ) -> None:
        self.problem_statement = problem_statement
        self.possible_answers = possible_answers
        self.correct_answer = correct_answer
        self.step = 0

    async def initial_system_or_user_messages(self) -> list[Message]:
        prompt = "Please answer the following multiple choice question:\n\n"
        prompt += self.problem_statement
        prompt += "\n\nPossible answers:\n"
        prompt += self.possible_answers
        prompt += "\n\nPlease reason step by step, and put the letter of your final answer within \\boxed{}, exactly as follows: \\boxed{z}"
        return [{"role": "user", "content": prompt}]

    async def next_user_messages(
        self, new_assistant_message: str
    ) -> list[Message] | None:
        self.llm_message = new_assistant_message
        return None

    async def get_reward(self) -> float:
        if "\\boxed{" not in self.llm_message:
            return 0.0
        m = self.llm_message.split("\\boxed{")[-1]  # type: ignore
        if len(m) == 0:
            return 0.0
        given_answer = m[0]
        correct = given_answer.lower() == self.correct_answer.lower()
        if correct:
            return 1.0
        else:
            return 0.0


class MathEnvironmentMaker(EnvironmentMaker):
    def __init__(self) -> None:
        self.dataset = list(
            load_dataset("allenai/math_qa", split="train", trust_remote_code=True)
        )
        Random(42).shuffle(self.dataset)

    def make_environments(
        self, epoch: int, n_groups: int, group_size: int
    ) -> list[list[Environment]]:
        environments: list[list[Environment]] = []
        for i in range(n_groups * epoch, n_groups * (epoch + 1)):
            datapoint: dict = self.dataset[i % len(self.dataset)]
            environments.append(
                [
                    MathEnvironment(
                        problem_statement=datapoint["Problem"],
                        possible_answers=datapoint["options"],
                        correct_answer=datapoint["correct"],
                    )
                    for _ in range(group_size)
                ]
            )
        return environments


def main():
    grpo_train(
        environment_maker=MathEnvironmentMaker(),
        cfg=GRPOConfig(
            model="unsloth/gpt-oss-20b-bf16",
            epochs=32,
            n_groups=256,
            group_size=8,
            clip_epsilon_low=3e-4,
            clip_epsilon_high=4e-4,
            normalize_advantages=False,
            unbias_advantages=False,
            unbias_completion_length=False,
            group_sequence_policy_optimization=True,
            truncated_importance_sampling=True,
            use_wandb=True,
            vllm_sleep=True,
            compile_huggingface_model=True,
            vllm_kwargs={"gpu_memory_utilization": 0.5, "max_model_len": 12288},
            vllm_sampling_params=SamplingParams(max_tokens=8192, temperature=1.0),
            restart_vllm_with_merged_lora=True,
            optimizer_kwargs={"lr": 5e-5},
            lora_rank=128,
            lora_kwargs={"lora_alpha": 32, "target_modules": "all-linear"},
        ),
    )


if __name__ == "__main__":
    main()