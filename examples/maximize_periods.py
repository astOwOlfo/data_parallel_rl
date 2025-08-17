from data_parallel_grpo import (
    grpo_train,
    GRPOConfig,
    Environment,
    EnvironmentMaker,
    Message,
)


class MaximizePeriodsEnvironment(Environment):
    def __init__(self) -> None:
        self.step = 0
        self.reward = 0

    async def initial_system_or_user_messages(self) -> list[Message]:
        return [{"role": "user", "content": "Please say something."}]

    async def next_user_messages(
        self, new_assistant_message: str
    ) -> list[Message] | None:
        self.reward += new_assistant_message.count(".")
        self.step += 1
        if self.step == 1:
            return [{"role": "user", "content": "Please say something else."}]
        else:
            return None

    async def get_reward(self) -> float:
        return float(self.reward)


class MaximizePeriodsEnvironmentMaker(EnvironmentMaker):
    def make_environments(
        self, epoch: int, n_groups: int, group_size: int
    ) -> list[list[Environment]]:
        return [
            [MaximizePeriodsEnvironment() for _ in range(group_size)]
            for _ in range(n_groups)
        ]


def main():
    grpo_train(
        environment_maker=MaximizePeriodsEnvironmentMaker(),
        cfg=GRPOConfig(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            epochs=64,
            n_groups=16,
            group_size=4,
            use_wandb=True,
            compile_huggingface_model=True,
        ),
    )


if __name__ == "__main__":
    main()
