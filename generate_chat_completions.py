from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import json


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--prompt-json-filename", type=str, required=True)
    parser.add_argument("--output-json-filename", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vllm-kwargs-json", type=str, required=True)
    parser.add_argument("--sampling-params-json", type=str, required=True)
    args = parser.parse_args()

    llm = LLM(args.model_name, **json.loads(args.vllm_kwargs_json))
    sampling_params = SamplingParams(**json.loads(args.sampling_params_json))

    with open(args.prompt_json_filename) as f:
        prompts = json.load(f)

    outputs = llm.chat(prompts, sampling_params=sampling_params)

    output_dicts = [
        {
            "completion_text": output.outputs[0].text,
            "prompt_token_ids": output.prompt_token_ids,
            "completion_token_ids": output.outputs[0].token_ids,
            "completion_logprobs": [
                logprobs[token].logprob
                for logprobs, token in zip(
                    output.outputs[0].logprobs,
                    output.outputs[0].token_ids,
                    strict=True,
                )
            ],
            "cumulative_completion_logprob": output.outputs[0].cumulative_logprob,
        }
        for output in outputs
    ]

    with open(args.output_json_filename, "w") as f:
        json.dump(output_dicts, f)


if __name__ == "__main__":
    main()
