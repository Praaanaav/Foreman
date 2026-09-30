from litellm import completion
from orchestrator import config


def chat(prompt: str, model: str | None = None) -> str:
    response = completion(
        model=model or config.SUPERVISOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Say hello in one sentence."))