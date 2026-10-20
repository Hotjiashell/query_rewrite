"""Optional model-backed prompt trials; imported only when a trial is requested."""
from __future__ import annotations

import os

from gen_query import LLMConfig, PromptMultiQueryGenerator, build_openai_client
import json


def generate_queries(prompt, dialogue, config_path):
    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    llm = config.get("llm", {})
    key_env = llm.get("api_key_env")
    settings = LLMConfig(
        base_url=llm.get("base_url") or os.getenv("OPENAI_BASE_URL", ""),
        model_name=llm.get("model_name") or os.getenv("OPENAI_MODEL", ""),
        api_key=llm.get("api_key") or (os.getenv(key_env, "") if key_env else "") or os.getenv("OPENAI_API_KEY", ""),
        temperature=llm.get("temperature", 0.0),
    )
    settings.validate()
    client = build_openai_client(settings).with_options(timeout=60, max_retries=0)
    try:
        generator = PromptMultiQueryGenerator(client, settings.model_name, "custom_multi",
                                              temperature=settings.temperature, prompt_template=prompt)
        return generator.generate_queries(dialogue)
    finally:
        client.close()
