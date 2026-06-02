"""
API adapter for AuditorAI.

Wraps any LLM API (OpenAI, Anthropic, or custom HTTP endpoint)
for use with the auditor system. Handles text-to-class conversion,
rate limiting with exponential backoff, and concurrent API calls.
"""

import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from auditorai.adapters.base import ModelAdapter


class APIAdapter(ModelAdapter):
    """
    Wraps any LLM API (OpenAI, Anthropic, or custom HTTP endpoint)
    for use with AuditorAI.

    Since API models output text, this adapter requires a
    parse_response function that converts the model's text output
    into a class label and confidence score.

    Usage (OpenAI):
      from auditorai import wrap

      def my_parser(response_text: str) -> tuple:
        # Return (predicted_class, confidence)
        ...

      adapter = wrap("gpt-4o",
                     adapter_type="openai",
                     api_key="sk-...",
                     parse_response=my_parser,
                     n_classes=2,
                     system_prompt="Classify the following as 0 or 1.")

    Usage (Anthropic):
      adapter = wrap("claude-sonnet-4-6",
                     adapter_type="anthropic",
                     api_key="sk-ant-...",
                     parse_response=my_parser,
                     n_classes=3)

    Usage (custom HTTP endpoint):
      adapter = APIAdapter(
        endpoint_url="https://my-model-api.com/predict",
        headers={"Authorization": "Bearer MY_TOKEN"},
        request_builder=lambda x: {"input": x},
        parse_response=my_parser,
        n_classes=2
      )

    Args:
      model_name: Model identifier string (e.g. "gpt-4o", "claude-sonnet-4-6")
                  OR None if using endpoint_url.
      provider: "openai", "anthropic", or "custom". Auto-detected from model_name.
      api_key: API key. If None, reads from environment variable
               (OPENAI_API_KEY or ANTHROPIC_API_KEY).
      parse_response: Callable(response_text: str) -> tuple[int, float]
                      Must return (class_index, confidence_score).
                      confidence_score is used to build the probability vector.
      n_classes: Total number of classes.
      system_prompt: System prompt prepended to every request.
      batch_size: Number of concurrent API calls. Default 5.
                  Higher values risk rate limiting.
      endpoint_url: For custom HTTP endpoints. Mutually exclusive with model_name.
      request_builder: Callable(input_text) -> dict for custom endpoints.
      headers: HTTP headers for custom endpoints.

    IMPORTANT: predict_proba for API models is constructed from the
    parse_response confidence score. If the parser returns class=1
    with confidence=0.87 and n_classes=2, the probability vector is
    [0.13, 0.87]. For n_classes > 2, only the predicted class gets
    the confidence score; remaining probability is distributed evenly.

    Rate limiting: The adapter implements exponential backoff with
    jitter for all API calls. Max retries = 3. Base delay = 1s.
    """

    def __init__(self, model_name: str = None,
                 provider: str = "auto",
                 api_key: str = None,
                 parse_response=None,
                 n_classes: int = 2,
                 system_prompt: str = None,
                 batch_size: int = 5,
                 endpoint_url: str = None,
                 request_builder=None,
                 headers: dict = None):
        self.model_name = model_name
        self.n_classes = n_classes
        self.system_prompt = system_prompt or "You are a classifier."
        self.batch_size = batch_size
        self.endpoint_url = endpoint_url
        self.request_builder = request_builder
        self.headers = headers or {}
        self.parse_response = parse_response
        self.max_retries = 3
        self.base_delay = 1.0

        if provider == "auto":
            self.provider = self._detect_provider(model_name)
        else:
            self.provider = provider

        self.api_key = self._get_api_key(self.provider, api_key)

    def _detect_provider(self, model_name: str) -> str:
        """
        Auto-detects provider from model name.
        "gpt-" or "o1" or "o3" -> "openai"
        "claude-" -> "anthropic"
        "gemini-" -> raises NotImplementedError with helpful message
        None -> "custom"
        """
        if model_name is None:
            return "custom"

        lower = model_name.lower()
        if lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3"):
            return "openai"
        if lower.startswith("claude"):
            return "anthropic"
        if lower.startswith("gemini"):
            raise NotImplementedError(
                "Gemini models are not yet supported as an API adapter. "
                "Use the Google Cloud AI Platform SDK directly and wrap "
                "with a custom adapter."
            )
        return "custom"

    def _get_api_key(self, provider: str, api_key: str) -> str:
        """
        Returns api_key if provided, else reads env var.
        openai    -> OPENAI_API_KEY
        anthropic -> ANTHROPIC_API_KEY
        Raises EnvironmentError with clear message if neither found.
        """
        if api_key is not None:
            return api_key

        if provider == "custom":
            return None

        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }

        env_var = env_map.get(provider)
        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key
            raise EnvironmentError(
                f"No API key found. Either pass api_key=... or set "
                f"the {env_var} environment variable."
            )

        return None

    def _call_with_retry(self, call_fn, text: str) -> str:
        """Wrapper that implements exponential backoff with jitter."""
        for attempt in range(self.max_retries + 1):
            try:
                return call_fn(text)
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate" in error_str or
                    "429" in error_str or
                    "too many" in error_str or
                    "overloaded" in error_str
                )
                if is_rate_limit and attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                raise

    def _call_openai(self, text: str) -> str:
        """
        Makes a single OpenAI chat completion call.
        Uses openai>=1.0 client API.
        Returns the text content of the first choice.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "APIAdapter with OpenAI requires the openai package. "
                "Install it with: pip install openai"
            )

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, text: str) -> str:
        """
        Makes a single Anthropic messages call.
        Uses anthropic>=0.20 client API.
        Returns the text content of the first content block.
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "APIAdapter with Anthropic requires the anthropic package. "
                "Install it with: pip install anthropic"
            )

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model_name,
            max_tokens=100,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": text},
            ],
        )
        return response.content[0].text

    def _call_custom(self, text: str) -> str:
        """
        Makes a POST request to self.endpoint_url.
        Uses self.request_builder(text) to build the request body.
        Returns response.json() as a string.
        """
        import requests

        if self.request_builder is None:
            body = {"input": text}
        else:
            body = self.request_builder(text)

        response = requests.post(
            self.endpoint_url,
            json=body,
            headers=self.headers,
            timeout=30,
        )

        if response.status_code == 429:
            raise Exception("Rate limited (HTTP 429)")

        response.raise_for_status()
        return str(response.json())

    def _call_api(self, text: str) -> str:
        """Routes to the correct API call method."""
        if self.provider == "openai":
            return self._call_with_retry(self._call_openai, text)
        elif self.provider == "anthropic":
            return self._call_with_retry(self._call_anthropic, text)
        elif self.provider == "custom":
            return self._call_with_retry(self._call_custom, text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _probas_from_parse(self, class_idx: int,
                           confidence: float) -> np.ndarray:
        """
        Builds a probability vector from a (class_idx, confidence) pair.
        Shape: (n_classes,).
        The predicted class gets confidence probability.
        Remaining probability is split evenly among other classes.
        Result sums to 1.0.
        """
        confidence = np.clip(confidence, 0.0, 1.0)
        probas = np.zeros(self.n_classes)
        probas[class_idx] = confidence
        remaining = 1.0 - confidence
        other_count = self.n_classes - 1
        if other_count > 0:
            for i in range(self.n_classes):
                if i != class_idx:
                    probas[i] = remaining / other_count
        return probas

    def _process_single(self, text: str) -> tuple:
        """Process a single input text through the API and parser."""
        response_text = self._call_api(text)
        if self.parse_response is not None:
            class_idx, confidence = self.parse_response(response_text)
        else:
            # Default: try to extract a digit as class
            class_idx = 0
            confidence = 0.5
            for char in response_text:
                if char.isdigit():
                    class_idx = int(char)
                    confidence = 0.7
                    break
        return class_idx, confidence

    def predict(self, X) -> np.ndarray:
        """
        X can be list of strings OR np.ndarray (converted to strings).
        Calls the API for each input. Runs calls with ThreadPoolExecutor
        using self.batch_size workers.
        Returns predicted class labels.
        """
        if isinstance(X, np.ndarray):
            X = [str(x) for x in X]
        if isinstance(X, str):
            X = [X]

        results = [None] * len(X)

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_idx = {
                executor.submit(self._process_single, text): i
                for i, text in enumerate(X)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                class_idx, confidence = future.result()
                results[idx] = class_idx

        return np.array(results, dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        """
        Same as predict but returns probability matrix.
        Shape: (n, n_classes).
        """
        if isinstance(X, np.ndarray):
            X = [str(x) for x in X]
        if isinstance(X, str):
            X = [X]

        results = [None] * len(X)

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_idx = {
                executor.submit(self._process_single, text): i
                for i, text in enumerate(X)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                class_idx, confidence = future.result()
                results[idx] = self._probas_from_parse(class_idx, confidence)

        probas = np.array(results)
        self.validate_probas(probas)
        return probas
