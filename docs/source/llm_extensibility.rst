.. _llm-extensibility:

***********************************************
Extending LLMs: BaseAnthropicLLM/BaseOpenAILLM
***********************************************

``BaseAnthropicLLM`` and ``BaseOpenAILLM`` are the shared base classes behind
:class:`~neo4j_graphrag.llm.anthropic_llm.AnthropicLLM` and
:class:`~neo4j_graphrag.llm.openai_llm.OpenAILLM` respectively. They hold all
the provider-agnostic logic (message building, schema conversion, response
parsing, structured output handling) and leave SDK client construction to
their subclasses. This page documents the ``http_client``/``base_url``
injection contract they expose, so you can point either provider at a custom
or self-hosted, API-compatible endpoint.

Both ``AnthropicLLM`` and ``OpenAILLM`` accept two related, independent
constructor arguments:

- ``base_url`` (``Optional[str]``): overrides the default API endpoint. Passed
  through to both the sync and async SDK clients (``anthropic.Anthropic`` /
  ``anthropic.AsyncAnthropic`` for Anthropic, and the equivalent OpenAI SDK
  clients for OpenAI).
- ``http_client`` (``Optional[httpx.Client | httpx.AsyncClient]``): an
  already-configured ``httpx`` client to use for requests, e.g. to add custom
  TLS settings, proxies, or timeouts. The concrete class inspects the type of
  the object you pass: an ``httpx.Client`` is routed to the sync SDK client,
  and an ``httpx.AsyncClient`` is routed to the async SDK client. Passing
  something else logs a warning and falls back to the SDK's default client.

Both arguments can be used together: ``base_url`` changes where requests go,
while ``http_client`` changes how they're sent.

Subclassing example
====================

Because ``BaseAnthropicLLM``/``BaseOpenAILLM`` only require the concrete
subclass to assign ``self.client``/``self.async_client``, you can build your
own thin subclass to reach a custom Anthropic-compatible endpoint with
different defaults or credential handling than the built-in ``AnthropicLLM``:

.. code:: python

    from typing import Any, Optional

    import anthropic

    from neo4j_graphrag.llm import BaseAnthropicLLM


    class MyCustomAnthropicLLM(BaseAnthropicLLM):
        """Talks to a self-hosted, Anthropic-compatible endpoint."""

        def __init__(
            self,
            model_name: str,
            model_params: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ):
            super().__init__(model_name=model_name, model_params=model_params, **kwargs)
            self.client = anthropic.Anthropic(
                base_url="https://my-custom-endpoint.example.com",
                api_key="my-custom-api-key",
            )
            self.async_client = anthropic.AsyncAnthropic(
                base_url="https://my-custom-endpoint.example.com",
                api_key="my-custom-api-key",
            )


    llm = MyCustomAnthropicLLM(model_name="claude-3-opus-20240229")
    llm.invoke("Who is the mother of Paul Atreides?")

All of ``invoke``/``ainvoke``, structured-output handling, and message
building are inherited from ``BaseAnthropicLLM`` unchanged; the subclass only
needs to decide how ``client``/``async_client`` get constructed.

The same pattern applies to :class:`~neo4j_graphrag.llm.openai_llm.BaseOpenAILLM`
for OpenAI-compatible endpoints.
