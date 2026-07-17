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
settings:

- ``base_url`` (``Optional[str]``): overrides the default API endpoint. Passed
  through to both the sync and async SDK clients (``anthropic.Anthropic`` /
  ``anthropic.AsyncAnthropic`` for Anthropic, and the equivalent OpenAI SDK
  clients for OpenAI). ``AnthropicLLM`` declares it as an explicit constructor
  parameter; ``OpenAILLM`` accepts it through ``**kwargs``, from where it is
  forwarded to both SDK clients (any string kwarg is safe to share between
  them).
- ``http_client`` (``Optional[httpx.Client | httpx.AsyncClient]``): an
  already-configured ``httpx`` client to use for requests, e.g. to add custom
  TLS settings, proxies, or timeouts. Accepted through ``**kwargs`` by both
  classes. The concrete class inspects the type of the object you pass: an
  ``httpx.Client`` is routed to the sync SDK client, and an
  ``httpx.AsyncClient`` is routed to the async SDK client. Passing something
  else emits a ``UserWarning`` (via ``warnings.warn``) and falls back to the
  SDK's default client.

Both settings can be used together: ``base_url`` changes where requests go,
while ``http_client`` changes how they're sent.

The sync/async routing is implemented by
:func:`neo4j_graphrag.llm.utils.split_http_client_kwargs`, which is exported
for exactly one reason: a custom subclass that constructs its own SDK clients
should call it too, so it preserves the same routing contract instead of
reintroducing the type-mismatch bug the helper fixes.

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
    from neo4j_graphrag.llm.utils import split_http_client_kwargs


    class MyCustomAnthropicLLM(BaseAnthropicLLM):
        """Talks to a self-hosted, Anthropic-compatible endpoint."""

        DEFAULT_ENDPOINT = "https://my-custom-endpoint.example.com"

        def __init__(
            self,
            model_name: str,
            model_params: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ):
            super().__init__(model_name=model_name, model_params=model_params, **kwargs)
            # Route an optional http_client kwarg to the matching sync/async
            # client, exactly as the built-in AnthropicLLM does.
            sync_params, async_params = split_http_client_kwargs(kwargs)
            sync_params.setdefault("base_url", self.DEFAULT_ENDPOINT)
            async_params.setdefault("base_url", self.DEFAULT_ENDPOINT)
            self.client = anthropic.Anthropic(**sync_params)
            self.async_client = anthropic.AsyncAnthropic(**async_params)


    llm = MyCustomAnthropicLLM(model_name="claude-3-opus-20240229")
    llm.invoke("Who is the mother of Paul Atreides?")

All of ``invoke``/``ainvoke``, structured-output handling, and message
building are inherited from ``BaseAnthropicLLM`` unchanged; the subclass only
needs to decide how ``client``/``async_client`` get constructed.

The same pattern applies to :class:`~neo4j_graphrag.llm.openai_llm.BaseOpenAILLM`
for OpenAI-compatible endpoints.
