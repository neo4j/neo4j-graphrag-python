.. _llm-page:

***********************************************
LLMs
***********************************************

This page gathers LLM-related documentation: configuring the built-in
providers and extending them to reach custom endpoints.

.. _llm-extensibility:

Extending LLMs: the Base* classes
=================================

:class:`~neo4j_graphrag.llm.anthropic_llm.BaseAnthropicLLM`,
:class:`~neo4j_graphrag.llm.openai_llm.BaseOpenAILLM`, and
:class:`~neo4j_graphrag.llm.google_genai_llm.BaseGeminiLLM` are the shared
base classes behind :class:`~neo4j_graphrag.llm.anthropic_llm.AnthropicLLM`,
:class:`~neo4j_graphrag.llm.openai_llm.OpenAILLM`, and
:class:`~neo4j_graphrag.llm.google_genai_llm.GeminiLLM` respectively. They
hold all the provider-agnostic logic (message building, schema conversion,
response parsing, structured output handling) and leave SDK client
construction to their subclasses. This page documents the
``http_client``/``base_url`` injection contract they expose, so you can point
each provider at a custom or self-hosted, API-compatible endpoint.

``AnthropicLLM`` and ``OpenAILLM`` accept two related, independent
constructor settings:

- ``base_url`` (``Optional[str]``): an explicit constructor parameter on both
  classes that overrides the default API endpoint. Passed through to both the
  sync and async SDK clients (``anthropic.Anthropic`` /
  ``anthropic.AsyncAnthropic`` for Anthropic, and the equivalent OpenAI SDK
  clients for OpenAI).
- ``http_client`` (``Optional[httpx.Client | httpx.AsyncClient]``): an
  already-configured ``httpx`` client to use for requests, e.g. to add custom
  TLS settings, proxies, or timeouts. Accepted through ``**kwargs`` by both
  classes. The concrete class inspects the type of the object you pass: an
  ``httpx.Client`` is routed to the sync SDK client, and an
  ``httpx.AsyncClient`` is routed to the async SDK client. Passing something
  else emits a ``UserWarning`` (via ``warnings.warn``) and falls back to the
  SDK's default client.

Both settings can be used together: ``base_url`` changes where requests go,
while ``http_client`` changes how they're sent. Note that a single
``http_client`` only customizes one direction: calling the other one (e.g.
``ainvoke`` after passing a sync ``httpx.Client``) still works and targets
``base_url``, but through the SDK's default transport.

.. note::

   A ``base_url`` configured on the ``httpx`` client itself is ignored (a
   ``UserWarning`` is emitted when one is detected): both
   SDKs build absolute request URLs from their own ``base_url`` and only use
   the ``httpx`` client as transport. To change the endpoint, always use the
   ``base_url`` constructor parameter:

   .. code-block:: python

      # IGNORED -- the httpx base_url is never used; requests still go to
      # the SDK's default endpoint
      AnthropicLLM(
          model_name="...",
          http_client=httpx.Client(base_url="https://my-endpoint"),
      )

      # WORKS -- requests go to the custom endpoint, with the custom transport
      AnthropicLLM(
          model_name="...",
          base_url="https://my-endpoint",
          http_client=httpx.Client(proxy="http://my-proxy:8080"),
      )

The sync/async routing is implemented by
:func:`neo4j_graphrag.llm.utils.split_http_client_kwargs`, which is exported
for exactly one reason: a custom subclass that constructs its own SDK clients
should call it too, so it preserves the same routing contract instead of
reintroducing the type-mismatch bug the helper fixes.

Subclassing example
-------------------

Because :class:`~neo4j_graphrag.llm.anthropic_llm.BaseAnthropicLLM` /
:class:`~neo4j_graphrag.llm.openai_llm.BaseOpenAILLM` only require the concrete
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
building are inherited from :class:`~neo4j_graphrag.llm.anthropic_llm.BaseAnthropicLLM` unchanged; the subclass only
needs to decide how ``client``/``async_client`` get constructed.

The same pattern applies to :class:`~neo4j_graphrag.llm.openai_llm.BaseOpenAILLM`
for OpenAI-compatible endpoints.

Gemini: single client, ``http_options`` instead of ``http_client``
------------------------------------------------------------------

:class:`~neo4j_graphrag.llm.google_genai_llm.GeminiLLM` follows the same
contract with two SDK-specific differences:

- The ``google.genai`` SDK uses a **single** ``genai.Client`` for both
  directions (async calls go through ``client.aio``), so there is no
  ``http_client`` sync/async routing and ``split_http_client_kwargs`` does
  not apply.
- ``genai.Client`` has no top-level ``base_url`` argument: the endpoint lives
  in ``http_options``. ``GeminiLLM`` still accepts the same explicit
  ``base_url`` constructor parameter and applies it through ``http_options``
  for you — if you also pass ``http_options`` (as a dict or
  ``types.HttpOptions``), only its ``base_url`` field is overridden.

.. code-block:: python

    from neo4j_graphrag.llm import GeminiLLM

    llm = GeminiLLM(
        model_name="gemini-2.0-flash",
        base_url="https://my-gemini-endpoint.example.com",
    )

A custom subclass of
:class:`~neo4j_graphrag.llm.google_genai_llm.BaseGeminiLLM` only needs to
assign ``self.client`` (a ``genai.Client``); all message building, config
handling, and response parsing are inherited.
