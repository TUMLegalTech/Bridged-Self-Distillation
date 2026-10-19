"""Record what the closed-teacher and ROPD judge calls actually cost.

The pinned ROPD client already lifts ``usage`` off every response into its
``response_metadata`` (ropd_clients.py), but keeps it only on the error path, so a
completed run leaves no record of its own token spend. Reconstructing it afterwards
means guessing at reasoning tokens, which are billed as output and never appear in
the text, so such an estimate can be off by a large factor.

third_party/ropd_official is vendored without semantic changes, so this hooks the one
seam the pinned client already exposes: ``client_factory``, a constructor keyword whose
default is ``openai.OpenAI`` (ropd_clients.py:1064). Replacing it on a built provider
swaps in a proxy that reads ``usage`` on the way past. Note the default argument is
evaluated at def time, so monkeypatching ``algo.ropd_clients.OpenAI`` after import does
nothing; the attribute has to be set on the provider instance, which is safe because
``_get_client`` reads it lazily and its client cache starts empty.
"""

from __future__ import annotations

import threading
from typing import Any

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cost")


def usage_of(response: Any) -> dict[str, float | int | None]:
    """Pull the billable counters off one response, tolerating either API style.

    ``reasoning_tokens`` is the one worth the trouble: at reasoning_effort=medium it is
    billed as output but invisible in the text. ``cost`` is filled in by gateways that
    price the call themselves (OpenRouter does) and stays None on OpenAI directly.
    """

    usage = getattr(response, "usage", None)
    if usage is None:
        return {field: None for field in USAGE_FIELDS}
    details = getattr(usage, "completion_tokens_details", None)
    def first_present(*names: str) -> float | int | None:
        # `a or b` would read a genuine 0 as missing, which is the very confusion this
        # module exists to remove: a recorded zero and an unrecorded field mean different
        # things, and only the second one should stay absent.
        for name in names:
            value = getattr(usage, name, None)
            if value is not None:
                return value
        return None

    reasoning_details = details if details is not None else getattr(usage, "output_tokens_details", None)
    # OpenAI and OpenRouter report automatic prompt-cache reads here. They are billed
    # at a fraction of the input rate, so without this field the realized input price
    # cannot be reconciled against the list price.
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is None:
        prompt_details = getattr(usage, "input_tokens_details", None)
    return {
        # chat/completions says prompt/completion; responses says input/output.
        "prompt_tokens": first_present("prompt_tokens", "input_tokens"),
        "completion_tokens": first_present("completion_tokens", "output_tokens"),
        "reasoning_tokens": getattr(reasoning_details, "reasoning_tokens", None),
        "cached_tokens": getattr(prompt_details, "cached_tokens", None),
        "cost": first_present("cost"),
    }


class UsageRecorder:
    """Thread-safe totals, kept per stage because the judges are billed separately.

    The ROPD provider pool issues requests from a worker pool, so this is written to
    from several threads at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, dict[str, float]] = {}

    def record(self, stage: str, response: Any) -> None:
        usage = usage_of(response)
        with self._lock:
            totals = self._stages.setdefault(stage, {})
            totals["calls"] = totals.get("calls", 0) + 1
            for field in USAGE_FIELDS:
                value = usage.get(field)
                if value is not None:
                    totals[field] = totals.get(field, 0) + value

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            snap = {stage: dict(totals) for stage, totals in self._stages.items()}
        combined: dict[str, float] = {}
        for totals in snap.values():
            for field, value in totals.items():
                combined[field] = combined.get(field, 0) + value
        if combined:
            snap["total"] = combined
        return snap


class _Proxy:
    """Forward everything to the wrapped SDK object except what we override."""

    def __init__(self, inner: Any, binding: dict[str, Any]) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_binding", binding)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class _RecordingCompletions(_Proxy):
    def create(self, **kwargs: Any) -> Any:
        response = self._inner.create(**kwargs)
        binding = self._binding
        binding["recorder"].record("+".join(sorted(binding["stages"])), response)
        return response


class _RecordingChat(_Proxy):
    @property
    def completions(self) -> Any:
        return _RecordingCompletions(self._inner.completions, self._binding)


class _RecordingClient(_Proxy):
    @property
    def chat(self) -> Any:
        return _RecordingChat(self._inner.chat, self._binding)

    @property
    def responses(self) -> Any:
        return _RecordingCompletions(self._inner.responses, self._binding)


def attach_usage_recorder(client: Any, recorder: UsageRecorder, stage: str) -> bool:
    """Point one built ROPD role client at ``recorder``. Returns False if it has no provider.

    The offline-index teacher has none -- it never calls out -- so a False here is the
    normal answer for that role, not a failure.
    """

    provider = getattr(client, "provider", None)
    if provider is None or not hasattr(provider, "_client_factory"):
        return False

    binding = getattr(provider._client_factory, "_usage_binding", None)
    if binding is not None and binding["recorder"] is not recorder:
        # A second recorder on the same provider. Wrapping again would leave the first
        # proxy underneath, so one request would be counted by BOTH recorders. Rebind in
        # place instead: the proxy chain reads the recorder out of this dict, so the new
        # recorder takes over and nothing double-counts. Last binding wins.
        binding["recorder"] = recorder
        binding["stages"] = {stage}
        return True
    if binding is not None:
        # Roles that resolve to the same profile get the SAME provider object -- with the
        # judges both on one gateway, rubricator and verifier literally share it. Wrapping
        # a second time would record every call twice and invent a per-role split that the
        # SDK layer cannot actually see, so widen the bucket label and leave the chain
        # alone. A "rubricator+verifier" bucket is the honest answer here.
        binding["stages"].add(stage)
        return True

    inner_factory = provider._client_factory
    binding = {"recorder": recorder, "stages": {stage}}

    def factory(**kwargs: Any) -> Any:
        return _RecordingClient(inner_factory(**kwargs), binding)

    factory._usage_binding = binding
    provider._client_factory = factory
    return True
