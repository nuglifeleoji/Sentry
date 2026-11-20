"""SDK response metadata must not silently authorize incomplete recoveries."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from Sentry import model_to_text_judge


RESOLVED_JSON = json.dumps({
    "status": "resolved",
    "justification": "The agent resumed observation-grounded progress.",
    "evidence": ["A visible candidate was inspected."],
})
RESPONSE_ERRORS = (TypeError, ValueError, RuntimeError)


def judge_response(response):
    return model_to_text_judge(lambda *args, **kwargs: response)("Judge the recovery")


def chat_response(*, finish_reason="stop", message=None, sdk=False):
    message = {"content": RESOLVED_JSON} if message is None else message
    if sdk:
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(**message),
        )])
    return {"choices": [{"finish_reason": finish_reason, "message": message}]}


@pytest.mark.parametrize("sdk", [False, True])
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls", "function_call", None])
def test_complete_looking_json_is_rejected_when_sdk_reports_incomplete_completion(sdk, finish_reason):
    with pytest.raises(RESPONSE_ERRORS):
        judge_response(chat_response(finish_reason=finish_reason, sdk=sdk))


@pytest.mark.parametrize("sdk", [False, True])
@pytest.mark.parametrize("extra", [
    {"refusal": "Request refused"},
    {"tool_calls": [{"function": {"name": "finish"}}]},
    {"function_call": {"name": "finish", "arguments": "{}"}},
])
def test_refusal_or_tool_request_is_not_treated_as_recovery_evidence(sdk, extra):
    response = chat_response(message={"content": RESOLVED_JSON} | extra, sdk=sdk)
    with pytest.raises(RESPONSE_ERRORS):
        judge_response(response)


@pytest.mark.parametrize("sdk", [False, True])
def test_multiple_choices_do_not_silently_select_a_successful_verdict(sdk):
    response = chat_response(sdk=sdk)
    choices = response.choices if sdk else response["choices"]
    choices.append(choices[0])
    with pytest.raises(RESPONSE_ERRORS):
        judge_response(response)


@pytest.mark.parametrize("block", [
    {"type": "tool_result", "content": RESOLVED_JSON},
    {"type": "refusal", "text": RESOLVED_JSON},
    {"type": "text", "text": 42},
    {"type": "text", "text": {"status": "resolved"}},
    42,
])
def test_unknown_or_nontext_content_blocks_are_rejected(block):
    with pytest.raises(RESPONSE_ERRORS):
        judge_response({"content": [block]})


@pytest.mark.parametrize("sdk", [False, True])
def test_single_finished_chat_completion_retains_exact_text(sdk):
    assert judge_response(chat_response(sdk=sdk)) == RESOLVED_JSON


@pytest.mark.parametrize("sdk", [False, True])
def test_completed_responses_sdk_output_text_is_supported(sdk):
    response = {"status": "completed", "output_text": RESOLVED_JSON}
    if sdk:
        response = SimpleNamespace(**response)
    assert judge_response(response) == RESOLVED_JSON


@pytest.mark.parametrize("sdk", [False, True])
@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress", "queued", "cancelled"])
def test_responses_status_is_checked_before_aggregate_output_text(sdk, status):
    response = {"status": status, "output_text": RESOLVED_JSON}
    if sdk:
        response = SimpleNamespace(**response)
    with pytest.raises(RESPONSE_ERRORS):
        judge_response(response)


def test_completed_response_with_incomplete_details_is_rejected():
    response = SimpleNamespace(
        status="completed", output_text=RESOLVED_JSON,
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    with pytest.raises(RESPONSE_ERRORS):
        judge_response(response)


@pytest.mark.parametrize("stop_reason", ["max_tokens", "tool_use", "refusal", "pause_turn"])
def test_messages_sdk_partial_json_is_not_a_verified_response(stop_reason):
    response = {"stop_reason": stop_reason, "content": [{"type": "text", "text": "{}"}]}
    with pytest.raises(TypeError):
        model_to_text_judge(lambda *a, **kw: response)("JSON")


def test_responses_sdk_text_cannot_hide_tool_or_refusal_output():
    for output in (
        [{"type": "function_call", "name": "external_action"}],
        [{"type": "message", "role": "assistant", "status": "completed", "content": [{"type": "refusal", "refusal": "declined"}]}],
    ):
        response = {"status": "completed", "output_text": "{}", "output": output}
        with pytest.raises(TypeError):
            model_to_text_judge(lambda *a, **kw: response)("JSON")


def test_responses_sdk_complete_message_is_extracted_without_reasoning():
    response = {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "{}"}]},
    ]}
    assert model_to_text_judge(lambda *a, **kw: response)("JSON") == "{}"
