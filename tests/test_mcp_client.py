from __future__ import annotations

from sa_candidate_finder.mcp_client import _parse_sse_jsonrpc


def test_parse_sse_jsonrpc_single_line_data_event():
    raw = (
        "event: message\n"
        "data: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"{}\"}]}}\n"
        "\n"
    )

    rpc = _parse_sse_jsonrpc(raw)
    assert rpc["id"] == 2
    assert rpc["jsonrpc"] == "2.0"


def test_parse_sse_jsonrpc_data_then_continuation_json_line():
    # Matches observed server shape where `data:` is empty and JSON comes on next line.
    raw = (
        "event: message\n"
        "data:\n"
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"{\\\"count\\\":457}\"}]}}\n"
        "\n"
    )

    rpc = _parse_sse_jsonrpc(raw)
    assert rpc["id"] == 2
    assert rpc["result"]["content"][0]["type"] == "text"
