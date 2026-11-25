"""POST payloads and monitor responses without changing learning inputs."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event, Lock, get_ident
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from agent_system import decision_model
from agent_system.decision_model import DecisionModelRequestBatch, DecisionModelUnavailable, submit_decision_model_requests


URL = "http://127.0.0.1:8009/v1/systemone"
REAL_POST = aiohttp.ClientSession.post


@pytest.fixture(autouse=True)
def mock_http_transport(monkeypatch):
    """Exercise async dispatch with the existing HTTP fixtures; GET still uses urllib."""
    @asynccontextmanager
    async def post(session, url, *, data, headers, timeout, allow_redirects):
        request = Request(url, data=data, headers=headers, method="POST")
        try:
            result = decision_model.urlopen(request, timeout=timeout.sock_read)
        except HTTPError as exc:
            raise aiohttp.ClientResponseError(SimpleNamespace(real_url=url), (), status=exc.code) from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise exc.reason
            raise aiohttp.ClientConnectionError(str(exc)) from exc
        async def read():
            return result.read()
        yield SimpleNamespace(status=result.status, read=read)

    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    yield
    decision_model._shutdown_service()


def payload(*ids):
    return {"model": "kev-latest", "state": "Search trace", "questions": {
        qid: {"type": "choice", "instructions": qid, "criteria": {"required": None, "no effect": None}}
        for qid in ids
    }}


def response(body=None, raw=None):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status = 200
    result.read.return_value = json.dumps(body).encode() if raw is None else raw
    return result


def answer(choice="required", p=0.8):
    return {"choice": choice, "probabilities": {"required": p, "no effect": 1 - p}}


def test_posts_exact_payloads_and_aggregates_answers(monkeypatch):
    requests = {"a": payload("step_1", "step_2"), "b": payload("step_1"), "empty": payload()}
    original = deepcopy(requests)

    def open_request(request, **kwargs):
        if "step_2" in json.loads(request.data)["questions"]:
            return response({"answers": {"step_1": answer(), "step_2": answer("no effect", 0.2)}})
        return response({"answers": {"step_1": answer(p=0.9)}})

    opener = MagicMock(side_effect=open_request)
    monkeypatch.setattr("agent_system.decision_model.urlopen", opener)
    metrics, probabilities = submit_decision_model_requests(requests, URL, timeout=123, api_key="test-token")
    assert probabilities == {
        "a": {"step_1": answer()["probabilities"], "step_2": answer("no effect", 0.2)["probabilities"]},
        "b": {"step_1": answer(p=0.9)["probabilities"]},
    }
    assert requests == original
    assert opener.call_count == 2
    sent_payloads = []
    for call in opener.call_args_list:
        sent = call.args[0]
        assert sent.full_url == URL and sent.get_method() == "POST"
        sent_payloads.append(json.loads(sent.data))
        assert sent.get_header("Content-type") == "application/json"
        assert sent.get_header("Authorization") == "Bearer test-token"
        assert call.kwargs["timeout"] == 123
    assert all(expected in sent_payloads for expected in [requests["a"], requests["b"]])
    assert metrics["decision_model/answer_count"] == 3
    assert metrics["decision_model/successful_response_count"] == 2
    assert metrics["decision_model/post_request_count"] == 2
    assert metrics["decision_model/skipped_empty_request_count"] == 1
    assert metrics["decision_model/failed_request_count"] == 0
    assert metrics["decision_model/unanswered_question_count"] == 0
    assert metrics["decision_model/choice_fraction/required"] == pytest.approx(2 / 3)
    assert metrics["decision_model/choice_count/no_effect"] == 1
    assert metrics["decision_model/probability_mean/required"] == pytest.approx(1.9 / 3)
    assert metrics["decision_model/selected_probability_mean"] == pytest.approx(2.5 / 3)
    assert metrics["decision_model/total_request_processing_time"] >= 0
    assert not any("post_response_time" in key for key in metrics)


@pytest.mark.parametrize("failure", [
    TimeoutError("timed out"), URLError(TimeoutError("timed out")),
    HTTPError(URL, 504, "Gateway timeout", {}, None),
])
def test_timeout_halts_and_counts_other_responses(monkeypatch, failure):
    def open_request(request, **kwargs):
        if "fail" in json.loads(request.data)["questions"]:
            raise failure
        return response({"answers": {"step_1": answer()}})

    opener = MagicMock(side_effect=open_request)
    monkeypatch.setattr("agent_system.decision_model.urlopen", opener)
    with pytest.raises(DecisionModelUnavailable) as caught:
        submit_decision_model_requests({"a": payload("fail"), "b": payload("step_1")}, URL)
    metrics = caught.value.metrics
    assert metrics["decision_model/timeout_count"] == 1
    assert metrics["decision_model/failed_request_count"] == 1
    assert metrics["decision_model/answer_count"] == 1
    assert metrics["decision_model/unanswered_question_count"] == 1
    assert metrics["decision_model/request_success_rate"] == 0.5
    assert opener.call_args.args[0].get_header("Authorization") is None


@pytest.mark.parametrize("body", [
    [], {}, {"answers": {}}, {"answers": {"wrong_id": answer()}},
    {"answers": {"step_1": {"choice": "unknown", "probabilities": {"required": 1, "no effect": 0}}}},
    {"answers": {"step_1": {"choice": "required", "probabilities": {"required": 1}}}},
    {"answers": {"step_1": answer(p=float("nan"))}},
    {"answers": {"step_1": answer(p=2)}},
    {"answers": {"step_1": {"choice": "required", "probabilities": {"required": True, "no effect": 0}}}},
    {"answers": {"step_1": {"choice": "required", "probabilities": {"required": 0.2, "no effect": 0.2}}}},
])
def test_invalid_answers_are_failed_requests(monkeypatch, body):
    monkeypatch.setattr("agent_system.decision_model.urlopen", MagicMock(return_value=response(body)))
    with pytest.raises(DecisionModelUnavailable) as caught:
        submit_decision_model_requests({"a": payload("step_1")}, URL)
    metrics = caught.value.metrics
    assert metrics["decision_model/invalid_response_count"] == 1
    assert metrics["decision_model/answer_count"] == 0
    assert metrics["decision_model/failed_request_count"] == 1
    assert "decision_model/selected_probability_mean" not in metrics


def test_invalid_json_waits_for_all_requests_and_full_statistics(monkeypatch):
    def open_request(request, **kwargs):
        if "fail" in json.loads(request.data)["questions"]:
            return response(raw=b"bad json")
        return response({"answers": {"step_1": answer()}})

    opener = MagicMock(side_effect=open_request)
    monkeypatch.setattr("agent_system.decision_model.urlopen", opener)
    requests = {"failed": payload("fail"), **{str(i): payload("step_1") for i in range(130)}, "empty": payload()}
    with pytest.raises(DecisionModelUnavailable) as caught:
        submit_decision_model_requests(requests, URL)
    assert opener.call_count == 131
    assert isinstance(caught.value.__cause__, ValueError)
    metrics = caught.value.metrics
    assert metrics["decision_model/post_request_count"] == 131
    assert metrics["decision_model/successful_response_count"] == 130
    assert metrics["decision_model/failed_request_count"] == 1
    assert metrics["decision_model/skipped_empty_request_count"] == 1
    assert metrics["decision_model/invalid_response_count"] == 1
    assert metrics["decision_model/answer_count"] == 130
    assert metrics["decision_model/unanswered_question_count"] == 1
    assert metrics["decision_model/total_request_processing_time"] >= 0
    assert not any("post_response_time" in key for key in metrics)


def test_overlapping_batches_share_one_thread_and_128_in_flight_requests(monkeypatch):
    release = Event()
    saturated = Event()
    lock = Lock()
    worker_ids = set()
    active = peak = attempted = 0

    @asynccontextmanager
    async def post(session, url, *, data, **kwargs):
        nonlocal active, peak, attempted
        with lock:
            worker_ids.add(get_ident())
            active += 1
            attempted += 1
            peak = max(peak, active)
            if active == 128:
                saturated.set()
        try:
            while not release.is_set():
                await asyncio.sleep(0.001)
            body = {"answers": {qid: answer() for qid in json.loads(data)["questions"]}}
            async def read():
                return json.dumps(body).encode()
            yield SimpleNamespace(status=200, read=read)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    requests = {str(i): payload(f"step_{i}") for i in range(129)}
    with ThreadPoolExecutor(max_workers=2) as callers:
        batches = [callers.submit(submit_decision_model_requests, requests, URL) for _ in range(2)]
        try:
            assert saturated.wait(10), "Requests did not run concurrently"
            with lock:
                assert active == 128
                assert attempted == 128
            assert all(not batch.done() for batch in batches)
        finally:
            release.set()
        for batch in batches:
            metrics, probabilities = batch.result(timeout=10)
            assert metrics["decision_model/successful_response_count"] == 129
            assert metrics["decision_model/http_background_threads"] == 1
            assert probabilities == {key: {f"step_{key}": answer()["probabilities"]} for key in requests}
    assert attempted == 258
    assert peak == 128
    assert len(worker_ids) == 1


def test_empty_requests_do_not_call_api(monkeypatch):
    opener = MagicMock()
    monkeypatch.setattr("agent_system.decision_model.urlopen", opener)
    metrics, probabilities = submit_decision_model_requests({"a": payload()}, URL)
    assert probabilities == {}
    opener.assert_not_called()
    assert metrics["decision_model/post_request_count"] == 0
    assert metrics["decision_model/skipped_empty_request_count"] == 1


@pytest.mark.parametrize("limit", [0, 129, True, 1.5])
def test_rejects_invalid_concurrency(limit):
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        DecisionModelRequestBatch(URL, max_concurrent_requests=limit)


def test_async_queue_respects_configured_limit_and_excludes_queue_from_timeout(monkeypatch):
    entered, release = Event(), Event()
    active = peak = 0

    @asynccontextmanager
    async def post(session, url, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        try:
            while not release.is_set():
                await asyncio.sleep(0.001)
            async def read():
                return json.dumps({"answers": {"step_1": answer()}}).encode()
            yield SimpleNamespace(status=200, read=read)
        finally:
            active -= 1

    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    with DecisionModelRequestBatch(URL, max_concurrent_requests=2) as pending:
        for i in range(10):
            pending.submit(str(i), payload("step_1"))
        try:
            assert entered.wait(5)
            # An async barrier lets all queued tasks register without waiting on the HTTP requests.
            async def barrier():
                await asyncio.sleep(0.02)
                return pending.active, pending.queued
            assert pending.service.submit(barrier()).result(timeout=5) == (2, 8)
        finally:
            release.set()
        metrics, probabilities = pending.finish()
    assert len(probabilities) == 10
    assert peak == metrics["decision_model/in_flight_peak"] == 2
    assert metrics["decision_model/queued_request_peak"] == 8
    assert metrics["decision_model/concurrency_wait_max_s"] >= 0.02
    assert metrics["decision_model/timeout_count"] == 0


def test_cpu_and_dispatch_metrics_detect_a_busy_event_loop(monkeypatch):
    from time import perf_counter

    monkeypatch.setattr(decision_model, "urlopen", MagicMock(return_value=response({"answers": {"step_1": answer()}})))
    entered = Event()

    async def stall():
        entered.set()
        until = perf_counter() + 0.3
        while perf_counter() < until:
            pass  # Deliberately occupy the HTTP thread to exercise its diagnostics.

    with DecisionModelRequestBatch(URL) as pending:
        pending.service.submit(stall())
        assert entered.wait(5)
        pending.submit("a", payload("step_1"))
        pending.futures["a"].result(timeout=5)
        metrics, _ = pending.finish()
    assert metrics["decision_model/event_loop_lag_max_s"] > 0.1
    assert metrics["decision_model/dispatch_delay_max_s"] > 0.1
    assert metrics["decision_model/event_loop_cpu_fraction_max"] > 0.1
    assert metrics["decision_model/completed_before_collection_fraction"] == 1
    assert metrics["decision_model/in_flight_peak"] == 1


def test_context_exit_cancels_active_and_queued_requests(monkeypatch):
    entered, cancelled = Event(), Event()

    @asynccontextmanager
    async def post(session, url, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            cancelled.set()

    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    with pytest.raises(RuntimeError, match="rollout failed"):
        with DecisionModelRequestBatch(URL, max_concurrent_requests=1) as pending:
            for i in range(4):
                pending.submit(str(i), payload("step_1"))
            assert entered.wait(5)
            raise RuntimeError("rollout failed")
    assert cancelled.is_set()
    assert pending.session.closed
    assert not pending.tasks
    assert all(future.cancelled() for future in pending.futures.values())


def test_discarded_turn_failure_also_aborts(monkeypatch):
    opener = MagicMock(side_effect=[TimeoutError("discarded"), response({"answers": {"step_1": answer()}})])
    monkeypatch.setattr(decision_model, "urlopen", opener)
    with DecisionModelRequestBatch(URL) as pending:
        pending.submit("discarded", payload("step_1"))
        pending.submit("retained", payload("step_1"))
        pending.submit("retained", payload("step_1"))  # Duplicate IDs are never resubmitted.
        with pytest.raises(DecisionModelUnavailable, match="discarded") as caught:
            pending.finish(retained_ids={"retained"})
        metrics = caught.value.metrics
    assert opener.call_count == 2
    assert metrics["decision_model/discarded_request_count"] == 1
    assert metrics["decision_model/timeout_count"] == 1


def test_real_async_http_handles_queueing_timeouts_statuses_and_invalid_json(monkeypatch):
    from aiohttp import web

    monkeypatch.setattr(aiohttp.ClientSession, "post", REAL_POST)
    service = decision_model._get_service()
    seen_threads = set()

    async def handle(request):
        seen_threads.add(get_ident())
        assert request.headers["Authorization"] == "Bearer test-token"
        body = await request.json()
        kind = body["state"]
        if kind == "http-error":
            return web.Response(status=504)
        if kind == "invalid":
            return web.Response(body=b"bad json")
        await asyncio.sleep(0.3 if kind == "timeout" else 0.04)
        return web.json_response({"answers": {qid: answer() for qid in body["questions"]}})

    async def start():
        app = web.Application()
        app.router.add_post("/v1/systemone", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return runner, site._server.sockets[0].getsockname()[1]

    runner, port = service.submit(start()).result(timeout=5)
    try:
        # localhost exercises caller-side DNS resolution without a DNS thread pool.
        with DecisionModelRequestBatch(f"http://localhost:{port}/v1/systemone", timeout=0.15,
                                       api_key="test-token", max_concurrent_requests=2) as pending:
            for i in range(10):
                pending.submit(str(i), payload("step_1"))
            for kind in ["http-error", "invalid", "timeout"]:
                pending.submit(kind, {**payload("step_1"), "state": kind})
            with pytest.raises(DecisionModelUnavailable) as caught:
                pending.finish()
        metrics = caught.value.metrics
        assert metrics["decision_model/successful_response_count"] == 10
        assert metrics["decision_model/timeout_count"] == 2
        assert metrics["decision_model/invalid_response_count"] == 1
        assert metrics["decision_model/failed_request_count"] == 3
        assert metrics["decision_model/concurrency_wait_max_s"] > 0.15
        assert metrics["decision_model/in_flight_peak"] == 2
        assert seen_threads == {service.thread.ident}
    finally:
        service.submit(runner.cleanup()).result(timeout=5)
