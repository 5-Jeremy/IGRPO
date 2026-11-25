"""Decision-model HTTP requests, submitted during rollout and collected before learning."""

import asyncio
import atexit
import errno
import json
import logging
import math
import re
import socket
import threading
from collections import Counter
from concurrent.futures import Future
from http.client import HTTPException
from time import perf_counter, thread_time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import aiohttp

logger = logging.getLogger(__name__)


class DecisionModelUnavailable(RuntimeError):
    """An API call failed; metrics remain available to the trainer."""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def validate_decision_model_response(body, request):
    """Validate every returned choice and its probability distribution against the request."""
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise ValueError("Response must contain an answers object")
    answers = body["answers"]
    if set(answers) != set(request["questions"]):
        raise ValueError("Response question IDs do not match the request")
    for qid, answer in answers.items():
        choices = request["questions"][qid]["criteria"]
        if not isinstance(answer, dict) or answer.get("choice") not in choices:
            raise ValueError(f"Invalid response choice for {qid}")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(choices):
            raise ValueError(f"Response probability labels do not match the request for {qid}")
        if any(isinstance(p, bool) or not isinstance(p, (int, float))
               or not 0 <= p <= 1 or not math.isfinite(p) for p in probabilities.values()):
            raise ValueError(f"Invalid response probabilities for {qid}")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.03):
            raise ValueError(f"Response probabilities do not sum to one for {qid}")


class _ResolvedHost(aiohttp.abc.AbstractResolver):
    """Resolve once on the caller, avoiding an implicit DNS worker pool in asyncio."""

    def __init__(self, url):
        parsed = urlsplit(url)
        self.hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not self.hostname:
            raise ValueError("Decision-model request URL must be HTTP or HTTPS")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.error = None
        try:
            self.addresses = [
                {"hostname": self.hostname, "host": address[0], "port": port,
                 "family": family, "proto": proto, "flags": socket.AI_NUMERICHOST}
                for family, _, proto, _, address in socket.getaddrinfo(
                    self.hostname, port, type=socket.SOCK_STREAM)
            ]
        except OSError as exc:
            self.error = exc

    async def resolve(self, host, port=0, family=socket.AF_INET):
        if self.error is not None:
            raise self.error
        if host != self.hostname:
            raise OSError("Unexpected decision-model redirect host")
        return self.addresses

    async def close(self):
        pass


class _HTTPService:
    """One event-loop thread and one process-wide 128-request limit."""

    def __init__(self):
        self.ready = Future()
        self.thread = threading.Thread(target=self._run, name="decision-model-http", daemon=True)
        self.thread.start()
        self.ready.result()

    def _run(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.slots = asyncio.Semaphore(128)
        except BaseException as exc:
            self.ready.set_exception(exc)
            return
        self.ready.set_result(None)
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


_service = None
_service_lock = threading.Lock()


def _get_service():
    global _service
    with _service_lock:
        if _service is None:
            _service = _HTTPService()
        return _service


def _shutdown_service():
    global _service
    with _service_lock:
        if _service is not None:
            _service.close()
            _service = None


atexit.register(_shutdown_service)


class DecisionModelRequestBatch:
    """Caller-owned step state; all HTTP tasks share one background event loop.

    Use as a context manager so failed/interrupted rollouts cancel queued and active
    requests. HTTP failures are retained until finish(), never raised during submit().
    """

    def __init__(self, url, timeout=300.0, *, api_key=None, max_concurrent_requests=128):
        if (isinstance(max_concurrent_requests, bool) or not isinstance(max_concurrent_requests, int)
                or not 1 <= max_concurrent_requests <= 128):
            raise ValueError("max_concurrent_requests must be an integer from 1 to 128")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request_timeout must be positive and finite")
        self.url, self.timeout = url, timeout
        self.limit = max_concurrent_requests
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.requests, self.futures = {}, {}
        self.started = None
        self.closed = self.finished = False
        self.service = _get_service()
        # DNS runs once here, not on the event loop and not in extra background threads.
        resolver = _ResolvedHost(url)
        self.service.submit(self._open(resolver)).result()

    async def _open(self, resolver):
        self.slots = asyncio.Semaphore(self.limit)
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.limit, resolver=resolver),
            cookie_jar=aiohttp.DummyCookieJar(),
        )
        self.tasks = set()
        self.stats = Counter()
        self.active = self.queued = 0
        self.sample_at, self.sample_cpu = perf_counter(), thread_time()
        self.heartbeat = self.service.loop.call_later(0.1, self._sample)

    def _sample(self, reschedule=True):
        now, cpu = perf_counter(), thread_time()
        elapsed = now - self.sample_at
        self.stats["event_loop_cpu_s"] += cpu - self.sample_cpu
        self.stats["event_loop_sample_s"] += elapsed
        if elapsed >= 0.1:
            lag = max(0.0, elapsed - 0.1)
            self.stats["event_loop_lag_sum_s"] += lag
            self.stats["event_loop_lag_samples"] += 1
            self.stats["event_loop_lag_max_s"] = max(self.stats["event_loop_lag_max_s"], lag)
            self.stats["event_loop_cpu_fraction_max"] = max(
                self.stats["event_loop_cpu_fraction_max"], (cpu - self.sample_cpu) / elapsed)
        self.sample_at, self.sample_cpu = now, cpu
        if reschedule:
            self.heartbeat = self.service.loop.call_later(0.1, self._sample)

    def submit(self, turn_id, payload):
        if self.closed or self.finished:
            raise RuntimeError("Decision-model batch is already closed or collected")
        if turn_id in self.requests:
            if self.requests[turn_id] != payload:
                raise ValueError(f"Conflicting decision-model payloads for turn {turn_id}")
            return
        # Serialization on the producer keeps Python CPU work off the HTTP thread.
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.requests[turn_id] = json.loads(data)
        if not payload["questions"]:
            return
        queued_at = perf_counter()
        if self.started is None:
            self.started = queued_at
        self.futures[turn_id] = self.service.submit(self._post(data, queued_at))

    async def _post(self, data, queued_at):
        task = asyncio.current_task()
        self.tasks.add(task)
        entered = perf_counter()
        dispatch = entered - queued_at
        self.stats["dispatch_delay_sum_s"] += dispatch
        self.stats["dispatch_delay_max_s"] = max(self.stats["dispatch_delay_max_s"], dispatch)
        self.queued += 1
        if self.slots.locked() or self.service.slots.locked():
            self.stats["queued_request_peak"] = max(self.stats["queued_request_peak"], self.queued)
        acquired = False
        try:
            async with self.slots, self.service.slots:
                acquired = True
                self.queued -= 1
                wait = perf_counter() - entered
                self.stats["concurrency_wait_sum_s"] += wait
                self.stats["concurrency_wait_max_s"] = max(self.stats["concurrency_wait_max_s"], wait)
                self.active += 1
                self.stats["in_flight_peak"] = max(self.stats["in_flight_peak"], self.active)
                try:
                    # Like urllib's timeout, bound socket connection/read waits, not local queue time.
                    timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.timeout, sock_read=self.timeout)
                    async with self.session.post(self.url, data=data, headers=self.headers,
                                                 timeout=timeout, allow_redirects=False) as response:
                        if not 200 <= response.status < 300:
                            raise aiohttp.ClientResponseError(response.request_info, response.history,
                                                              status=response.status, message=response.reason)
                        raw = await response.read()
                    return raw, None, False
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                    timed_out = (isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or getattr(exc, "errno", None) == errno.ETIMEDOUT
                                 or getattr(exc, "status", None) in {408, 504})
                    return None, exc, timed_out
                finally:
                    self.active -= 1
        finally:
            if not acquired:
                self.queued -= 1
            self.tasks.discard(task)

    async def _diagnostics(self):
        self.heartbeat.cancel()
        self._sample(reschedule=False)
        stats = self.stats.copy()
        n = max(1, len(self.futures))
        return {
            "http_background_threads": 1,
            "max_concurrent_requests": self.limit,
            "in_flight_peak": stats["in_flight_peak"],
            "queued_request_peak": stats["queued_request_peak"],
            "dispatch_delay_mean_s": stats["dispatch_delay_sum_s"] / n,
            "dispatch_delay_max_s": stats["dispatch_delay_max_s"],
            "concurrency_wait_mean_s": stats["concurrency_wait_sum_s"] / n,
            "concurrency_wait_max_s": stats["concurrency_wait_max_s"],
            "event_loop_lag_mean_s": stats["event_loop_lag_sum_s"] / max(1, stats["event_loop_lag_samples"]),
            "event_loop_lag_max_s": stats["event_loop_lag_max_s"],
            "event_loop_cpu_fraction": stats["event_loop_cpu_s"] / max(1e-9, stats["event_loop_sample_s"]),
            "event_loop_cpu_fraction_max": stats["event_loop_cpu_fraction_max"],
        }

    def finish(self, retained_ids=None):
        if self.closed or self.finished:
            raise RuntimeError("Decision-model batch is already closed or collected")
        self.finished = True
        waiting = perf_counter()
        completed = sum(future.done() for future in self.futures.values())
        results = {turn_id: future.result() for turn_id, future in self.futures.items()}
        wait_time = perf_counter() - waiting
        diagnostics = self.service.submit(self._diagnostics()).result()
        diagnostics["collection_wait_s"] = wait_time
        diagnostics["completed_before_collection_fraction"] = completed / max(1, len(self.futures))
        return _collect_results(self.requests, results, self.url, retained_ids,
                                self.started, diagnostics)

    async def _close(self):
        self.heartbeat.cancel()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.session.close()

    def close(self):
        if not self.closed:
            # This coroutine is scheduled after all submissions, including ones not yet started.
            self.service.submit(self._close()).result()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _collect_results(requests, results, url, retained_ids, started, diagnostics):
    counts, choices, probabilities = Counter(), Counter(), Counter()
    returned_probabilities = {}
    retained = set(requests) if retained_ids is None else set(retained_ids)
    labels = {label for request in requests.values() for question in request["questions"].values()
              for label in question["criteria"]}
    total_questions = sum(len(request["questions"]) for request in requests.values())
    selected_probability_sum = 0.0
    first_failure = None
    for turn_id, payload in requests.items():
        if not payload["questions"]:
            counts["skipped_empty_request_count"] += 1
            continue
        counts["post_request_count"] += 1
        raw, failure, timed_out = results[turn_id]
        counts["timeout_count"] += int(timed_out)
        if failure is None:
            try:
                body = json.loads(raw)
                validate_decision_model_response(body, payload)
            except (ValueError, TypeError) as exc:
                failure = exc
                counts["invalid_response_count"] += 1
        if failure is not None:
            counts["failed_request_count"] += 1
            logger.warning("Decision model POST failed for turn %s: %s", turn_id, failure)
            if first_failure is None:
                first_failure = turn_id, failure
            continue
        counts["successful_response_count"] += 1
        counts["answer_count"] += len(body["answers"])
        if turn_id in retained:
            returned_probabilities[turn_id] = {qid: dict(answer["probabilities"]) for qid, answer in body["answers"].items()}
        for answer in body["answers"].values():
            choices[answer["choice"]] += 1
            selected_probability_sum += answer["probabilities"][answer["choice"]]
            probabilities.update(answer["probabilities"])
    metrics = {f"decision_model/{name}": counts[name] for name in (
        "post_request_count", "successful_response_count", "failed_request_count",
        "skipped_empty_request_count", "answer_count", "timeout_count", "invalid_response_count",
    )}
    metrics.update({
        "decision_model/request_count": len(requests),
        "decision_model/question_count": total_questions,
        "decision_model/discarded_request_count": len(set(requests) - retained),
        "decision_model/unanswered_question_count": total_questions - counts["answer_count"],
        "decision_model/request_success_rate": counts["successful_response_count"] / max(1, counts["post_request_count"]),
    })
    if counts["answer_count"]:
        metrics["decision_model/selected_probability_mean"] = selected_probability_sum / counts["answer_count"]
        for label in sorted(labels):
            key = re.sub(r"\s+", "_", label)
            metrics[f"decision_model/choice_count/{key}"] = choices[label]
            metrics[f"decision_model/choice_fraction/{key}"] = choices[label] / counts["answer_count"]
            metrics[f"decision_model/probability_mean/{key}"] = probabilities[label] / counts["answer_count"]
    metrics.update({f"decision_model/{key}": value for key, value in diagnostics.items()})
    metrics["decision_model/total_request_processing_time"] = perf_counter() - started if started is not None else 0.0
    if first_failure is not None:
        turn_id, failure = first_failure
        raise DecisionModelUnavailable(
            f"Decision model POST failed at {url} for turn {turn_id}: {failure}", metrics,
        ) from failure
    return metrics, returned_probabilities


def submit_decision_model_requests(requests, url, timeout=300.0, *, api_key=None,
                                   max_concurrent_requests=128):
    """Compatibility entry point for callers that already have a complete batch."""
    with DecisionModelRequestBatch(url, timeout, api_key=api_key, max_concurrent_requests=max_concurrent_requests) as batch:
        for turn_id, payload in requests.items():
            batch.submit(turn_id, payload)
        return batch.finish()


def check_decision_model(url, timeout=10.0, *, api_key=None, model=None):
    """Probe the API and optionally identify the model run from the same response."""
    started = perf_counter()
    status = 0  # No HTTP response received.
    alive = 0
    failure = None
    timed_out = False
    try:
        request = Request(url, method="GET")
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
            alive = int(200 <= status < 300)
    except HTTPError as exc:
        failure = exc
        status = exc.code
        timed_out = status in {408, 504}
        exc.close()
        logger.warning("Decision model probe failed: HTTP %s (%s)", status, url)
    except (URLError, OSError, HTTPException) as exc:
        failure = exc
        reason = exc.reason if isinstance(exc, URLError) else exc
        timed_out = isinstance(reason, TimeoutError) or getattr(reason, "errno", None) == errno.ETIMEDOUT
        logger.warning("Decision model probe failed (%s): %s", url, exc)
    metrics = {
        "decision_model/alive": alive,
        "decision_model/http_status": status,
        "decision_model/response_time_s": perf_counter() - started,
        "decision_model/timed_out": int(timed_out),
    }
    if not alive:
        raise DecisionModelUnavailable(f"Decision model unavailable at {url}: {failure or status}", metrics) from failure
    if model is not None:
        # Some health endpoints only return a status. The configured model is
        # still known when the service does not advertise a run identifier.
        run_name = model
        try:
            cards = json.loads(body)["models"]
            run = next(card["run"] for card in cards if card.get("name") == model)
            if isinstance(run, str) and run.strip():
                run_name = run
        except (ValueError, TypeError, KeyError, AttributeError, StopIteration):
            pass
        metrics["decision_model/model_name"] = run_name
    return metrics
