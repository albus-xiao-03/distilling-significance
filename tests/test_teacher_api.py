"""Teacher API wrapper with a fake client: filter, chunks, cache, retries, throttle."""

import httpx
import pytest
from behaviorgpt import Order, View

from fakes import FakeClient, http_error
from significance.teacher import api

HIST = [[("view", "a"), ("view", "b"), ("order", ("a", "b"))], [("view", "c")]]


OPEN_CACHES = []


@pytest.fixture(autouse=True)
def _close_caches():
    yield
    while OPEN_CACHES:
        OPEN_CACHES.pop().close()


def _teacher(client, tmp_path=None, **kw):
    cache = api.ResponseCache(tmp_path / "cache.sqlite") if tmp_path else None
    if cache is not None:
        OPEN_CACHES.append(cache)
    return api.Teacher(client, "cat_test", cache=cache, sleep=lambda s: None, **kw)


def test_response_cache_is_a_context_manager(tmp_path):
    with api.ResponseCache(tmp_path / "c.sqlite") as cache:
        cache.put("k", {"a": 1.0})
        assert cache.get("k") == {"a": 1.0} and "k" in cache and len(cache) == 1
    with api.ResponseCache(tmp_path / "c.sqlite") as again:
        assert again.get("k") == {"a": 1.0}


def test_to_sdk_and_history_key():
    sdk = api.to_sdk([*HIST, []])
    assert len(sdk) == 2
    assert isinstance(sdk[0][0], View) and sdk[0][0].item == "a"
    assert isinstance(sdk[0][2], Order) and [c.item for c in sdk[0][2].items] == ["a", "b"]
    assert api.history_key([*HIST, []]) == [
        [["view", "a"], ["view", "b"], ["order", ["a", "b"]]],
        [["view", "c"]],
    ]
    with pytest.raises(ValueError):
        api.to_sdk([[("click", "a")]])


def test_score_uses_the_id_filter_in_chunks_of_100():
    client = FakeClient()
    teacher = _teacher(client)
    cands = [f"N{i}" for i in range(250)]
    scores = teacher.score(HIST, cands)
    assert set(scores) == set(cands)
    assert [len(c["ids"]) for c in client.calls] == [100, 100, 50]
    assert all(c["catalog_id"] == "cat_test" and c["predict"] == "view" for c in client.calls)
    assert teacher.stats.api_calls == 3


def test_cache_hits_make_no_calls_and_key_ignores_id_order(tmp_path):
    client = FakeClient()
    teacher = _teacher(client, tmp_path)
    first = teacher.score(HIST, ["x", "y", "z"])
    again = _teacher(client, tmp_path).score(HIST, ["z", "y", "x"])
    assert first == again and len(client.calls) == 1
    other = teacher.score([[("view", "c")]], ["x", "y", "z"])  # different history -> new call
    assert len(client.calls) == 2 and other != first
    assert teacher.cache_key(HIST, ["x"], "view") != teacher.cache_key(HIST, ["x"], "purchase")


def test_retries_on_rate_limit_server_errors_and_transport():
    client = FakeClient(fail=[http_error(429), http_error(503), httpx.ConnectError("down")])
    teacher = _teacher(client)
    assert set(teacher.score(HIST, ["x"])) == {"x"}
    assert teacher.stats.retries == 3 and teacher.stats.api_calls == 1


def test_no_retry_on_bad_request_and_gives_up_after_retries():
    with pytest.raises(api.UnboxAIError):
        _teacher(FakeClient(fail=[http_error(400)])).score(HIST, ["x"])
    with pytest.raises(api.UnboxAIError):
        _teacher(FakeClient(fail=[http_error(429)] * 3), retries=2).score(HIST, ["x"])


def test_missing_ids_are_counted_and_absent():
    teacher = _teacher(FakeClient(drop={"y"}))
    scores = teacher.score(HIST, ["x", "y"])
    assert set(scores) == {"x"} and teacher.stats.missing_ids == 1


def test_offline_teacher_uses_cache_only(tmp_path):
    _teacher(FakeClient(), tmp_path).score(HIST, ["x"])
    offline = _teacher(FakeClient(), tmp_path, offline=True)
    assert set(offline.score(HIST, ["x"])) == {"x"}
    with pytest.raises(LookupError):
        offline.score(HIST, ["new"])


def test_validation_of_inputs():
    teacher = _teacher(FakeClient())
    with pytest.raises(ValueError):
        teacher.score(HIST, ["x", "x"])
    with pytest.raises(ValueError):
        teacher.score(HIST, ["x"], predict="buy")
    with pytest.raises(ValueError):
        api.Teacher(FakeClient(), "c", max_ids=101)


def test_throttle_spaces_call_starts():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    throttle = api.Throttle(4.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(3):
        throttle.wait()
    assert slept == [0.25, 0.25]
    api.Throttle(None).wait()  # no limit, no sleep
    with pytest.raises(ValueError):
        api.Throttle(0)


def test_cache_key_changes_with_the_catalog_upload():
    a = api.Teacher(FakeClient(), "cat", catalog_version="job-1")
    b = api.Teacher(FakeClient(), "cat", catalog_version="job-2")
    same = api.Teacher(FakeClient(), "cat", catalog_version="job-1")
    assert a.cache_key(HIST, ["x"], "view") != b.cache_key(HIST, ["x"], "view")
    assert a.cache_key(HIST, ["x"], "view") == same.cache_key(HIST, ["x"], "view")
    assert a.cache_key(HIST, ["x"], "view") != a.cache_key(HIST, ["x"], "view").upper()
