from datetime import datetime, timezone

from infra.metrics import CloudWatchMeter


class FakeCloudWatch:
    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.calls: list[dict] = []

    def get_metric_data(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        query_id = kwargs["MetricDataQueries"][0]["Id"]
        return {"MetricDataResults": self.results.get(query_id, [{"Values": []}])}


def make_meter(client, **kwargs):
    now = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    return CloudWatchMeter(client, now_fn=lambda: now, **kwargs)


class TestCloudWatchMeter:
    def test_reads_response_p90_and_cost(self):
        client = FakeCloudWatch(
            results={
                "response_p90": [{"Values": [0.4, 0.8]}],
                "cost_monthly": [{"Values": [12.5]}],
            }
        )
        metrics = make_meter(client).read()
        assert metrics == {"response_time_p90": 0.8, "cost_monthly": 12.5}

    def test_cost_takes_maximum_datapoint(self):
        client = FakeCloudWatch(results={"response_p90": [{"Values": [0.9]}], "cost_monthly": [{"Values": [10.0, 42.5]}]})
        metrics = make_meter(client).read()
        assert metrics["cost_monthly"] == 42.5

    def test_missing_datapoints_are_none(self):
        metrics = make_meter(FakeCloudWatch()).read()
        assert metrics == {"response_time_p90": None, "cost_monthly": None}

    def test_client_error_degrades_to_none(self):
        metrics = make_meter(FakeCloudWatch(error=RuntimeError("cw down"))).read()
        assert metrics == {"response_time_p90": None, "cost_monthly": None}

    def test_non_numeric_value_is_none(self):
        client = FakeCloudWatch(results={"response_p90": [{"Values": ["alto"]}], "cost_monthly": [{"Values": [3.0]}]})
        metrics = make_meter(client).read()
        assert metrics["response_time_p90"] is None
        assert metrics["cost_monthly"] == 3.0

    def test_metric_names_configurable(self):
        client = FakeCloudWatch(results={"response_p90": [{"Values": [1.0]}], "cost_monthly": [{"Values": [2.0]}]})
        meter = make_meter(client, namespace="Ns", response_metric="Resp", cost_metric="Cost")
        meter.read()
        metric = client.calls[0]["MetricDataQueries"][0]["MetricStat"]["Metric"]
        assert metric == {"Namespace": "Ns", "MetricName": "Resp"}
