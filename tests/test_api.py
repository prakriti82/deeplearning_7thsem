"""API tests (M9). Synthetic fixtures are used for testing the code only."""
import pytest
from fastapi.testclient import TestClient

import api.main as main


@pytest.fixture
def client(fixture_df, monkeypatch):
    monkeypatch.setattr(main, "_load_dataset", lambda: fixture_df)
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_naive_forecast_returns_24_hours(client, fixture_df):
    day = str((fixture_df.index[24 * 30]).date())
    r = client.get("/forecast", params={"date": day, "model": "seasonal_naive_24"})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_hours"] == 24 and len(body["forecast"]) == 24
    assert body["forecast"][0]["timestamp"].startswith(day)


def test_unknown_model_is_rejected(client):
    assert client.get("/forecast", params={"model": "nope"}).status_code == 400


def test_missing_trained_model_gives_404(client, fixture_df):
    day = str((fixture_df.index[24 * 30]).date())
    r = client.get("/forecast", params={"date": day, "model": "lstm", "variant": "weather"})
    assert r.status_code == 404
