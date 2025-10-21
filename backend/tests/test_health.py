def test_health_ok(client):
    resp = client.get('/api/v1/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body.get('status') in {'ok', 'healthy', 'UP'}


def test_health_options_cors(client):
    resp = client.options('/api/v1/health')
    # CORS preflight should not 404
    assert resp.status_code in {200, 204}
    # Access-Control headers may be present depending on config
    # Just ensure server handles OPTIONS
