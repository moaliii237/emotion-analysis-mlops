"""Edge case unit tests."""


class TestEmptyInputs:
    def test_predict_text_empty_no_500(self, client):
        r = client.post("/predict/text", json={"text": ""})
        assert r.status_code in (200, 422)

    def test_predict_text_missing_field_422(self, client):
        r = client.post("/predict/text", json={})
        assert r.status_code == 422

    def test_predict_text_null_text_422(self, client):
        r = client.post("/predict/text", json={"text": None})
        assert r.status_code == 422


class TestMalformedRequests:
    def test_wrong_type_for_text(self, client):
        r = client.post("/predict/text", json={"text": 42})
        assert r.status_code == 422

    def test_segments_as_string(self, client):
        r = client.post("/predict/timeline", json={"segments": "not a list"})
        assert r.status_code == 422


class TestBoundaryInputs:
    def test_single_character(self, client):
        r = client.post("/predict/text", json={"text": "a"})
        assert r.status_code in (200, 422)

    def test_very_long_text(self, client):
        r = client.post("/predict/text", json={"text": "I am furious. " * 200})
        assert r.status_code in (200, 422)

    def test_unicode(self, client):
        r = client.post("/predict/text", json={"text": "I'm happy 你好 résumé"})
        assert r.status_code == 200
