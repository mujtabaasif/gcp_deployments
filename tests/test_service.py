REC_ID = "Single_JKHY/2009/page_28.pdf-3"

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_record_ok(client):
    r = client.get(f"/records/{REC_ID}")
    assert r.status_code == 200
    body = r.json()
    assert body["record_id"] == REC_ID
    assert "document_pre_text" in body
    assert "table" in body
    assert "post_text" in body


def test_get_record_not_found(client):
    r = client.get("/records/does-not-exist")
    assert r.status_code == 404


def test_full_dialogue_flow(client):
    # Create session
    r = client.post("/sessions", json={"record_id": REC_ID})
    assert r.status_code == 200
    session = r.json()
    sid = session["session_id"]

    # Q1
    r1 = client.post("/chat", json={"session_id": sid, "message": "what is the net cash from operating activities in 2009?"})
    assert r1.status_code == 200
    assert r1.json()["answer"] == "206588"

    # Q2
    r2 = client.post("/chat", json={"session_id": sid, "message": "what about in 2008?"})
    assert r2.status_code == 200
    assert r2.json()["answer"] == "181001"

    # Q3
    r3 = client.post("/chat", json={"session_id": sid, "message": "what is the difference?"})
    assert r3.status_code == 200
    assert r3.json()["answer"] == "25587"

    # Q4
    r4 = client.post("/chat", json={"session_id": sid, "message": "what percentage change does this represent?"})
    assert r4.status_code == 200
    assert r4.json()["answer"] == "14.1%"

    # Cleanup
    rdel = client.delete(f"/sessions/{sid}")
    assert rdel.status_code == 200
    assert rdel.json().get("deleted") is True
