from app.tools import merge_subject_deltas, parse_subject_deltas


def test_merge_subject_deltas_adds_subject():
    registry = merge_subject_deltas(
        [],
        [
            {
                "op": "add",
                "id": "person_A",
                "label": "红衣男子",
                "first_seen_t": 12.3,
                "attributes": ["持有背包"],
                "evidence_frames": [12.3],
            }
        ],
    )

    assert registry == [
        {
            "id": "person_A",
            "label": "红衣男子",
            "first_seen_t": 12.3,
            "last_seen_t": 12.3,
            "attributes": ["持有背包"],
            "evidence_frames": [12.3],
        }
    ]


def test_merge_subject_deltas_updates_subject():
    registry = [
        {
            "id": "person_B",
            "label": "蓝衣女子",
            "first_seen_t": 10.0,
            "last_seen_t": 20.0,
            "attributes": ["站立"],
            "evidence_frames": [10.0],
        }
    ]

    merged = merge_subject_deltas(
        registry,
        [
            {
                "op": "update",
                "id": "person_B",
                "attributes_add": ["在跑步", "站立"],
                "last_seen_t": 45.0,
                "evidence_frames_add": [45.0, 10.0],
            }
        ],
    )

    assert merged[0]["attributes"] == ["在跑步", "站立"]
    assert merged[0]["evidence_frames"] == [10.0, 45.0]
    assert merged[0]["last_seen_t"] == 45.0


def test_merge_subject_deltas_prunes_to_recent_15():
    registry = [
        {
            "id": f"person_{i}",
            "label": f"person_{i}",
            "first_seen_t": float(i),
            "last_seen_t": float(i),
            "attributes": [],
            "evidence_frames": [],
        }
        for i in range(20)
    ]

    merged = merge_subject_deltas(registry, [])

    assert len(merged) == 15
    assert [item["id"] for item in merged[:3]] == ["person_19", "person_18", "person_17"]
    assert "person_0" not in {item["id"] for item in merged}


def test_parse_subject_deltas_strips_valid_trailing_line():
    answer, deltas = parse_subject_deltas(
        'A person runs. [FRAME:t=12.3]\n'
        'SUBJECT_DELTAS: {"deltas": [{"op": "add", "id": "person_A"}]}'
    )

    assert answer == "A person runs. [FRAME:t=12.3]"
    assert deltas == [{"op": "add", "id": "person_A"}]


def test_parse_subject_deltas_accepts_multiline_json_block():
    answer, deltas = parse_subject_deltas(
        "A person runs. [FRAME:t=12.3]\n"
        "SUBJECT_DELTAS: {\n"
        '  "deltas": [\n'
        '    {"op": "update", "id": "person_A"}\n'
        "  ]\n"
        "}"
    )

    assert answer == "A person runs. [FRAME:t=12.3]"
    assert deltas == [{"op": "update", "id": "person_A"}]


def test_parse_subject_deltas_keeps_answer_on_garbage_line():
    raw = "A person runs.\nSUBJECT_DELTAS: not-json"

    answer, deltas = parse_subject_deltas(raw)

    assert answer == raw
    assert deltas == []
