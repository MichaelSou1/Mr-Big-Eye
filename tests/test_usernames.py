from app.usernames import USERNAME_POOL, pick_username


def test_pick_username_from_available_pool():
    name = pick_username(set(USERNAME_POOL[1:]))

    assert name == USERNAME_POOL[0]


def test_pick_username_appends_suffix_when_pool_taken():
    taken = set(USERNAME_POOL)
    name = pick_username(taken)

    assert name.endswith("_2")
