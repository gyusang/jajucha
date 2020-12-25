def get_jajucha():
    try:
        import jajucha
        return True
    except:
        return False


def test_get_jajucha():
    assert get_jajucha()
