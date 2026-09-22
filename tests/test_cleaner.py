from src.data.cleaner import contains_forbidden_elements


def test_forbidden_elements_detected():
    """Проверка: уран, актиний и ртуть должны быть заблокированы."""
    assert contains_forbidden_elements("UO2") is True
    assert contains_forbidden_elements("Ac2O3") is True
    assert contains_forbidden_elements("HgTe") is True


def test_safe_semiconductors_allowed():
    """Проверка: безопасные полупроводники (Si, SiC, GaN, Diamond) должны проходить."""
    assert contains_forbidden_elements("Si") is False
    assert contains_forbidden_elements("SiC") is False
    assert contains_forbidden_elements("GaN") is False
    assert contains_forbidden_elements("C") is False