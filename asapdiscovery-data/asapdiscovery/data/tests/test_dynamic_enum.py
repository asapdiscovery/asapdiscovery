import pytest
from asapdiscovery.data.dynamic_enum import make_dynamic_enum


@pytest.fixture()
def dynamic_enum_yaml(tmp_path):
    """Fixture to create a dynamic enum yaml file."""
    yaml = tmp_path / "test.yaml"
    yaml.write_text(
        """
        MEMBER1: 1
        MEMBER2: 2
        """
    )
    return yaml


def test_make_dynamic_enum(dynamic_enum_yaml):
    """Test make_dynamic_enum."""
    enum = make_dynamic_enum(dynamic_enum_yaml, "TestEnum")
    assert enum.MEMBER1.value == 1
    assert enum.MEMBER2.value == 2
