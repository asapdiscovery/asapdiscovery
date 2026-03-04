import os
from unittest import mock

import pytest
from asapdiscovery.data.services.services_config import (
    CloudfrontSettings,
    PosteraSettings,
    S3Settings,
)


@pytest.fixture(autouse=True)
def mock_postera_env_vars():
    with mock.patch.dict(
        os.environ,
        {
            "POSTERA_API_KEY": "mock_key",
            "POSTERA_API_URL": "mock_url",
            "POSTERA_API_VERSION": "mock_version",
        },
    ):
        yield


def test_postera_settings(mock_postera_env_vars):
    ps = PosteraSettings()
    assert ps.POSTERA_API_KEY == "mock_key"
    assert ps.POSTERA_API_URL == "mock_url"
    assert ps.POSTERA_API_VERSION == "mock_version"


@pytest.fixture(autouse=True)
def mock_s3_env_vars():
    with mock.patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "mock_id",
            "AWS_SECRET_ACCESS_KEY": "mock_key",
            "BUCKET_NAME": "mock_bucket",
            "BUCKET_PREFIX": "mock_prefix",
        },
    ):
        yield


def test_s3_settings(mock_s3_env_vars):
    s3 = S3Settings()
    assert s3.AWS_ACCESS_KEY_ID == "mock_id"
    assert s3.AWS_SECRET_ACCESS_KEY == "mock_key"
    assert s3.BUCKET_NAME == "mock_bucket"
    assert s3.BUCKET_PREFIX == "mock_prefix"


@pytest.fixture(autouse=True)
def mock_cloudfront_env_vars():
    with mock.patch.dict(
        os.environ,
        {
            "CLOUDFRONT_DOMAIN": "mock_domain",
            "CLOUDFRONT_KEY_ID": "mock_id",
            "CLOUDFRONT_PRIVATE_KEY_PEM": "mock_pem",
        },
    ):
        yield


@pytest.fixture()
def mock_cloudfront_pem_file_exists():
    with mock.patch(
        "asapdiscovery.data.services.services_config.Path.exists", return_value=True
    ):
        yield


def test_cloudfront_settings(mock_cloudfront_env_vars, mock_cloudfront_pem_file_exists):
    cf = CloudfrontSettings()
    assert cf.CLOUDFRONT_DOMAIN == "mock_domain"
    assert cf.CLOUDFRONT_KEY_ID == "mock_id"
    assert cf.CLOUDFRONT_PRIVATE_KEY_PEM == "mock_pem"


def test_cloudfront_settings_pem_file_validator(mock_cloudfront_env_vars):
    with pytest.raises(ValueError, match="Cloudfront private key file"):
        _ = CloudfrontSettings()
