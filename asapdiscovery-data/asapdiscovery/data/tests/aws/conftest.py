import os

import pytest
from asapdiscovery.data.services.aws.s3 import S3
from boto3.session import Session
from moto.server import ThreadedMotoServer


@pytest.fixture(scope="module")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="module")
def s3(aws_credentials):
    server = ThreadedMotoServer()
    server.start()

    session = Session()

    s3 = S3(session, "test-bucket", "test-prefix", endpoint_url="http://127.0.0.1:5000")
    s3.initialize()

    yield s3

    server.stop()


@pytest.fixture
def s3_fresh(s3):
    s3.reset()
    s3.initialize()

    return s3
