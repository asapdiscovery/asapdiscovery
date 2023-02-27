import pytest
from asapdiscovery.data.aws.s3 import S3
from boto3.session import Session
from moto.server import ThreadedMotoServer


@pytest.fixture
def s3():
    server = ThreadedMotoServer()
    server.start()

    session = Session(
        #    aws_access_key_id="test-access-key-id",
        #    aws_secret_access_key="test-secret-access-key",
        #    aws_session_token="test-session-token",
        #    region_name="us-east-1",
    )

    s3 = S3(session, "test-bucket", "test-prefix", endpoint_url="http://127.0.0.1:5000")
    s3.initialize()
    yield s3
    s3.reset()
    server.stop()
