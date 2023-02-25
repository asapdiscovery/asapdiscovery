import os
import uuid

import pytest
from asapdiscovery.data.aws.s3 import S3
from boto3.session import Session
from moto import mock_s3


class TestS3:
    @pytest.fixture
    def s3(self):
        with mock_s3():
            session = Session(
                aws_access_key_id="test-access-key-id",
                aws_secret_access_key="test-secret-access-key",
                aws_session_token="test-session-token",
                region_name="us-east-1",
            )

            s3 = S3(session, "test-bucket", "test-prefix")
            s3.initialize()
            yield s3

    def test_push_file(self, s3, tmpdir):
        with tmpdir.as_cwd():
            with open("testfile.txt", "w") as f:
                f.write("Welcome to the *#&#@*^ enrichment center")

            dest_location = f"{str(uuid.uuid4())}.txt"

            s3.push_file("testfile.txt", dest_location)

            # examine object metadata
            objs = list(s3.resource.Bucket(s3.bucket).objects.all())

            assert len(objs) == 1
            assert objs[0].key == os.path.join(s3.prefix, dest_location)
