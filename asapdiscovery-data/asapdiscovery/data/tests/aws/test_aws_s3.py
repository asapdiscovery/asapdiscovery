import os
import uuid
from asapdiscovery.data.services_config import S3Settings
from asapdiscovery.data.aws.s3 import S3


class TestS3:
    def test_push_file(self, s3_fresh, tmpdir):
        s3 = s3_fresh
        filepath = "testfile.txt"

        with tmpdir.as_cwd():
            with open(filepath, "w") as f:
                f.write("Welcome to the *#&#@*^ enrichment center")

            dest_location = f"{str(uuid.uuid4())}.txt"

            s3.push_file(filepath, dest_location)

            # examine object metadata
            objs = list(s3.resource.Bucket(s3.bucket).objects.all())

            assert len(objs) == 1
            assert objs[0].key == os.path.join(s3.prefix, dest_location)

    def test_from_settings(self):
        s3_settings = S3Settings()  # will read from env vars set in conftest.py
        S3.from_settings(s3_settings)
