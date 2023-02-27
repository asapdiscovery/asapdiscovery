"""Interface for object generation on S3.

"""

import os
from typing import Optional


class S3Error(Exception):
    ...


class S3:
    def __init__(
        self,
        session: "boto3.Session",
        bucket: str,
        prefix: Optional[str] = None,
        endpoint_url=None,
    ):
        """ """
        self.session = session
        self.resource = self.session.resource("s3", endpoint_url=endpoint_url)

        self.bucket = bucket
        self.prefix = prefix if prefix is not None else ""

    def initialize(self):
        """Initialize bucket.

        Creates bucket if it does not exist.

        """
        bucket = self.resource.Bucket(self.bucket)
        bucket.create()
        bucket.wait_until_exists()

    def reset(self):
        """Delete all objects, including bucket itself.

        Inverse operation of `initialize`.

        """
        bucket = self.resource.Bucket(self.bucket)

        # delete all objects, then the bucket
        bucket.objects.delete()
        bucket.delete()
        bucket.wait_until_not_exists()

    def push_file(self, path, location=None, content_type=None):

        if content_type is None:
            extra_args = {}
        else:
            extra_args = {"ContentType": content_type}

        if location is None:
            location = os.path.basename(path)

        key = os.path.join(self.prefix, location)

        self.resource.Bucket(self.bucket).upload_file(path, key, ExtraArgs=extra_args)

    def pull_file(self):
        ...
