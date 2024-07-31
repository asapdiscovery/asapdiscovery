"""Interface for object generation on S3.

"""

import os
from os import PathLike
from typing import Optional

import boto3


class S3Error(Exception): ...  # noqa: E701


class S3:
    """Interface for AWS S3."""

    def __init__(
        self,
        session: boto3.Session,
        bucket: str,
        prefix: Optional[str] = None,
        endpoint_url=None,
    ):
        """Create an interface to AWS S3.

        Parameters
        ----------
        session
            A `boto3.Session` object, already parameterized with credentials,
            region, etc.
        bucket
            The name of the S3 bucket to target.
        prefix
            The prefix to use for referencing objects in the bucket;
            functions as a working sub-folder/directory.
        endpoint_url
            The S3 endpoint to use; used for testing, generally not needed.

        """
        self.session = session
        self.resource = self.session.resource("s3", endpoint_url=endpoint_url)

        self.bucket = bucket
        self.prefix = prefix.strip("/") if prefix is not None else ""

    @classmethod
    def from_settings(cls, settings):
        """Create an interface to AWS S3 from a `Settings` object.

        Parameters
        ----------
        settings
            A `S3Settings` object.
        prefix
            The prefix to use for referencing objects in the bucket;
            functions as a working sub-folder/directory.

        Returns
        -------
        S3
            S3 interface object.
        """
        session = boto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        return cls(session, settings.BUCKET_NAME, prefix=settings.BUCKET_PREFIX)

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

    def push_file(
        self, path: PathLike, location: PathLike = None, content_type: str = None
    ):
        """Push a file at the local filesystem `path` to an object `location`
        in this S3 Bucket.

        `location` is relative to the `prefix` set for use of this bucket; e.g.
        if ``location='foo/bar.html'`` and ``self.prefix == 'baz'``, then the object will
        be located at ``baz/foo/bar.html`` in the bucket.

        Parameters
        ----------
        path
            Path to file on local filesystem to push.
        location
            Location in the S3 bucket to place object relative to ``self.prefix``;
            should not contain a leading ``/``.
        content_type
            Media type of the file being pushed. This will impact how the file
            is handled by a browser upon URL access, e.g. for ``html`` you want
            rendered on access, use ``'text/html'``.
        """
        if content_type is None:
            extra_args = {}
        else:
            extra_args = {"ContentType": content_type}

        if location is None:
            location = os.path.basename(path)

        key = os.path.join(self.prefix, location)

        self.resource.Bucket(self.bucket).upload_file(path, key, ExtraArgs=extra_args)

    def push_dir(self, path: PathLike, location: PathLike = None):
        """Push a directory at the local filesystem `path` to an object `location`
        in this S3 Bucket.

        `location` is relative to the `prefix` set for use of this bucket; e.g.
        if ``location='foo'`` and ``self.prefix == 'baz'``, then the object will
        be located at ``baz/foo`` in the bucket.

        Parameters
        ----------
        path
            Path to directory on local filesystem to push.
        location
            Location in the S3 bucket to place object relative to ``self.prefix``;
            should not contain a leading ``/``.
        """
        if location is None:
            location = os.path.basename(path)

        for root, _, files in os.walk(path):
            for file in files:
                local_path = os.path.join(root, file)
                remote_path = os.path.relpath(local_path, path)
                self.push_file(local_path, os.path.join(location, remote_path))

    def pull_file(self): ...

    def to_uri(self, location: PathLike):
        """Convert a location in the S3 bucket to a URI.

        Parameters
        ----------
        location
            Location in the S3 bucket to convert to a URI.

        Returns
        -------
        str
            URI for the object in the S3 bucket.

        """
        return f"s3://{self.bucket}/{os.path.join(self.prefix, location)}"
