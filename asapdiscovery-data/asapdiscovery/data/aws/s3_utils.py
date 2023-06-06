import boto3
import os
import random

from typing import Iterable, Union
from pathlib import Path

from .s3 import S3


def makeS3URL(s3_bucket_prefix: str, bucket_name: str):
    """For a given s3 bucket name, returns a randomly generated URL and its corresponding bucket path

    Parameters
    ----------
    s3_bucket_prefix : str
        The prefix to use for referencing objects in the bucket
    bucket_name : str
        The name of the S3 bucket to use.
    """

    # create the S3 path for the content to be stored in. Use a random string instead of
    # the content name to harden security.

    random_string = "".join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits)
        for _ in range(20)
    )

    s3_bucket_path = f"{s3_bucket_prefix}/{random_string}.html"
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_bucket_path}"

    return s3_url, s3_bucket_path


def create_S3_session_with_token_from_envvars(bucket_name: str):
    """Creates an S3 session with a given bucket name

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket to use.
    """

    if not bucket_name:
        raise ValueError("No bucket name provided.")

    if not os.getenv("AWS_ACCESS_KEY_ID"):
        raise ValueError("No AWS_ACCESS_KEY_ID provided.")
    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        raise ValueError("No AWS_SECRET_ACCESS_KEY provided.")
    if not os.getenv("AWS_SESSION_TOKEN"):
        raise ValueError("No AWS_SESSION_TOKEN provided.")

    session = boto3.session.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
    )

    s3 = S3(session, bucket=bucket_name)
    return s3


def upload_artifacts_from_content(
    s3_instance: S3,
    content_paths: Iterable[Union[str, Path]],
    s3_bucket_prefix: str,
    content_type: str = "text/html",
):
    if not content_type in ["text/html", "image/gif"]:
        raise ValueError(
            f"content_type must be one of ['text/html', 'image/gif'], not {content_type}"
        )
    urls = []
    for content_path in content_paths:
        url, bucket_path = makeS3URL(s3_bucket_prefix)
        s3_instance.push_file(
            content_path, location=bucket_path, content_type=content_type
        )
        urls.append(url)
    return urls
