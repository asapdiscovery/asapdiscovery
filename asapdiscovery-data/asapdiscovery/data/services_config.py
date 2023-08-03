from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, validator, Field


class PosteraSettings(BaseSettings):
    POSTERA_API_KEY: str
    POSTERA_API_URL: str = "https://api.asap.postera.ai"
    POSTERA_API_VERSION: str = "v1"


class S3Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = Field(description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(description="AWS secret access key")
    BUCKET_NAME: str = Field(description="S3 bucket name")
    BUCKET_PREFIX: Optional[str] = Field(None, description="The prefix to use for referencing objects in the bucket")


class CloudfrontSettings(BaseSettings):
    CLOUDFRONT_DOMAIN: str = Field(description="Cloudfront domain name")
    CLOUDFRONT_KEY_ID: str = Field(description="Cloudfront public key ID")
    CLOUDFRONT_PRIVATE_KEY_PEM: str = Field(description="Path to Cloudfront private key PEM file")

    # validate cloudfront_private_key_pem exists
    @validator("CLOUDFRONT_PRIVATE_KEY_PEM")
    def validate_cloudfront_private_key_pem_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Cloudfront private key file does not exist: {v}")
        return v
