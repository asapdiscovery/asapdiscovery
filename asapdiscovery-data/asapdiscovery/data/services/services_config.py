import logging
from pathlib import Path
from typing import Optional

from pydantic.v1 import BaseSettings, Field, validator

logger = logging.getLogger(__name__)


class PosteraSettings(BaseSettings):
    POSTERA_API_KEY: str
    POSTERA_API_URL: str = "https://api.asap.postera.ai"
    POSTERA_API_VERSION: str = "v1"


class S3Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = Field(description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(description="AWS secret access key")
    BUCKET_NAME: str = Field(description="S3 bucket name")
    BUCKET_PREFIX: Optional[str] = Field(
        None, description="The prefix to use for referencing objects in the bucket"
    )


class CloudfrontSettings(BaseSettings):
    CLOUDFRONT_DOMAIN: str = Field(description="Cloudfront domain name")
    CLOUDFRONT_KEY_ID: str = Field(description="Cloudfront public key ID")
    CLOUDFRONT_PRIVATE_KEY_PEM: str = Field(
        description="Path to Cloudfront private key PEM file"
    )

    # validate cloudfront_private_key_pem exists
    @validator("CLOUDFRONT_PRIVATE_KEY_PEM")
    def validate_cloudfront_private_key_pem_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Cloudfront private key file does not exist: {v}")
        return v


class CDDSettings(BaseSettings):
    CDD_API_KEY: str = Field(
        description="The CDD API key with access to the specified vault."
    )
    CDD_VAULT_NUMBER: int = Field(
        description="The id of the CDD vault you wish to query."
    )
    CDD_API_URL: str = Field(
        "https://app.collaborativedrug.com", description="The base url of the CCD API"
    )
    CDD_API_VERSION: str = Field("v1", description="The version of CDD API to use.")
