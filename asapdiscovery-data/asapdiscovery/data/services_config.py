from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic import BaseSettings


class PosteraSettings(BaseSettings):
    POSTERA_API_KEY: str
    POSTERA_API_URL: str = "https://api.asap.postera.ai"
    POSTERA_API_VERSION: str = "v1"


class S3Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    BUCKET_NAME: str
    BUCKET_PREFIX: Optional[str] = None


class CloudfrontSettings(BaseSettings):
    CLOUDFRONT_DOMAIN: str
    CLOUDFRONT_KEY_ID: str
    CLOUDFRONT_PRIVATE_KEY_PEM: str

    # validate cloudfront_private_key_pem exists
    @field_validator("cloudfront_private_key_pem")
    @classmethod
    def validate_cloudfront_private_key_pem_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Cloudfront private key file does not exist: {v}")
        return v
