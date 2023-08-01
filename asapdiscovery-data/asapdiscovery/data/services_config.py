from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class PosteraSettings(BaseSettings):
    postera_api_key: str
    postera_api_url: str = "https://api.asap.postera.ai"
    postera_api_version: str = "v1"


class S3Settings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    bucket_name: str
    prefix: Optional[str] = None


class CloudfrontSettings(BaseSettings):
    cloudfront_domain: str
    cloudfront_key_id: str
    cloudfront_private_key_pem: str

    # validate cloudfront_private_key_pem exists
    @field_validator("cloudfront_private_key_pem")
    @classmethod
    def validate_cloudfront_private_key_pem_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Cloudfront private key file does not exist: {v}")
        return v
