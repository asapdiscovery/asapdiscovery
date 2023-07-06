"""Interface for signed URL generation via CloudFront.

"""

import datetime
from os import PathLike

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from botocore.signers import CloudFrontSigner


class CloudFront:

    def __init__(self, domain_name: str, key_id: str, private_key_pem_path: PathLike):
        """Create an interface to AWS CloudFront.

        Parameters
        ----------
        domain_name
            The domain name of the CloudFront Distribution to use.
        key_id
            The ID of the public key registered on CloudFront to use for signing URLs.
        private_key_pem_path
            Path to the private key, in PEM format, to use for signing.
            Must correspond to the public key registered on CloudFront under `key_id`.

        """

        self.domain_name = domain_name
        self.key_id = key_id

        with open(private_key_pem_path, 'rb') as key_file:
            self._private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )

    def generate_signed_url(self, object_path: str, expire: datetime.datetime):
        """Generate a signed URL for a given object hosted on S3, served through CloudFront.

        Parameters
        ----------
        object_path
            The path of the target object within its S3 bucket.
            No leading slash.
        expire
            Expiration datetime of the signed URL. Can be set arbitrarily far
            into the future. A signed URL with an expire datetime in the past
            is no longer valid for use.

        """

        url = f"https://{self.domain_name.strip('/')}/{object_path.strip('/')}"

        def rsa_signer(message):
            return self._private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())

        cloudfront_signer = CloudFrontSigner(self.key_id, rsa_signer)

        signed_url = cloudfront_signer.generate_presigned_url(
            url, date_less_than=expire)

        return signed_url
