import pandas as pd

from datetime import datetime, timedelta

from enum import Enum
from typing import Optional
from uuid import UUID

from asapdiscovery.docking.docking_data_validation import DockingResultCols

from .molecule_set import MoleculeUpdateList, MoleculeSetAPI
from ..aws.cloudfront import CloudFront
from ..aws.s3 import S3


class ArtifactType(Enum):
    DOCKING_POSE = "docking-pose"
    MD_POSE = "md-pose"


ARTIFACT_TYPE_TO_S3_CONTENT_TYPE = {
    ArtifactType.DOCKING_POSE: "text/html",
    ArtifactType.MD_POSE: "image/gif",
}


class ManifoldArtifactUploader:
    def __init__(
        self,
        molecule_dataframe: pd.DataFrame,
        bucket_name: str,
        molecule_set_id: uuid.UUID,
        artifact_type: ArtifactType,
        moleculeset_api: MoleculeSetAPI,
        cloud_front: CloudFront,
        s3: S3,
        artifact_column: str,
        manifold_id_column: Optional[str] = DockingResultCols.LIGAND_ID.value,
    ):
        self.molecule_dataframe = molecule_dataframe
        self.bucket_name = bucket_name
        self.molecule_set_id = molecule_set_id
        self.artifact_column = artifact_column
        self.artifact_type = artifact_type
        self.moleculeset_api = moleculeset_api
        self.cloud_front = cloud_front
        self.s3 = s3
        self.manifold_id_column = manifold_id_column

    def generate_cloudfront_url(
        self, bucket_path, expires_delta: timedelta = timedelta(days=365 * 5)
    ) -> str:
        # make a signed url with default timedelta of 5 years
        expiry = datetime.utcnow() + expires_delta
        return self.cloud_front.generate_signed_url(bucket_path, expiry)

    def upload_artifacts(self) -> None:
        # use a lambda to generate cloudfront urls
        self.molecule_dataframe[self.artifact_type.value] = self.molecule_dataframe[
            self.manifold_id_column
        ].apply(
            lambda x: self.generate_cloudfront_url(
                f"{self.artifact_type.value}/{self.molecule_set_id}/{x}.html"
            )
        )
        # push to postera
        update = MoleculeUpdateList.from_pandas_df(self.molecule_dataframe)
        self.moleculeset_api.update_molecules(self.molecule_set_id, update)

        # upload to s3
        self.molecule_dataframe[[self.artifact_column, self.artifact_type.value]].apply(
            lambda x, y: self.S3.push_file(
                x,
                location=y,
                content_type=ARTIFACT_TYPE_TO_S3_CONTENT_TYPE[self.artifact_type],
            )
        )
