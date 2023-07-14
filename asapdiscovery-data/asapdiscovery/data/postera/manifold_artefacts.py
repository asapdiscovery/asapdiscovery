import pandas as pd

from datetime import datetime, timedelta

from enum import Enum
from typing import Optional
from uuid import UUID

from asapdiscovery.docking.docking_data_validation import DockingResultCols

from .manifold_data_validation import TargetTags, rename_output_columns_for_manifold
from .molecule_set import MoleculeUpdateList, MoleculeSetAPI
from ..aws.cloudfront import CloudFront
from ..aws.s3 import S3


class ArtifactType(Enum):
    DOCKING_POSE = "docking-pose-POSIT"
    MD_POSE = "md-pose"


ARTIFACT_TYPE_TO_S3_CONTENT_TYPE = {
    ArtifactType.DOCKING_POSE: "text/html",
    ArtifactType.MD_POSE: "image/gif",
}


class ManifoldArtifactUploader:
    def __init__(
        self,
        molecule_dataframe: pd.DataFrame,
        molecule_set_id: UUID,
        artifact_type: ArtifactType,
        moleculeset_api: MoleculeSetAPI,
        cloud_front: CloudFront,
        s3: S3,
        target: str,
        artifact_column: str,
        bucket_name: str = "asapdiscovery-ccc-artifacts",
        manifold_id_column: Optional[str] = DockingResultCols.LIGAND_ID.value,
    ):
        self.molecule_dataframe = molecule_dataframe
        self.molecule_set_id = molecule_set_id
        self.artifact_type = artifact_type
        self.moleculeset_api = moleculeset_api
        self.cloud_front = cloud_front
        self.s3 = s3
        self.target = target
        self.artifact_column = artifact_column
        self.bucket_name = bucket_name
        self.manifold_id_column = manifold_id_column

        if not TargetTags.is_in_values(target):
            raise ValueError(
                f"Target {target} not in allowed values {TargetTags.get_values()}"
            )

    def generate_cloudfront_url(
        self, bucket_path, expires_delta: timedelta = timedelta(days=365 * 5)
    ) -> str:
        # make a signed url with default timedelta of 5 years
        expiry = datetime.utcnow() + expires_delta
        return self.cloud_front.generate_signed_url(bucket_path, expiry)

    def upload_artifacts(self) -> None:
        # use a lambda to generate bucket_paths
        self.molecule_dataframe["_bucket_path"] = self.molecule_dataframe[
            self.manifold_id_column
        ].apply(lambda x: f"{self.artifact_type.value}/{self.molecule_set_id}/{x}.html")

        # now make urls
        self.molecule_dataframe[self.artifact_type.value] = self.molecule_dataframe[
            "_bucket_path"
        ].apply(lambda x: self.generate_cloudfront_url(x))

        # generate correct column names
        # allow all names except for the ones we are renaming
        # TODO: clumsy, but works for now
        allowed = [
            col
            for col in self.molecule_dataframe.columns
            if col != self.artifact_type.value
        ]

        self.molecule_dataframe = rename_output_columns_for_manifold(
            self.molecule_dataframe,
            self.target,
            [ArtifactType],
            manifold_validate=True,
            allow=allowed,
        )

        # this will trim the dataframe to only the columns we want to update
        self.moleculeset_api.update_molecules_from_df_with_manifold_validation(
            self.molecule_set_id,
            self.molecule_dataframe,
            id_field=self.manifold_id_column,
        )

        # push to s3
        self.molecule_dataframe.apply(
            lambda x: self.s3.push_file(
                x[self.artifact_column],
                location=x["_bucket_path"],
                content_type=ARTIFACT_TYPE_TO_S3_CONTENT_TYPE[self.artifact_type],
            ),
            axis=1,
        )
