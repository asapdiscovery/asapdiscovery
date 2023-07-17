import pandas as pd

from datetime import datetime, timedelta

from enum import Enum
from uuid import UUID

from asapdiscovery.docking.docking_data_validation import DockingResultCols

from .manifold_data_validation import (
    TargetTags,
    map_output_col_to_manifold_tag,
)
from .molecule_set import MoleculeSetAPI
from ..aws.cloudfront import CloudFront
from ..aws.s3 import S3


class ArtifactType(Enum):
    DOCKING_POSE_POSIT = "docking-pose-POSIT"
    MD_POSE = "md-pose"


ARTIFACT_TYPE_TO_S3_CONTENT_TYPE = {
    ArtifactType.DOCKING_POSE_POSIT: "text/html",
    ArtifactType.MD_POSE: "image/gif",
}


class ManifoldArtifactUploader:
    """
    This class is used to upload artifacts to the Postera Manifold and AWS simultaneously, linking the two together
    via CloudFront and the Manifold MoleculeSetAPI.

    """

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
        bucket_name: str,
        manifold_id_column: str = DockingResultCols.LIGAND_ID.value,
    ):
        """
        Parameters
        ----------
        molecule_dataframe : pd.DataFrame
            The dataframe containing the molecules to upload. Must contain a column with the name of the artifact
        molecule_set_id : UUID
            The UUID of the molecule set to upload to
        artifact_type : ArtifactType
            The type of artifact to upload
        moleculeset_api : MoleculeSetAPI
            The MoleculeSetAPI object to use to upload to Manifold
        cloud_front : CloudFront
            The CloudFront object to use to generate signed urls
        s3 : S3
            The S3 object to use to upload to S3
        target : str
            The target to upload to
        artifact_column : str
            The name of the column containing the artifact
        bucket_name : str
            The name of the bucket to upload to
        manifold_id_column : str
            The name of the column containing the manifold id
        """
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
        """
        Generate a signed url for a given bucket path

        Parameters
        ----------
        bucket_path : str
            The path to the file in the bucket
        expires_delta : timedelta
            The timedelta for the signed url to be valid for

        Returns
        -------
        str
            The signed url for the file on S3
        """
        # make a signed url with default timedelta of 5 years
        expiry = datetime.utcnow() + expires_delta
        return self.cloud_front.generate_signed_url(bucket_path, expiry)

    def upload_artifacts(self) -> None:
        """
        Upload the artifacts to Postera Manifold and S3
        """

        # rename columns to match manifold
        output_tag_name = map_output_col_to_manifold_tag(ArtifactType, {}, self.target)[
            self.artifact_type.value
        ]

        self.molecule_dataframe["_bucket_path"] = self.molecule_dataframe[
            self.manifold_id_column
        ].apply(lambda x: f"{output_tag_name}/{self.molecule_set_id}/{x}.html")

        # now make urls
        self.molecule_dataframe[output_tag_name] = self.molecule_dataframe[
            "_bucket_path"
        ].apply(lambda x: self.generate_cloudfront_url(x))

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
