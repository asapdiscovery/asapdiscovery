from datetime import datetime, timedelta
from enum import Enum
from typing import Union
from uuid import UUID

import pandas as pd
from asapdiscovery.data.services_config import (
    CloudfrontSettings,
    PosteraSettings,
    S3Settings,
)
from asapdiscovery.docking.docking_data_validation import DockingResultCols

from ..aws.cloudfront import CloudFront
from ..aws.s3 import S3
from .manifold_data_validation import TargetTags, map_output_col_to_manifold_tag
from .molecule_set import _POSTERA_COLUMN_BLEACHING_ACTIVE, MoleculeSetAPI


class ArtifactType(Enum):
    DOCKING_POSE_POSIT = "docking-pose-POSIT"
    DOCKING_POSE_FITNESS_POSIT = "docking-pose-fitness-POSIT"
    MD_POSE = "md-pose"


ARTIFACT_TYPE_TO_S3_CONTENT_TYPE = {
    ArtifactType.DOCKING_POSE_POSIT: "text/html",
    ArtifactType.DOCKING_POSE_FITNESS_POSIT: "text/html",
    ArtifactType.MD_POSE: "image/gif",
}


class ManifoldArtifactUploader:
    """
    This class is used to upload artifacts to the Postera Manifold and AWS simultaneously, linking the two together
    via CloudFront and the Manifold MoleculeSetAPI.

    """

    def __init__(
        self,
        target: str,
        molecule_dataframe: pd.DataFrame,
        molecule_set_id_or_name: Union[str, UUID],
        bucket_name: str,
        artifact_columns: list[str],
        artifact_types: list[ArtifactType],
        moleculeset_api: MoleculeSetAPI = None,
        cloudfront: CloudFront = None,
        s3: S3 = None,
        manifold_id_column: str = DockingResultCols.LIGAND_ID.value,
    ):
        """
        Parameters
        ----------
        molecule_dataframe : pd.DataFrame
            The dataframe containing the molecules to upload.
            Must contain a column with the name of the artifact
        molecule_set_id : UUID
            The UUID of the molecule set to upload to
        artifact_type : ArtifactType
            The type of artifact to upload
        moleculeset_api : MoleculeSetAPI
            The MoleculeSetAPI object to use to upload to Manifold
        cloudfront : CloudFront
            The CloudFront object to use to generate signed urls
        s3 : S3
            The S3 object to use to upload to S3
        target : str
            The biological target string for the artifact, one of asapdiscovery.data.postera.manifold_data_validation.TargetTags
        artifact_column : str
            The name of the column containing the filesystem path to the artifact that will be uploaded.
        bucket_name : str
            The name of the bucket to upload to
        manifold_id_column : str
            The name of the column containing the manifold id
        """
        self.target = target
        self.molecule_dataframe = molecule_dataframe.copy()
        self.molecule_set_id_or_name = molecule_set_id_or_name
        self.bucket_name = bucket_name
        self.artifact_columns = artifact_columns
        self.artifact_types = artifact_types

        if moleculeset_api is None:
            moleculeset_api = MoleculeSetAPI().from_settings(PosteraSettings())
        self.moleculeset_api = moleculeset_api

        self.molset_id = self.moleculeset_api.molecule_set_id_or_name(
            self.molecule_set_id_or_name, self.moleculeset_api.list_available()
        )

        if cloudfront is None:
            cloudfront = CloudFront.from_settings(CloudfrontSettings())
        self.cloudfront = cloudfront

        if s3 is None:
            s3 = S3.from_settings(S3Settings())
        self.s3 = s3

        self.manifold_id_column = manifold_id_column
        if len(self.artifact_columns) != len(self.artifact_types):
            raise ValueError(
                "Number of artifact columns must match number of artifact types"
            )

        if not TargetTags.is_in_values(target):
            raise ValueError(
                f"Target {target} not in allowed values {TargetTags.get_values()}"
            )

        # drop rows in molecule df where artifact_columns is None or NaN
        self.molecule_dataframe = self.molecule_dataframe.dropna(
            subset=self.artifact_columns
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
        return self.cloudfront.generate_signed_url(bucket_path, expiry)

    def upload_artifacts(self, bleached=_POSTERA_COLUMN_BLEACHING_ACTIVE) -> None:
        """
        Upload the artifacts to Postera Manifold and S3
        """
        for artifact_column, artifact_type in zip(
            self.artifact_columns, self.artifact_types
        ):
            if bleached:
                artifact_column = artifact_column.replace(
                    "-", "_"
                )  # NOTE: remove when bleaching is removed
            subset_df = self.molecule_dataframe[
                [artifact_column, self.manifold_id_column]
            ].copy()
            # rename columns to match manifold
            output_tag_name = map_output_col_to_manifold_tag(ArtifactType, self.target)[
                artifact_type.value
            ]

            if bleached:
                output_tag_name = output_tag_name.replace(
                    "-", "_"
                )  # NOTE: remove when bleaching is removed

            subset_df[f"_bucket_path_{artifact_column}"] = subset_df[
                self.manifold_id_column
            ].apply(lambda x: f"{output_tag_name}/{self.molset_id}/{x}.html")

            # now make urls
            subset_df[output_tag_name] = subset_df[
                f"_bucket_path_{artifact_column}"
            ].apply(lambda x: self.generate_cloudfront_url(x))

            # this will trim the dataframe to only the columns we want to update
            self.moleculeset_api.update_molecules_from_df_with_manifold_validation(
                self.molset_id,
                subset_df,
                id_field=self.manifold_id_column,
                bleached=bleached,
            )

            # push to s3
            subset_df.apply(
                lambda x: self.s3.push_file(
                    x[artifact_column],
                    location=x[f"_bucket_path_{artifact_column}"],
                    content_type=ARTIFACT_TYPE_TO_S3_CONTENT_TYPE[artifact_type],
                ),
                axis=1,
            )
