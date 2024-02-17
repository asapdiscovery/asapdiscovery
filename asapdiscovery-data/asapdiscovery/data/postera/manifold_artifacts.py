from datetime import datetime, timedelta
from enum import Enum
from typing import Union, Optional
from uuid import UUID
from pydantic import BaseModel, Field, validator, root_validator
import pandas as pd
from asapdiscovery.data.services_config import (
    CloudfrontSettings,
    PosteraSettings,
    S3Settings,
)
from asapdiscovery.docking.docking_data_validation import DockingResultCols

from asapdiscovery.data.aws.cloudfront import CloudFront
from asapdiscovery.data.aws.s3 import S3
from asapdiscovery.data.postera.manifold_data_validation import (
    TargetTags,
    map_output_col_to_manifold_tag,
)
from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI


class ArtifactType(Enum):
    DOCKING_POSE_POSIT = "docking-pose-POSIT"
    DOCKING_POSE_FITNESS_POSIT = "docking-pose-fitness-POSIT"
    MD_POSE = "md-pose"


ARTIFACT_TYPE_TO_S3_CONTENT_TYPE = {
    ArtifactType.DOCKING_POSE_POSIT: "text/html",
    ArtifactType.DOCKING_POSE_FITNESS_POSIT: "text/html",
    ArtifactType.MD_POSE: "image/gif",
}


class ManifoldArtifactUploader(BaseModel):
    target: TargetTags = Field(
        ..., description="The biological target string for the artifact"
    )
    molecule_dataframe: pd.DataFrame = Field(
        ...,
        description="The dataframe containing the molecules and artifacts to upload",
    )
    molecule_set_id: Optional[Union[UUID, str]] = Field(
        ..., description="The UUID of the molecule set to upload to"
    )
    molecule_set_name: Optional[str] = Field(
        ..., description="The name of the molecule set to upload to"
    )

    bucket_name: str = Field(..., description="The name of the bucket to upload to")

    artifact_columns: list[str] = Field(
        ...,
        description="The name of the column containing the filesystem path to the artifacts that will be uploaded.",
    )

    artifact_types: list[ArtifactType] = Field(
        ..., description="The type of artifacts to upload"
    )

    moleculeset_api: Optional[MoleculeSetAPI] = Field(
        ..., description="The MoleculeSetAPI object to use to upload to Manifold"
    )

    cloudfront: Optional[CloudFront] = Field(
        ..., description="The CloudFront object to use to generate signed urls"
    )

    s3: Optional[S3] = Field(..., description="The S3 object to use to upload to S3")

    manifold_id_column: str = Field(
        DockingResultCols.LIGAND_ID.value,
        description="The name of the column containing the manifold id",
    )

    @root_validator
    @classmethod
    def validate_artifact_columns_and_types(cls, values):
        if len(values["artifact_columns"]) != len(values["artifact_types"]):
            raise ValueError(
                "Number of artifact columns must match number of artifact types"
            )
        return values

    @root_validator
    @classmethod
    def name_id_mutually_exclusive(cls, values):
        if values["molecule_set_id"] and values["molecule_set_name"]:
            raise ValueError(
                "molecule_set_id and molecule_set_name are mutually exclusive"
            )
        return values

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

    def upload_artifacts(self) -> None:
        """
        Upload the artifacts to Postera Manifold and S3
        """

        if self.cloudfront is None:
            self.cloudfront = CloudFront.from_settings(CloudfrontSettings())

        if self.s3 is None:
            self.s3 = S3.from_settings(S3Settings())

        if self.moleculeset_api is None:
            self.moleculeset_api = MoleculeSetAPI.from_settings(PosteraSettings())

        if self.molset_id is None:
            self.molset_id = self.ms_api.get_id_from_name(self.molecule_set_name)

        for artifact_column, artifact_type in zip(
            self.artifact_columns, self.artifact_types
        ):
            subset_df = self.molecule_dataframe[
                [artifact_column, self.manifold_id_column]
            ].copy()
            # rename columns to match manifold
            output_tag_name = map_output_col_to_manifold_tag(ArtifactType, self.target)[
                artifact_type.value
            ]

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
