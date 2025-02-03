import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from asapdiscovery.data.services.aws.cloudfront import CloudFront
from asapdiscovery.data.services.aws.s3 import S3
from asapdiscovery.data.services.postera.manifold_data_validation import (
    TargetTags,
    map_output_col_to_manifold_tag,
)
from asapdiscovery.data.services.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.services.services_config import (
    CloudfrontSettings,
    PosteraSettings,
    S3Settings,
)
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from pydantic.v1 import BaseModel, Field, root_validator


class ArtifactType(Enum):
    DOCKING_POSE_POSIT = "docking-pose-POSIT"
    DOCKING_POSE_FITNESS_POSIT = "docking-pose-fitness-POSIT"
    MD_POSE = "md-pose"


ARTIFACT_TYPE_TO_S3_CONTENT_TYPE = {
    ArtifactType.DOCKING_POSE_POSIT: "text/html",
    ArtifactType.DOCKING_POSE_FITNESS_POSIT: "text/html",
    ArtifactType.MD_POSE: "image/gif",
}

logger = logging.getLogger(__name__)


class ManifoldArtifactUploader(BaseModel):
    target: TargetTags = Field(
        None, description="The biological target string for the artifact"
    )
    molecule_dataframe: pd.DataFrame = Field(
        ...,
        description="The dataframe containing the molecules and artifacts to upload",
    )
    molecule_set_id: Optional[str] = Field(
        None, description="The UUID of the molecule set to upload to"
    )
    molecule_set_name: Optional[str] = Field(
        None, description="The name of the molecule set to upload to"
    )

    bucket_name: str = Field(..., description="The name of the bucket to upload to")

    artifact_columns: list[str] = Field(
        None,
        description="The name of the column containing the filesystem path to the artifacts that will be uploaded.",
    )

    artifact_types: list[ArtifactType] = Field(
        None, description="The type of artifacts to upload"
    )

    moleculeset_api: Optional[MoleculeSetAPI] = Field(
        None, description="The MoleculeSetAPI object to use to upload to Manifold"
    )

    cloudfront: Optional[CloudFront] = Field(
        None, description="The CloudFront object to use to generate signed urls"
    )

    s3: Optional[S3] = Field(None, description="The S3 object to use to upload to S3")

    manifold_id_column: str = Field(
        DockingResultCols.LIGAND_ID.value,
        description="The name of the column containing the manifold id",
    )

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    @classmethod
    def validate_artifact_columns_and_types(cls, values):
        artifact_columns = values.get("artifact_columns")
        artifact_types = values.get("artifact_types")
        if len(artifact_columns) != len(artifact_types):
            raise ValueError(
                "Number of artifact columns must match number of artifact types"
            )
        if len(artifact_columns) == len(artifact_types) == 0:
            raise ValueError("Must have at least one artifact column")

        return values

    @root_validator
    @classmethod
    def name_id_mutually_exclusive(cls, values):
        molecule_set_id = values.get("molecule_set_id")
        molecule_set_name = values.get("molecule_set_name")

        if not molecule_set_id and not molecule_set_name:
            raise ValueError("Must provide molecule_set_id or molecule_set_name")

        if molecule_set_id and molecule_set_name:
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

    def upload_artifacts(self, sort_column=None, sort_ascending=False) -> None:
        """
        Upload the artifacts to Postera Manifold and S3
        """

        if self.cloudfront is None:
            self.cloudfront = CloudFront.from_settings(CloudfrontSettings())

        if self.s3 is None:
            self.s3 = S3.from_settings(S3Settings())

        if self.moleculeset_api is None:
            self.moleculeset_api = MoleculeSetAPI.from_settings(PosteraSettings())

        if self.molecule_set_id is None:
            self.molecule_set_id = self.moleculeset_api.get_id_from_name(
                self.molecule_set_name
            )

        # remove duplicates by tag
        self.molecule_dataframe = self.remove_duplicates(
            self.molecule_dataframe, sort_column, sort_ascending=sort_ascending
        )

        for artifact_column, artifact_type in zip(
            self.artifact_columns, self.artifact_types
        ):
            logger.info(f"Uploading {artifact_type} artifacts from {artifact_column}")
            subset_df = self.molecule_dataframe[
                [artifact_column, self.manifold_id_column]
            ].copy()

            # check if there is any data to upload
            if subset_df[artifact_column].isnull().all():
                logger.info(
                    f"No data to upload for {artifact_type} from {artifact_column}"
                )
                continue

            # rename columns to match manifold
            output_tag_name = map_output_col_to_manifold_tag(ArtifactType, self.target)[
                artifact_type.value
            ]

            # subselect non-null artifact column rows
            subset_df = subset_df.dropna(subset=[artifact_column])

            subset_df[f"_bucket_path_{artifact_column}"] = subset_df[
                self.manifold_id_column
            ].apply(lambda x: f"{output_tag_name}/{self.molecule_set_id}/{x}.html")

            # now make urls
            subset_df[output_tag_name] = subset_df[
                f"_bucket_path_{artifact_column}"
            ].apply(lambda x: self.generate_cloudfront_url(x))

            # this will trim the dataframe to only the columns we want to update
            self.moleculeset_api.update_molecules_from_df_with_manifold_validation(
                self.molecule_set_id,
                subset_df,
                id_field=self.manifold_id_column,
            )

            self._upload_column_to_s3(
                subset_df,
                artifact_column,
                f"_bucket_path_{artifact_column}",
                artifact_type,
            )

    def _upload_column_to_s3(self, row, artifact_column, bucket_path, artifact_type):
        for _, row in row.iterrows():
            if pd.notnull(row[artifact_column]) and pd.notnull(row[bucket_path]):
                logger.debug(f"S3 push: {row[artifact_column]} to {row[bucket_path]}")
                self.s3.push_file(
                    row[artifact_column],
                    location=row[bucket_path],
                    content_type=ARTIFACT_TYPE_TO_S3_CONTENT_TYPE[artifact_type],
                )

    def remove_duplicates(self, data, sort_column, sort_ascending=False):
        """
        Remove duplicates from the dataframe

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload
        id_field : str
            Name of the column in the dataframe to use as the ligand id
        sort_column : str
            The column to sort the data by if duplicates are found
        sort_ascending : bool
            Whether the data should be sorted in ascending order

        Returns
        -------
        DataFrame
            The input dataframe with duplicates removed
        """
        dup, _ = self._check_for_duplicates(
            data, self.manifold_id_column, allow_empty=True, raise_error=False
        )
        if dup:
            if not sort_column:
                raise ValueError("sort_column must be provided if duplicates are found")
            if sort_column not in data.columns:
                raise ValueError(f"sort_column {sort_column} not found in dataframe")
            data = data.sort_values(by=sort_column, ascending=sort_ascending)
            data = data.drop_duplicates(subset=[self.manifold_id_column], keep="first")

        return data

    @staticmethod
    def _check_for_duplicates(
        df,
        id_field,
        allow_empty=True,
        raise_error=False,
        sort_column=None,
        sort_ascending=False,
    ):
        """
        Check for duplicate UUIDs in the dataframe

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload
        id_field : str
            Name of the column in the dataframe to use as the ligand id
        allow_empty : bool
            Whether to allow empty UUIDs to be exempt from the check
        raise_error : bool
            Whether to raise an error if duplicates are found

        Raises
        ------
        ValueError
            If there are duplicate UUIDs
        """
        df = df.copy()
        df = df.replace("", np.nan)
        if allow_empty:
            df = df[~df[id_field].isna()]
        if df[id_field].duplicated().any():
            duplicates = df[df[id_field].duplicated()]
            num_duplicates = len(duplicates)
            if raise_error:
                raise ValueError(f"{num_duplicates} duplicate UUIDs found in dataframe")
            return True, duplicates
        else:
            return False, None
