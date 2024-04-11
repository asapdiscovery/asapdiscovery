from pydantic import BaseSettings, Field


class AlchemiscaleSettings(BaseSettings):
    """
    General settings class to capture Alchemiscale credentials from the environment.
    """

    ALCHEMISCALE_ID: str = Field(
        ..., description="Your personal alchemiscale ID used to login."
    )
    ALCHEMISCALE_KEY: str = Field(
        ..., description="Your personal alchemiscale Key used to login."
    )
    ALCHEMISCALE_ADDRESS: str = Field(
        "https://api.alchemiscale.org",
        description="The address of the alchemiscale instance to interface with."
    )


class BespokeFitSettings(BaseSettings):
    """
    General settings class to capture BespokeFit credentials from the environment.
    """

    BESPOKEFIT_ADDRESS: str = Field(
        ..., description="The address of the BespokeFit server."
    )
