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
    ALCHEMISCALE_URL: str = Field(
        "https://api.alchemiscale.org",
        description="The address of the alchemiscale instance to connect to.",
    )
