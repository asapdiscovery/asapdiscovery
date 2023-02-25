from pathlib import Path

import click

from ..cli import cli

# use these extra kwargs with any option from a settings parameter.
SETTINGS_OPTION_KWARGS = {
    "show_envvar": True,
}


def aws_params(func):
    access_key_id = click.option(
        "--access-key-id",
        type=str,
        default=None,
        envvar="AWS_ACCESS_KEY_ID",
        **SETTINGS_OPTION_KWARGS,
    )
    secret_access_key = click.option(
        "--secret-access-key",
        type=str,
        default=None,
        envvar="AWS_SECRET_ACCESS_KEY",
        **SETTINGS_OPTION_KWARGS,
    )
    session_token = click.option(
        "--session-token",
        type=str,
        default=None,
        envvar="AWS_SESSION_TOKEN",
        **SETTINGS_OPTION_KWARGS,
    )
    default_region = click.option(
        "--default-region",
        type=str,
        envvar="AWS_DEFAULT_REGION",
        **SETTINGS_OPTION_KWARGS,
    )
    return access_key_id(secret_access_key(session_token(default_region(func))))


def s3_params(func):
    s3_bucket = click.option(
        "--s3-bucket", type=str, envvar="AWS_S3_BUCKET", **SETTINGS_OPTION_KWARGS
    )
    s3_prefix = click.option(
        "--s3-prefix", type=str, envvar="AWS_S3_PREFIX", **SETTINGS_OPTION_KWARGS
    )
    return s3_bucket(s3_prefix(func))


@cli.group(help="Commands for interacting with the AWS API")
@aws_params
@click.pass_context
def aws(ctx, access_key_id, secret_access_key, session_token, default_region):
    from boto3.session import Session

    ctx.ensure_object(dict)

    ctx.obj["access_key_id"] = access_key_id
    ctx.obj["secret_access_key"] = secret_access_key
    ctx.obj["session_token"] = session_token
    ctx.obj["default_region"] = default_region

    # create a boto3 Session and parameterize with keys
    session = Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token,
        region_name=default_region,
    )

    ctx.obj["session"] = session


@aws.group(help="Commands for interacting with AWS S3")
@s3_params
@click.pass_context
def s3(ctx, s3_bucket, s3_prefix):
    from .s3 import S3

    ctx.obj["s3"] = S3(ctx.obj["session"], s3_bucket, s3_prefix)


@s3.command(help="Push artifacts to S3")
@click.pass_context
@click.argument("files", type=click.Path(exists=True), nargs=-1)
def push(ctx, files):

    s3 = ctx.obj["s3"]

    click.echo(f"Pushing {len(files)} to 's3://{s3.bucket}/{s3.prefix}'")

    for file in files:
        path = Path(file)
        s3.push_file(path.name, path)
        click.echo(f" : '{file}'")

    click.echo(f"Done")


@s3.command(help="Pull artifacts from S3")
@click.pass_context
def pull(ctx):
    ...
