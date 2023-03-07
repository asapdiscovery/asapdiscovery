import contextlib
import os
import traceback
import uuid

from asapdiscovery.data.cli import cli
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


# based on https://stackoverflow.com/a/34333710
@contextlib.contextmanager
def set_env_vars(env):
    old_env = dict(os.environ)
    try:
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


class TestS3CLI:
    def test_s3_push_file(self, s3_fresh, tmpdir):

        s3 = s3_fresh

        filename = f"{str(uuid.uuid4())}.txt"

        with tmpdir.as_cwd():
            with open(filename, "w") as f:
                f.write("Welcome to the *#&#@*^ enrichment center")

            # run the CLI
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "aws",
                    "s3",
                    "--bucket",
                    "test-bucket",
                    "--prefix",
                    "test-prefix",
                    "--endpoint-url",
                    "http://127.0.0.1:5000",
                    "push",
                    filename,
                ],
            )
            assert click_success(result)

            # examine object metadata
            objs = list(s3.resource.Bucket(s3.bucket).objects.all())

            assert len(objs) == 1
            assert objs[0].key == os.path.join(s3.prefix, filename)
