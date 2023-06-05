def makeS3URL(s3_bucket_prefix):
    """For a given s3 bucket name, returns a randomly generated URL and its corresponding bucket path."""

    # create the S3 path for the pose HTML to be stored in. Use a random string instead of
    # ligand name to improve security.
    random_string = "".join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits)
        for _ in range(20)
    )

    s3_bucket_path = f"{s3_bucket_prefix}/{random_string}.html"
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_bucket_path}"

    return s3_url, s3_bucket_path
