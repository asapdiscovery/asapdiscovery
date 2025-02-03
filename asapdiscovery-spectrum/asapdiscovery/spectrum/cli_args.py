import click


def seq_file(func):
    return click.option(
        "-f",
        "--seq-file",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="File containing reference sequences",
    )(func)


def seq_type(func):
    return click.option(
        "-t",
        "--seq_type",
        type=click.Choice(["fasta", "pdb", "pre-calc"]),
        help="Type of input from which the sequence will be read.",
        default="fasta",
        show_default=True,
    )(func)


def blast_json(func):
    return click.option(
        "--blast-json",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="Path to a json file containing parameters for the blast search.",
    )(func)


def email(func):
    return click.option(
        "--email",
        type=str,
        default="",
        help="Email for Entrez search.",
    )(func)


def max_mismatches(func):
    return click.option(
        "--max-mismatches",
        default=0,
        help="Maximum number of aminoacid group missmatches to be allowed in color-seq-match mode.",
    )(func)


def gen_ref_pdb(func):
    return click.option(
        "--gen-ref-pdb",
        is_flag=True,
        default=False,
        help="Whether to retrieve a pdb file for the query structure.",
    )(func)


def multimer(func):
    return click.option(
        "--multimer",
        is_flag=True,
        default=False,
        help="Store the output sequences for a multimer ColabFold run (from identical chains)."
        ' If not set, "--n-chains" will not be used. ',
    )(func)


def n_chains(func):
    return click.option(
        "--n-chains",
        type=int,
        default=None,
        help="Number of repeated chains that will be saved in csv file."
        ' Requires calling the "--multimer" flag',
    )(func)


def pymol_save(func):
    return click.option(
        "--pymol-save",
        type=str,
        default="session.pse",
        help="Path to file where session will be saved.",
    )(func)
