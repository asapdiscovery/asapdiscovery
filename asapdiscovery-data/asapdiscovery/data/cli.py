from .postera.molecule_set import MoleculeSetCRUD, MoleculeList, MoleculeUpdateList

import pandas as pd

import argparse


def molecule_set_arg_parser(parser):

    parser.add_argument(
        "action",
        action="store",
        default="",
        choices=["create", "read", "update"],
        help="Actions for molecule set.\n"
        "create - create a molecule set (requires --input_file)\n"
        "read - export molecule set to csv file (requires --molecule_set_id)\n"
        "update - update custom data associated to molecules in a set (requires --molecule_set_id and --input_file)\n",
    )

    parser.add_argument(
        "--input_file",
        dest="input_file",
        action="store",
        default="",
        help="Path to csv file containing data.\n"
        "If using create, the file must include a field containing SMILES.\n"
        "If using update, the file must include a field containing postera molecule ids\n",
    )

    parser.add_argument(
        "--molecule_set_name",
        dest="molecule_set_name",
        action="store",
        default="New_Set",
        help="Name of molecule set - required when using create",
    )

    parser.add_argument(
        "--molecule_set_id",
        dest="molecule_set_id",
        action="store",
        default="",
        help="Molecule set id - required when using read or update",
    )

    parser.add_argument(
        "--smiles_field",
        dest="smiles_field",
        action="store",
        default="SMILES",
        help="Input csv field that stores the SMILES - required when using create",
    )

    parser.add_argument(
        "--id_field",
        dest="postera_id_field",
        action="store",
        default="postera_molecule_id",
        help="Input csv field that stores the postera molecule id - required when using update",
    )


def arg_parser():

    parser = argparse.ArgumentParser(
        description="Command line interface to PostEra API",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Usage:\n"
        "python postera_api <POSTERA_API_URL> <YOUR_POSTERA_API_TOKEN> <POSTERA_API_ENDPOINT> <ACTION> --<OTHER_ARGS>\n"
        "Molecule set help:\n"
        "python postera_api <POSTERA_API_URL> <YOUR_POSTERA_API_TOKEN> moleculeset --help",
    )

    parser.add_argument("url", action="store", default="", help="PostEra API URL")

    parser.add_argument(
        "api_token", action="store", default="", help="PostEra API token"
    )

    parser.add_argument(
        "--api_version",
        dest="api_version",
        action="store",
        default="v1",
        help="PostEra API version",
    )

    subparsers = parser.add_subparsers(
        help="sub-command help",
        dest="subparser_name",
    )

    molecule_set_subparser = subparsers.add_parser(
        "moleculeset",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Molecule set command line interface\n",
        epilog="To create a molecule set:\n"
        "python postera_api <POSTERA_API_URL> <YOUR_POSTERA_API_TOKEN> moleculeset create --input_file <INPUT_CSV>\n\n"
        "To read a molecule set:\n"
        "python postera_api <POSTERA_API_URL> <YOUR_POSTERA_API_TOKEN> moleculeset read --molecule_set_id <SET_ID>\n\n"
        "To update the custom fields of molecules in a set:\n"
        "python postera_api <POSTERA_API_URL> <YOUR_POSTERA_API_TOKEN> moleculeset update --molecule_set_id <SET_ID>"
        " --input_file <INPUT_CSV>\n\n",
    )

    molecule_set_arg_parser(molecule_set_subparser)

    args = parser.parse_args()

    return args

def molecule_set(args):
    """
    Handle molecule set commands defined in args.action given:

    Manifold API URL defined in args.url
    Manifold API version defined in args.api_version
    Manifold API token defined in args.api_token

    args.action = read - get the molecule set with id args.molecule_set_id and write to csv

    args.action = create - create a molecule set with name args.molcule_set_name
    from data in args.input_file with SMILES field defined in args.smiles_field

    args.action = update - update the custom fields in the molecule set with id args.molecule_set_id
    with data supplied in args.input_file given the postera molecule ids supplied in args.postera_id_field
    """

    molecule_set = MoleculeSetCRUD(args.url, args.api_version, args.api_token)

    if args.action == "read":

        if not args.molecule_set_id:
            raise ValueError("Molecule Set id required for read command")

        df = molecule_set.read(args.molecule_set_id)
        outfile_name = f"Molecule_set_{args.molecule_set_id}.csv"
        df.to_csv(outfile_name, index=False)
        print(f"Wrote molecule set to {outfile_name}")

    elif args.action == "create":

        if not args.input_file:
            raise ValueError("Input file required for create command")

        df = pd.read_csv(args.input_file)

        if args.smiles_field not in df.columns:
            raise ValueError(
                f"Smiles field {args.smiles_field} not found in input file"
            )

        molecule_list = MoleculeList()
        molecule_list.from_pandas_df(
            df, smiles_field=args.smiles_field, first_entry=0, last_entry=len(df)
        )

        molecule_set_id = molecule_set.create(molecule_list, args.molecule_set_name)
        print(f"Created molecule set with id {molecule_set_id}")

    elif args.action == "update":

        if not args.input_file:
            raise ValueError("Input file required for update command")

        if not args.molecule_set_id:
            raise ValueError("Molecule Set id required for update command")

        df = pd.read_csv(args.input_file)

        if args.postera_id_field not in df.columns:
            raise ValueError(
                f"PostEra molecule id field {args.postera_id_field} not found in input file"
            )

        update_molecule_list = MoleculeUpdateList()
        update_molecule_list.from_pandas_df(
            df,
            postera_id_field=args.postera_id_field,
            first_entry=0,
            last_entry=len(df),
        )

        molecules_updated = molecule_set.update_custom_data(
            args.molecule_set_id, update_molecule_list
        )
        print(f"Updated molecules {molecules_updated} in set {args.molecule_set_id}")


def main():

    args = arg_parser()

    if args.subparser_name == "moleculeset":
        molecule_set(args)


if __name__ == "__main__":
    main()
