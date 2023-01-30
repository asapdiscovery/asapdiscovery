from src.arg_parser import arg_parser
from src.molecule_set_crud import MoleculeSetCRUD, MoleculeList, MoleculeUpdateList

import pandas as pd


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
