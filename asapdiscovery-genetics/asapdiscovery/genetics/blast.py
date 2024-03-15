import subprocess
import sys

import pandas as pd
import requests

# BioPython
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

_E_VALUE_THRESH = 1e-20


def parse_blast(results_file: str, verbose: bool=False) -> pd.DataFrame:
    """Parse data from BLAST xml file

    Args:
        results_file (str): Path to BLAST results
        verbose (bool): Whether to print information

    Returns:
        pd.DataFrame: DataFrame with BLAST entries
    """
    # Return DataFrame with BLAST results
    dfs = []
    for record in NCBIXML.parse(open(results_file)):
        if record.alignments:
            query = record.query[:100]
            if verbose:
                print("\n")
                print("query: %s" % query)
            for align in record.alignments:
                for hsp in align.hsps:
                    if hsp.expect < _E_VALUE_THRESH:
                        # Print sequence identity, title, and gapless sequence substring that aligns
                        hsps0 = align.hsps[0]
                        sequence_to_model = hsps0.sbjct.replace("-", "")
                        pidentity = round(
                            100.0
                            * hsps0.identities
                            / (hsps0.query_end - hsps0.query_start + 1),
                            2,
                        )
                        title = align.title
                        id = "".join(title.split(" ")[0])
                        description = " ".join(title.split(" ")[1:])
                        if verbose:
                            print(
                                f"length {hsps0.identities}, score {pidentity}: {align.title}"
                            )
                        data = {
                            "query": [query],
                            "ID": [id],
                            "description": [description],
                            "sequence": [sequence_to_model],
                            "score": [pidentity],
                        }
                        df_row = pd.DataFrame(data)
                        dfs.append(df_row)
                df = pd.concat(dfs, axis=0, ignore_index=True, sort=False).dropna(
                    axis=1, how="all"
                )
    return df


def get_blast_seqs(
    seq_source: str,
    save_folder: str,
    input_type="fasta",
    nhits=100,
    nalign=500,
    database="refseq_protein",
    xml_file="results.xml",
    verbose=True,
    save_csv=None,
) -> pd.DataFrame:
    """Run a BLAST search on a protein sequence.
    Args:
        seq_source (str): Source with the sequence.
        input_type (str, optional): Type of sequence source ["pre-cal", "fasta", "sequence"]. Defaults to "fasta".
        save_file (str, optional): Name of output file storing BLAST results. Defaults to "results.xml".

    Args:
        seq_source (str): Source with the sequence.
        save_folder (str): Path to folder to save BLAST results
        input_type (str, optional): Type of sequence source ["pre-cal", "fasta", "sequence"]. Defaults to "fasta".
        nhits (int, optional): Number of hits, hitlist_size parameter in BLAST. Defaults to 100.
        nalign (int, optional): Number of alignments, alignments parameter in BLAST. Defaults to 500.
        database (str, optional): Name of BLAST database. Defaults to "refseq_protein".
        xml_file (str, optional): Name to be given to XML with BLAST results. Defaults to "results.xml".
        verbose (bool, optional): Whether to print info on BLAST search. Defaults to True.
        save_csv (Union[str, None], optional): CSV file name to optionally save dataframe. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with Blast results.
    """

    from pathlib import Path

    if input_type == "pre-calc":
        matches_df = parse_blast(seq_source, verbose)
        return matches_df
    elif input_type == "fasta":
        # Input is file name
        sequence = open(seq_source).read()
    elif input_type == "sequence":
        # Input is sequence
        sequence = seq_source
    else:  # Another source?
        raise ValueError("unknown input type")

    # Retrieve blastp results
    result_handle = NCBIWWW.qblast(
        "qblast", database, sequence, hitlist_size=nhits, alignments=nalign
    )

    # Create folder if doesn't already exists
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_folder}/{xml_file}"

    with open(save_file, "w") as file:
        blast_results = result_handle.read()
        file.write(blast_results)

    matches_df = parse_blast(save_file, verbose)
    if save_csv:
        matches_df.to_csv(f"{save_folder}/{save_csv}", index=False)

    return matches_df


class PDB_record:
    def __init__(
        self, label: str, query_seq: str, description: str, chain: int
    ) -> None:
        self.label = label
        self.description = description
        self.seq = query_seq
        self.chain = chain


class PDBentry:
    def __init__(self, input_seq: str, input_type: str) -> None:
        self.input = input_seq
        self.type = input_type

    def retrieve_pdb(self, results_folder: str, min_id_match=99, ref_only=False):
        """Retrieve the PDB record of a given sequence."""
        from pathlib import Path

        import prody

        record_name = "pdb_blast.xml"

        # Find pdb matching given sequence
        matches_df = get_blast_seqs(
            self.input,
            results_folder,
            input_type=self.type,
            xml_file=record_name,
            nalign=500,
            database="pdb",
            verbose=False,
        )
        print(f"Saving blast results in {results_folder}/{record_name}")

        # Load original sequence names and descriptors for reference
        pdb_file_record = []
        for seq_record in SeqIO.parse(self.input, self.type):
            seq_id = seq_record.id
            seq_description = seq_record.description
            seq_chain = int(seq_record.id.split("|")[1].split(".")[-1])
            pdb_file_record.append(
                PDB_record(seq_id, seq_record.seq, seq_description, seq_chain)
            )

        best_scores = []
        for query in matches_df["query"].unique():
            best_scores.append(
                matches_df.loc[matches_df["query"] == query, "score"].max()
            )

        match_hits = self.parse_pdb_blast_results(matches_df, min_score=min_id_match)

        for i, hits in enumerate(match_hits):
            if len(hits) == 0:
                print("The record doesn't have a PDB with high confidence")
                if i == 0:
                    raise ValueError(
                        "The reference sequence MUST have an existing PDB entry. Maybe check the sequence?"
                    )
            else:
                # Return the pdb entry with the max alignment and max resolution (if there are multiple matches)
                best_pdb_record = self.choose_best_pdb_entry(hits, best_scores[i])
                filename = prody.fetchPDB(best_pdb_record, folder=results_folder)
                subprocess.run(["gunzip", filename])
                pdb_file_record[i].pdb_file = filename
                pdb_file_record[i].pdb_id = best_pdb_record
                # If I'm only requesting the reference PDB, there's no need to generate more files.
                if ref_only:
                    break

        return pdb_file_record

    @staticmethod
    def parse_pdb_blast_results(blast_df, min_score):
        """For a pdb database search extract pdb ids with a minimum score.
        Outputs are lists because iterables are needed for functions down the pipeline.
        """

        match_hits = []
        for q in blast_df["query"].unique():  # Loop over queries
            hits = {}
            # for e in range(len(blast_ids)): # Loop over BLAST entries
            for idx, e in blast_df.loc[blast_df["query"] == q].iterrows():
                if e["score"] >= min_score:
                    raw_id = e["ID"].split("|")
                    pdb_index = raw_id.index("pdb")
                    pdb_id = raw_id[pdb_index + 1]

                    hits[pdb_id] = {}
                    hits[pdb_id]["percent_identity"] = e["score"]
                    hits[pdb_id]["sequence"] = e["sequence"]

            match_hits.append(hits)

        return match_hits

    @staticmethod
    def request_rcsb_pdbid(pdb_id):
        """A function to request a protein entry from the rcsb API with a pdb ID"""
        requestURL = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        r = requests.get(requestURL, headers={"Accept": "application/json"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        return r.json()

    @classmethod
    def check_resolution(self, pdb_id):
        fjson = self.request_rcsb_pdbid(pdb_id)
        return fjson["rcsb_entry_info"]["resolution_combined"][0]

    @classmethod
    def choose_best_pdb_entry(self, hits, best_match_percent):
        """If a sequence have different PDB files all with the same alignment match, we choose and retrieve the one with the highest resolution"""
        max_res = 10  # some unrealistically large number?
        best_pdb_record = ""
        for h in hits:
            id_match = hits[h]["percent_identity"]
            if id_match == best_match_percent:
                res = self.check_resolution(h)
                if res < max_res:
                    best_pdb_record = h
                    max_res = res
            else:  # We're only interested in the entries with the max possible alignment
                break
        print(
            f"The best PDB entry is {best_pdb_record}, with match {best_match_percent}% and res {max_res}A"
        )
        return best_pdb_record
