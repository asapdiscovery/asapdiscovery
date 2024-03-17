import pandas as pd
import requests
from pathlib import Path
import subprocess

# BioPython
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.PDB import PDBList

from pydantic import BaseModel, Field

_E_VALUE_THRESH = 1e-20


def parse_blast(results_file: str, verbose: bool = False) -> pd.DataFrame:
    """Parse data from BLAST xml file

    Parameters
    ----------
    results_file : str
        Path to BLAST results
    verbose : bool, optional
        Whether to print information, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with BLAST entries
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
    save_folder: Path,
    input_type="fasta",
    nhits=100,
    nalign=500,
    database="refseq_protein",
    xml_file="results.xml",
    verbose=True,
    save_csv=None,
    ) -> pd.DataFrame:
    """Run a BLAST search on a protein sequence.

    Parameters
    ----------
    seq_source : str
        Source with the sequence.
    save_folder : Path
        Path to folder to save BLAST results
    input_type : str, optional
        Type of sequence source ["pre-cal", "fasta", "sequence"], by default "fasta"
    nhits : int, optional
        Number of hits, hitlist_size parameter in BLAST, by default 100
    nalign : int, optional
        Number of alignments, alignments parameter in BLAST, by default 500
    database : str, optional
        Name of BLAST database, by default "refseq_protein"
    xml_file : str, optional
        Name to be given to XML with BLAST results, by default "results.xml"
    verbose : bool, optional
        Whether to print info on BLAST search, by default True
    save_csv : Union[str, None], optional
        CSV file name to optionally save dataframe, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame with Blast results.
    """

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
        "blastp", database, sequence, hitlist_size=nhits, alignments=nalign
    )

    save_file = save_folder / xml_file 

    with open(save_file, "w") as file:
        blast_results = result_handle.read()
        file.write(blast_results)

    matches_df = parse_blast(save_file, verbose)
    if save_csv:
        matches_df.to_csv(save_folder / save_csv, index=False)

    return matches_df

class PDBRecord(BaseModel):
    label: str = Field(description="RefID label of the Blast PDB record")
    description: str = Field(description="Description of the Blast record")
    seq: str = Field(description="Sequence")
    chain: int = Field(description="Chain identifier")
    pdb_id: str = Field("", description="The pdb ID of entry")
    pdb_file: str = Field("", description="The PDB file Path that was saved")


class PDBEntry(BaseModel):
    seq: str = Field(description="Input with the protein sequence")
    type: str = Field(description="Type of input")

    def retrieve_pdb(
            self, 
            results_folder: Path, 
            min_id_match=99, 
            ref_only=False,
            ):#->list[PDBRecord]:
        """Retrieve the PDB record of a given sequence.

        Parameters
        ----------
        results_folder : Path
            Path store results
        min_id_match : int, optional
            Minimum match score (in %) to use as criteria to filter PDB blast entries, by default 99
        ref_only : bool, optional
            Whether to save the PDBs of only the reference seq or for all BLAST selections, by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        import prody

        record_name = "pdb_blast.xml"

        # Find pdb matching given sequence
        matches_df = get_blast_seqs(
            self.seq,
            results_folder,
            input_type=self.type,
            xml_file=record_name,
            nalign=500,
            database="pdb",
            verbose=False,
        )
        print(f"Saving blast results in {results_folder / record_name}")

        # Load original sequence names and descriptors for reference
        pdb_file_record = []
        for seq_record in SeqIO.parse(self.seq, self.type):
            seq_id = seq_record.id
            seq_description = seq_record.description
            seq_chain = int(seq_record.id.split("|")[1].split(".")[-1])
            pdb_file_record.append(
                PDBRecord(label=seq_id, 
                          description=seq_description, 
                          seq=str(seq_record.seq),
                          chain=seq_chain)
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

                # pdbl = PDBList()
                # filename = pdbl.retrieve_pdb_file(best_pdb_record.lower(), file_format="pdb", pdir=results_folder)
                filename = prody.fetchPDB(best_pdb_record, folder=str(results_folder))
                subprocess.run(["gunzip", filename])
                pdb_file_record[i].pdb_file = filename
                pdb_file_record[i].pdb_id = best_pdb_record
                # If I'm only requesting the reference PDB, there's no need to generate more files.
                if ref_only:
                    break

        return pdb_file_record

    @staticmethod
    def parse_pdb_blast_results(blast_df: pd.DataFrame, min_score: int)->list[dict]:
        """For a pdb database search extract pdb ids with a minimum score.
        Outputs are lists because iterables are needed for functions down the pipeline.

        Parameters
        ----------
        blast_df : pd.DataFrame
            DataFrame with BLAST results
        min_score : int
            The minimum match score (in %) that a BLAST entry must have w.r.t the reference

        Returns
        -------
        list[dict]
            Blast hits for each query, where data on each hit can be accessed through its PDB ID.
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
        return r.json()

    @classmethod
    def check_resolution(self, pdb_id):
        fjson = self.request_rcsb_pdbid(pdb_id)
        return fjson["rcsb_entry_info"]["resolution_combined"][0]

    @classmethod
    def choose_best_pdb_entry(self, hits, best_match_percent):
        """If a sequence have different PDB files all with the same alignment match, we choose and retrieve the one with the highest resolution"""
        max_res = 100  # some unrealistically large number?
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
