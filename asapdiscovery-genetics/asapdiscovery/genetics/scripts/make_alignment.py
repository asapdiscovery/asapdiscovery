from asapdiscovery.genetics.blast import get_blast_seqs, PDBentry
from asapdiscovery.genetics.seq_alignment import Alignment, do_MSA

def make_alignment(fasta_input, results_folder, naligns=1000, 
                   select_mode="human", align_output="", plot_width=1500,
                   save_blast="blast.csv"):

    # Perform BLAST search on input sequence
    matches_df = get_blast_seqs(fasta_input, results_folder, input_type="fasta", 
                                save_csv=save_blast,
                                nalign=naligns, nhits=naligns*3/4,
                                database="refseq_protein", verbose=False)
    
    # Perform alignment for each entry in the FASTA file
    for query in matches_df['query'].unique():
        alignment = Alignment(matches_df, query, results_folder)
        file_prefix = f"{align_output}{alignment.query_label}"
        selection_fasta, plot = do_MSA(alignment, select_mode, file_prefix, plot_width)

        # Generate PDB file for template (only for the reference structure)
        pdb_entry = PDBentry(selection_fasta, "fasta")
        pdb_file_record = pdb_entry.retrieve_pdb(results_folder=results_folder, min_id_match=99.9, ref_only=True)

        record = pdb_file_record[0]
        print(f"A PDB template for {record.label} was saved as {record.pdb_file}")

    return 

#make_alignment("../mers-mpro.aln", "../results", naligns=1000, 
#               select_mode="human", align_output="", plot_width=1500,
#               save_blast="blast.csv")
