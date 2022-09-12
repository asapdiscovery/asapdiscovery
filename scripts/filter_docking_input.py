import csv
from openeye import oechem

def filter_docking_inputs(smarts_queries, docking_inputs, output_file,
							ignore_comment=False,
							verbose=True):
	"""
    Filter an input file of compound SMILES by SMARTS filters using OEchem matching.

    Parameters
    ----------
    smarts_queries : str
        Path to file containing SMARTS entries to filter by (comma-separated).
    docking_inputs : str
    	Path to file containing SMILES entries to filter using smarts_queries.
    output_file : str
    	Path to output file containing filtered SMILES entries in equal format to 
    	docking_inputs file.
    ignore_comment : bool
    	How to handle first-character hashtags on SMARTS entries. True overrides them,
    	False sets these entries to be ignored during filtering.
    verbose : bool
    	Whether or not to print a message stating the number of compounds filtered.

	"""
	with open(smarts_queries, "r") as queries, \
		open(docking_inputs, "r") as inputs:
		queries_reader = csv.reader(queries)
		inputs_reader = csv.reader(inputs)

		query_smarts = [ q[0] for q in queries_reader ]

		if ignore_comment:
			# only keep smarts queries that are not commented.
			query_smarts = [ q for q in query_smarts if not q[0] == "#"]
		else:
			# some of the SMARTS queries are commented - remove these for now.
			query_smarts = [ q if not q[0] == "#" else q[1:] for q in query_smarts]

		input_cpds = []
		num_input_cpds = 0 # initiate counter for verbose setting.
		output_cpds = []
		for cpd in inputs_reader:
			num_input_cpds += 1
			# read input cpd into OE.
			mol = oechem.OEGraphMol()
			oechem.OESmilesToMol(mol, cpd[0].split()[0])

			# now loop over queried SMARTS patterns, flag input compound if hit.
			for query in query_smarts:
				# create a substructure search object.
				ss = oechem.OESubSearch(query)
				oechem.OEPrepareSearch(mol, ss)	

				# compare this query to the reference mol.	
				if ss.SingleMatch(mol):
					# if match is found we can stop querying and output the cpd.
					output_cpds.append(cpd)
					break

	# with the list of filtered compounds write out the new docking inputs file.
	with open(output_file, "w") as writefile:
		writer = csv.writer(writefile)
		for output_cpd in output_cpds:
			writer.writerow(output_cpd)

	if verbose:
		print(f"Retained {len(output_cpds)/num_input_cpds*100:.2f}% of compounds after " \
			+f"filtering ({len(query_smarts)} SMARTS filter(s); {num_input_cpds}-->" \
			+f"{len(output_cpds)}). Wrote output to {output_file}.")


if __name__ == "__main__":
	filter_docking_inputs(
		"tmp_filter_inputs/smarts_queries.csv",
		"tmp_filter_inputs/docking_inputs.csv",
		"tmp_filter_inputs/docking_inputs_filtered.csv",
		ignore_comment=True,
		verbose=True)



"""
- I/O how intended? files at all times?
- delimiters? fixed format or make flexible?
- comments on SMARTS strings? 
- Filter is remove or include? or either?
"""