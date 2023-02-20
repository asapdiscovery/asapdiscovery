### Storage of exported experimental data on COVID Moonshot compounds.

Files:
- `CDD CSV Export - 2023-02-20 00h18m18s.csv` : raw export from CDD vault containing all data points (n=2826)

- `clean_cdd_export.py` : script to clean CDD data 

- `clean_cdd_export.out` : stdout captured from above python script

- `fullseries_clean.csv` : cleaned dataset with computed pIC50s and free energies. For each compound, we selected the fluorescence measurement with the highest quality data (lowest curve class corresponding to highest reliability, and then the smallest confidence interval if multiple measurements had the same curve class). This yielded the single most reliable measurement. The CDD for to IC50 and 95% CI for this measurement was used. (n=1177)

- `fullseries_bulky.csv` : also included compounds for which IC50s were >99 (artifically set to 100), and compounds where fields were missing (e.g. postera ID) (n=2494)

Dependencies for running this script to regenerate output files:
- Pandas
- Numpy 
