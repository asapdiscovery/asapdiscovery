from pathlib import Path
from unittest import mock

from asapdiscovery.simulation.szybki import (
    SzybkiFreeformConformerAnalyzer,
    SzybkiFreeformResult,
)


# this is a super cheating but it's the only way to test this without a lot of mocking of OESzybki internals
def test_szybki(ligand_path, szybki_results, tmp_path):
    def run_szybki_on_ligand_patch(
        self, ligand_path: Path, output_path: Path
    ) -> SzybkiFreeformResult:
        return szybki_results

    sk = SzybkiFreeformConformerAnalyzer([ligand_path], [tmp_path])
    with mock.patch.object(
        SzybkiFreeformConformerAnalyzer,
        "run_szybki_on_ligand",
        run_szybki_on_ligand_patch,
    ):
        res = sk.run_all_szybki()
        assert res[0] == szybki_results
