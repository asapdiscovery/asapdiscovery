import base64
import warnings
from typing import Literal, Optional

import bokeh.models.widgets.tables
import bokeh.palettes
import bokeh.plotting
import cinnabar
import numpy as np
import pandas as pd
import panel
import plotmol
from bokeh.models import Band, ColumnDataSource, Range1d, Whisker
from cinnabar import stats
from openff.units import unit
from rdkit import Chem
from rdkit.Chem import Draw

# run to enable plotting with bokeh
panel.extension()


def get_magnitude(quantity: unit.Quantity) -> float:
    """
    Convenience method to return the magnitude of an OpenFF unit quantity.
    This is useful in cases where data needs to be handled by libraries that
    are not able to parse pint-like quantities.

    Args:
        quantity: The value to be transformed.

    Returns:
        magnitude: The magnitude computed from the quantity.
    """
    if type(quantity) is float:
        magnitude = quantity
    else:
        magnitude = quantity.magnitude

    return magnitude


def negative_log(value: float, inverse: bool = False) -> float:
    """
    Convenience method to take the negative log of value, or to
    invert it. Handy for round tripping e.g. pIC50 and IC50.

    Args:
        value: The value to be transformed
        inverse: If inverse take the negative natural log of the value

    Returns:
        transformed_value: The transformed value.
    """
    if not inverse:
        transformed_value = 10 ** (-1 * value)
    else:
        transformed_value = -1 * np.log(value)

    return transformed_value


def ki_to_dg(
    ki: unit.Quantity,
    uncertainty: unit.Quantity,
    temperature: unit.Quantity = 298.15 * unit.kelvin,
) -> tuple[unit.Quantity, unit.Quantity]:
    """
    Convenience method to convert a Ki w/ a given uncertainty to an
    experimental estimate of the binding free energy.

    Args:
        ki: Experimental Ki value (e.g. 5 * unit.nanomolar)
        uncertainty: Experimental error. Note: returns 0 if =< 0 * unit.nanomolar.
        temperature: Experimental temperature. Default: 298.15 * unit.kelvin.

    Returns:
        dg: Gibbs binding free energy.
        ddg: Error in binding free energy.
    """
    if ki > 1e-15 * unit.nanomolar:
        dg = (
            unit.molar_gas_constant
            * temperature.to(unit.kelvin)
            * np.log(ki / unit.molar)
        ).to(unit.kilocalorie_per_mole)
    else:
        raise ValueError("negative Ki values are not supported")
    # propagate the uncertainty <https://en.wikipedia.org/wiki/Propagation_of_uncertainty>
    if uncertainty > 0 * unit.molar:
        ddg = (
            unit.molar_gas_constant * temperature.to(unit.kelvin) * uncertainty / ki
        ).to(unit.kilocalorie_per_mole)
    else:
        ddg = 0 * unit.kilocalorie_per_mole

    return dg, ddg


def pic50_to_dg(
    pic50: float,
    uncertainty: float,
    temperature: unit.Quantity = 298.15 * unit.kelvin,
) -> tuple[unit.Quantity, unit.Quantity]:
    """
    Convert the PIC50 value and its uncertainty to dg.

    Notes:
        We use a different function for this due to the slightly different error propagation formula.
        uncertainty calculated using:
        G = RTLn(Ki)
        Ki = 10^(-pK)
        sigma(G) = dG/dKi * sigma(Ki)
        sigma(G) = sigma(pK) * RT * d Ln(10^-pK) / dpK
        sigma(G) = sigma(pK) * RT * Ln(10)

    Args:
        pic50: The pIC50 value.
        uncertainty: The standard deviation in the pIC50 value
        temperature: Experimental temperature. Default: 298.15 * unit.kelvin.

    Returns:
        dg: Gibbs binding free energy.
        ddg: Error in binding free energy.
    """
    ki = negative_log(pic50) * unit.molar
    dg = (
        unit.molar_gas_constant * temperature.to(unit.kelvin) * np.log(ki / unit.molar)
    ).to(unit.kilocalorie_per_mole)
    ddg = (
        unit.molar_gas_constant * temperature.to(unit.kelvin) * np.log(10) * uncertainty
    ).to(unit.kilocalorie_per_mole)
    return dg, ddg


def shift_and_add_prediction_error(df: pd.DataFrame, point_type: str) -> pd.DataFrame:
    """
    Wrangles a cinnabar `FEMap` DataFrame and returns a DF ready for plotting. In case of absolute
    (`point_type`=='DG'), shifts the prediction values to the mean experimental value.

    Args:
        df: The dataframe we want to shift and add the prediction error to.
        point_type: Whether the points are absolute or relative. Can be "DG" or "DDG", resp.
    """
    if point_type == "DG":
        # shift the FECS predicted values to the mean of the experimental values.
        df[f"{point_type} (kcal/mol) (FECS)"] = (
            df[f"{point_type} (kcal/mol) (FECS)"]
            - df[f"{point_type} (kcal/mol) (FECS)"].mean()
            + df[f"{point_type} (kcal/mol) (EXPT)"].mean()
        )

    # calculate the prediction error.
    df["prediction error (kcal/mol)"] = abs(
        df[f"{point_type} (kcal/mol) (FECS)"] - df[f"{point_type} (kcal/mol) (EXPT)"]
    )
    return df


def dG_to_pIC50(dG):
    """
    Converts a DG kcal/mol unit to a pIC50 value.
    Note: uses Ki = pIC50 approximation (assumption). May need to implement a Cheng-Prusoff version.
    """
    kT = 0.593  # kcal/mol for 298 K (25C)

    # NOTE: SHOULD WE BE TAKING ABS HERE? IS THIS ACCURATE?
    return abs(
        dG / np.log(10.0) / (-kT)
    )  # abs to prevent pIC50s from being negative in cases where 0<DG<1.


def dg_to_postera_dataframe(absolute_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Given a wrangled `FEMap` dataframe of absolute predicted DG values (kcal/mol), replace 'kcal/mol' columns with
    pIC50 values and rename the columns to match what's expected by manifold. These names are defined in
    data.services.postera.manifold_data_tags.yaml

    Args:
        absolute_predictions: The dataframe of absolute DG predictions from asap-alchemy

    Returns:
        A copy of the dataframe with calculated pIC50 values rather than DGs ready for upload to postera.
    """
    # use the expected column names from alchemy predict and convert them into the allowed column names defined in
    # data.services.postera.manifold_data_tags.yaml
    postera_df = absolute_predictions.copy(deep=True)

    for column, new_name in [
        ("DG (kcal/mol) (FECS)", "FEC"),
        ("uncertainty (kcal/mol) (FECS)", "FEC-uncertainty"),
    ]:
        # replace the kcal/mol values with pIC50s.
        postera_df[column] = dG_to_pIC50(postera_df[column].values)

        # rename the column
        postera_df.rename(columns={column: f"computed-{new_name}-pIC50"}, inplace=True)
    # rename the label column to be clear in postera
    postera_df.rename(columns={"label": "Ligand_ID"}, inplace=True)

    return postera_df


def add_identifiers_to_df(dataframe: pd.DataFrame, ligands: list) -> pd.DataFrame:
    """
    Given a wrangled DF containing either `label` (absolute) or `labelA` and `labelB` (relative),
    add molecule identifiers for each row.

    Args:
        dataframe: The pandas dataframe we should add the smiles to.
        ligands: The list of ligands from which we should extract the identifiers based on matching by name.
    Returns:
        The dataframe with added column(s) with molecule identifiers

    Notes:
        Currently adds the smiles and inchi key
    """
    # get the identifiers for each ligand by name
    ligands_by_name = {
        ligand.compound_name: {"smiles": ligand.smiles, "inchi_key": ligand.inchikey}
        for ligand in ligands
    }

    # get the identifiers. Relative dataframe is a bit more involved than absolute.
    if "labelA" in dataframe.columns:
        smiles_a, smiles_b, inchi_key_a, inchi_key_b = [], [], [], []
        for labelA, labelB in dataframe[["labelA", "labelB"]].values:
            smiles_a.append(ligands_by_name[labelA]["smiles"])
            smiles_b.append(ligands_by_name[labelB]["smiles"])
            inchi_key_a.append(ligands_by_name[labelA]["inchi_key"])
            inchi_key_b.append(ligands_by_name[labelB]["inchi_key"])
        dataframe["SMILES_A"] = smiles_a
        dataframe["SMILES_B"] = smiles_b
        dataframe["Inchi_Key_A"] = inchi_key_a
        dataframe["Inchi_Key_B"] = inchi_key_b

    elif "label" in dataframe.columns:
        smiles = [
            ligands_by_name[label]["smiles"] for label in dataframe["label"].values
        ]
        inchi_key = [
            ligands_by_name[label]["inchi_key"] for label in dataframe["label"].values
        ]
        dataframe["SMILES"] = smiles
        dataframe["Inchi_Key"] = inchi_key

    return dataframe


def extract_experimental_data(
    reference_csv: str, assay_units: Literal["pIC50", "IC50"]
) -> pd.DataFrame:
    """
    Extract the experimental data from the given csv file, this assumes the csv has been downloaded from cdd.
    Where the molecule identifier is under column 'Molecule Name' and the experimental data is pIC50 / IC50
    TODO make more general

    Args:
        reference_csv: The name of the csv file with the experimental data
        assay_units: The assay units of 'pIC50' or 'IC50' that the experimental data is given in.

    Returns:
        A pandas dataframe of the reference data with added columns containing the calculated binding affinity and
        its associated uncertainty converted to Gibbs free energy in kcal / mol.
    """
    assay_tags = {
        "pIC50": ("pIC50_Mean", "pIC50_Mean Standard Deviation (±)"),
        "IC50": ("IC50_GMean (µM)", "IC50_GMean (µM) Standard Deviation (×/÷)"),
    }
    exp_data = pd.read_csv(reference_csv).fillna(0)

    # work out the columns for the ref data and the uncertainty
    assay_endpoint_tag, assay_endpoint_confidence_tag = None, None
    for col in exp_data.columns:
        if col.endswith(assay_tags[assay_units][0]):
            assay_endpoint_tag = col
        elif col.endswith(assay_tags[assay_units][1]):
            assay_endpoint_confidence_tag = col
    if assay_endpoint_tag is None:
        raise RuntimeError(
            f"Could not determine the assay tag from the provided units {assay_units}."
        )
    if assay_endpoint_confidence_tag is None:
        warnings.warn(
            f"Failed to detect Standard Deviation in experimental reference file {reference_csv}."
        )

    if assay_units == "pIC50":
        converter = pic50_to_dg
        units = unit.dimensionless
    # CDD IC50 uses micromolar units
    else:
        converter = ki_to_dg
        units = unit.micromolar

    experimental_affinity, experimental_affinity_error = [], []
    # add the calculated affinity and error to the dataframe
    for _, row in exp_data.iterrows():
        exp_value = row[assay_endpoint_tag]
        if assay_endpoint_confidence_tag:
            uncertainty = row[assay_endpoint_confidence_tag]
        else:
            uncertainty = 0

        dg, ddg = converter(exp_value * units, uncertainty * units)
        experimental_affinity.append(dg.m)
        experimental_affinity_error.append(ddg.m)

    # the column names here match those made by `parse_fluorescence_data_cdd`
    exp_data["exp_binding_affinity_kcal_mol"] = experimental_affinity
    exp_data["exp_binding_affinity_kcal_mol_stderr"] = experimental_affinity_error
    return exp_data


def _find_ligand_data(
    name: str, inchi_key: str, experimental_data: pd.DataFrame
) -> dict:
    """
    Multi search method which tries to match the name then the inchi key when looking for a molecules experimental data.

    Notes:
        Columns should have names `Molecule Name` and `Inchi Key`, if the molecule can not be found dummy data
        is returned.
        We use InchiKey when matching as this is atom order and cheminformatics toolkit independent.

    Args:
        name: The name which should be used to match the molecules
        inchi_key: The inchi key which should be used to match the molecules
        experimental_data: The experimental dataframe which should be searched for the target molecule

    Returns:
        A dictionary of the data from the dataframe which matches the provided name or inchi key
    """

    ligand_data = experimental_data[experimental_data["Molecule Name"] == name]
    if len(ligand_data) < 1:
        ligand_data = experimental_data[experimental_data["Inchi Key"] == inchi_key]
    if len(ligand_data) == 1:
        ligand_data = ligand_data.iloc[0]
    else:
        # dummy data if not found easy to drop later
        ligand_data = {
            "exp_binding_affinity_kcal_mol": np.nan,
            "exp_binding_affinity_kcal_mol_stderr": np.nan,
        }

    return ligand_data


def add_absolute_expt(
    dataframe: pd.DataFrame,
    experimental_data: pd.DataFrame,
):
    """
    Edit the dataframe inplace by adding experimental data provided to it.

    Args:
        dataframe: The dataframe of absolute free energy predictions to add the experimental data to, this should
            contain the name, smiles and inchi key of the molecules.
        experimental_data: A dataframe containing the experimental free energies in units of kcal/mol to add to the
            dataframe.
    """
    experimental_col, uncertainty_col = [], []
    for _, row in dataframe.iterrows():
        # use the two stage lookup
        ligand_data = _find_ligand_data(
            name=row["label"],
            inchi_key=row["Inchi_Key"],
            experimental_data=experimental_data,
        )
        experimental_col.append(ligand_data["exp_binding_affinity_kcal_mol"])
        uncertainty_col.append(ligand_data["exp_binding_affinity_kcal_mol_stderr"])
    dataframe["DG (kcal/mol) (EXPT)"] = experimental_col
    dataframe["uncertainty (kcal/mol) (EXPT)"] = uncertainty_col


def add_relative_expt(
    dataframe: pd.DataFrame,
    experimental_data: pd.DataFrame,
):
    """
    Edit the relative dataframe in place by adding experimental data provided to it.

    Args:
        dataframe: The dataframe of relative free energy predictions to add the experimental data to.
        experimental_data: A dictionary of experimental free energies in units of kcal/mol to add to the dataframe.
    """
    experimental_col, uncertainty_col = [], []
    for _, row in dataframe.iterrows():
        ligand_a_data = _find_ligand_data(
            name=row["labelA"],
            inchi_key=row["Inchi_Key_A"],
            experimental_data=experimental_data,
        )
        ligand_b_data = _find_ligand_data(
            name=row["labelB"],
            inchi_key=row["Inchi_Key_B"],
            experimental_data=experimental_data,
        )

        # compute experimental DDG for this edge
        ddg = (
            ligand_b_data["exp_binding_affinity_kcal_mol"]
            - ligand_a_data["exp_binding_affinity_kcal_mol"]
        )
        # take the average uncertainty between measurements for this edge.
        delta_ddg = np.mean(
            [
                ligand_a_data["exp_binding_affinity_kcal_mol_stderr"],
                ligand_b_data["exp_binding_affinity_kcal_mol_stderr"],
            ]
        )
        experimental_col.append(ddg)
        uncertainty_col.append(delta_ddg)

    dataframe["DDG (kcal/mol) (EXPT)"] = experimental_col
    dataframe["uncertainty (kcal/mol) (EXPT)"] = uncertainty_col


def get_data_from_femap(
    fe_map: cinnabar.FEMap,
    ligands: list,
    assay_units: Optional[str] = None,
    reference_dataset: Optional[str] = None,
    cdd_protocol: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a `cinnabar` `FEMap` add the experimental reference data and generate and return:
    1. a Pandas DataFrame that has all absolute predictions and measurements (DG in kcal/mol) and (pIC50)
    2. a pd DF that has all relative predictions and measurements (DDG in kcal/mol) and PIC50

    Args:
        fe_map: The cinnabar FEMap which has all calculated edges present and the absolute estimates.
        ligands: The list of asap ligands which are part of the network.
        assay_units: The units of the experimental data, which should be extracted from the reference dataset.
        reference_dataset: The name of the cdd csv file which contains the experimental data.
        cdd_protocol: The name of the CDD protocol from which we should extract experimental data.

    Returns:
         An absolute and relative free energy prediction dataframe.
    """
    # extract the dataframes from cinnabar and format
    absolute_df = (
        fe_map.get_absolute_dataframe()
        .rename(
            columns={
                "DG (kcal/mol)": "DG (kcal/mol) (FECS)",
                "uncertainty (kcal/mol)": "uncertainty (kcal/mol) (FECS)",
            }
        )
        .drop(["source", "computational"], axis=1)
    )
    relative_df = (
        fe_map.get_relative_dataframe()
        .rename(
            columns={
                "DDG (kcal/mol)": "DDG (kcal/mol) (FECS)",
                "uncertainty (kcal/mol)": "uncertainty (kcal/mol) (FECS)",
            }
        )
        .drop(["source", "computational"], axis=1)
    )

    # add identifiers to each dataframe these help with matching if names fail
    for df in [absolute_df, relative_df]:
        add_identifiers_to_df(df, ligands)

    # add experimental data if available
    experimental_data = None
    if reference_dataset:
        experimental_data = extract_experimental_data(
            reference_csv=reference_dataset, assay_units=assay_units
        )
    elif cdd_protocol:
        experimental_data = download_cdd_data(protocol_name=cdd_protocol)

    if experimental_data is not None:
        add_absolute_expt(dataframe=absolute_df, experimental_data=experimental_data)
        add_relative_expt(dataframe=relative_df, experimental_data=experimental_data)

        absolute_df = shift_and_add_prediction_error(df=absolute_df, point_type="DG")
        relative_df = shift_and_add_prediction_error(df=relative_df, point_type="DDG")

    # now also add the calculated pCI50 absolute dataframe which is used .
    # absolute_df = dg_to_pic50_dataframe(absolute_df)
    # relative_df_wrangled_pic50 = dg_to_pic50_dataframe(relative_df_wrangled)

    return absolute_df, relative_df


def draw_mol(smiles: str) -> str:
    """
    Create SVG text of a 2D depiction of a molecule which can be embed in an html report.

    Args:
        smiles: The smiles of the molecule which should be drawn.

    Returns:
        The SVG text of the drawing.

    Notes:
        This function will draw multi molecules side by side.
    """
    rdkit_mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    rdkit_mol = Draw.PrepareMolForDrawing(rdkit_mol, forceCoords=True)
    mols = Chem.GetMolFrags(rdkit_mol, asMols=True)
    if len(mols) == 2:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400, 200, 200, 200)
        drawer.DrawMolecules(mols)
    else:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400, 200)
        drawer.DrawMolecule(rdkit_mol)
    drawer.FinishDrawing()

    data = base64.b64encode(drawer.GetDrawingText().encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}"></img>'


def plotmol_absolute(
    calculated: np.array,
    experimental: np.array,
    smiles: list[str],
    titles: Optional[list[str]] = None,
    calculated_uncertainty: Optional[np.array] = None,
    experimental_uncertainty: Optional[np.array] = None,
) -> bokeh.plotting.figure:
    """
    Create an interactive plot using Plotmol for the absolute predictions of the free energies.

    Args:
        calculated: An array of calculated predictions.
        experimental: An array of experimental predictions in order of the calculated values.
        smiles: A list of smiles strings in order of the calculated values.
        titles: A list of titles in order of the calculated values.
        calculated_uncertainty: An optional array of calculated uncertainty values.
        experimental_uncertainty: An optional array of experimental uncertainty values.

    Returns:
        The bokeh interactive plot.
    """

    # set up our own tooltip to show the title and other info
    custom_tooltip_template = """
    <div>
        <div>
            <span>@title</span>
            <span><br>experimental:@experimental (kcal/mol)<br>calculated:@prediction (kcal/mol)</span>
            <img src="@image" ></img>
        </div>
    </div>
    """

    figure = bokeh.plotting.figure(
        tooltips=custom_tooltip_template,
        title="Predicted affinity",
        x_axis_label="Experimental ΔG (kcal / mol)",
        y_axis_label="Calculated ΔG (kcal / mol)",
        width=800,
        height=800,
    )
    # create the tooltip data needed for this plot type
    tooltip_data = {
        "title": titles,
        "experimental": experimental,
        "prediction": calculated,
    }
    return _plot_with_plotmol(
        figure=figure,
        calculated=calculated,
        experimental=experimental,
        smiles=smiles,
        calculated_uncertainty=calculated_uncertainty,
        experimental_uncertainty=experimental_uncertainty,
        custom_column_data=tooltip_data,
    )


def _plot_with_plotmol(
    figure: bokeh.plotting.figure,
    calculated: np.array,
    experimental: np.array,
    smiles: list[str],
    custom_column_data: dict[str, list],
    calculated_uncertainty: Optional[np.array] = None,
    experimental_uncertainty: Optional[np.array] = None,
) -> bokeh.plotting.figure:
    # set up the x and y range and plot x=y line
    axis_padding = 0.5
    ax_min = min(min(experimental), min(calculated)) - axis_padding
    ax_max = max(max(experimental), max(calculated)) + axis_padding
    scale = Range1d(ax_min, ax_max)
    figure.x_range = scale
    figure.y_range = scale
    figure.line(
        [ax_min, ax_max], [ax_min, ax_max], line_color="black", line_dash="dashed"
    )
    band_data = ColumnDataSource(
        {
            "x": [ax_min, ax_max],
            "upper_in": [ax_min + 0.5, ax_max + 0.5],
            "upper_out": [ax_min + 1, ax_max + 1],
            "lower_in": [ax_min - 0.5, ax_max - 0.5],
            "lower_out": [ax_min - 1, ax_max - 1],
        }
    )
    in_band = Band(
        base="x",
        upper="upper_in",
        lower="lower_in",
        source=band_data,
        fill_alpha=0.2,
        fill_color="grey",
    )
    outer_band = Band(
        base="x",
        upper="upper_out",
        lower="lower_out",
        source=band_data,
        fill_alpha=0.2,
        fill_color="grey",
    )
    palette = bokeh.palettes.Spectral4
    figure.add_layout(in_band)
    figure.add_layout(outer_band)

    # add error bars if provided
    if calculated_uncertainty is not None:
        calc_error_data = ColumnDataSource(
            {
                "x": experimental,
                "upper": calculated + calculated_uncertainty,
                "lower": calculated - calculated_uncertainty,
            }
        )
        calc_error = Whisker(
            base="x", upper="upper", lower="lower", source=calc_error_data
        )
        figure.add_layout(calc_error)
    if experimental_uncertainty is not None:
        exp_error_data = ColumnDataSource(
            {
                "x": calculated,
                "upper": experimental + experimental_uncertainty,
                "lower": experimental - experimental_uncertainty,
            }
        )
        exp_error = Whisker(
            base="x",
            upper="upper",
            lower="lower",
            source=exp_error_data,
            dimension="width",
        )
        figure.add_layout(exp_error)

    plotmol.scatter(
        figure=figure,
        x=experimental,
        y=calculated,
        smiles=smiles,
        marker="o",
        marker_size=15,
        marker_color=palette[0],
        custom_column_data=custom_column_data,
    )
    figure.axis.axis_label_text_font_size = "20pt"
    return figure


def plotmol_relative(
    calculated: np.array,
    experimental: np.array,
    smiles: list[str],
    titles: Optional[list[tuple[str, str]]] = None,
    calculated_uncertainty: Optional[np.array] = None,
    experimental_uncertainty: Optional[np.array] = None,
) -> bokeh.plotting.figure:
    """
    Create an interactive plot using Plotmol for the relative predictions of the free energies.

    Args:
        calculated: An array of calculated predictions.
        experimental: An array of experimental predictions in order of the calculated values.
        smiles: A list of smiles strings in order of the calculated values.
        titles: A list of titles in order of the calculated values.
        calculated_uncertainty: An optional array of calculated uncertainty values.
        experimental_uncertainty: An optional array of experimental uncertainty values.

    Returns:
        The bokeh interactive plot.
    """
    # set up our own tooltip to show the title and other info
    custom_tooltip_template = """
        <div>
            <div>
                <span>StateA:@titleA → StateB:@titleB</span>
                <span><br>experimental:@experimental (kcal/mol)<br>calculated:@prediction (kcal/mol)</span>
                <img src="@image" ></img>
            </div>
        </div>
        """

    figure = bokeh.plotting.figure(
        tooltips=custom_tooltip_template,
        title="Relative prediction",
        x_axis_label="Experimental ΔΔG (kcal / mol)",
        y_axis_label="Calculated ΔΔG (kcal / mol)",
        width=800,
        height=800,
    )
    titles_a, titles_b = [], []
    for title_a, title_b in titles:
        titles_a.append(title_a)
        titles_b.append(title_b)
    # create the tooltip data needed for this plot type
    tooltip_data = {
        "titleA": titles_a,
        "titleB": titles_b,
        "experimental": experimental,
        "prediction": calculated,
    }
    return _plot_with_plotmol(
        figure=figure,
        calculated=calculated,
        experimental=experimental,
        smiles=smiles,
        calculated_uncertainty=calculated_uncertainty,
        experimental_uncertainty=experimental_uncertainty,
        custom_column_data=tooltip_data,
    )


def create_absolute_report(dataframe: pd.DataFrame) -> panel.Column:
    """
    Create a cinnabar style interactive report for the dataframe of absolute free energy predictions, this dataframe
    should include experimental data already.

    Args:
        dataframe: The dataframe of absolute predictions to construct the report for.

    Returns:
        A panel column containing an interactive plot and table of the free energy predictions.

    Notes:
        Only plots molecules with experimental values
    """
    # create a plotting dataframe which drops rows with nans
    plotting_df = dataframe.dropna(axis=0, inplace=False, ignore_index=True)
    # add drawn molecule as a column
    mols = [draw_mol(smiles) for smiles in dataframe["SMILES"]]
    dataframe["Molecule"] = mols
    # create the DG plot
    fig = plotmol_absolute(
        calculated=plotting_df["DG (kcal/mol) (FECS)"],
        experimental=plotting_df["DG (kcal/mol) (EXPT)"],
        smiles=plotting_df["SMILES"],
        titles=plotting_df["label"],
        calculated_uncertainty=plotting_df["uncertainty (kcal/mol) (FECS)"],
        experimental_uncertainty=plotting_df["uncertainty (kcal/mol) (EXPT)"],
    )
    # calculate the bootstrapped stats using cinnabar
    stats_data = []
    for statistic in ["RMSE", "MUE", "R2", "rho"]:
        s = stats.bootstrap_statistic(
            plotting_df["DG (kcal/mol) (EXPT)"],
            plotting_df["DG (kcal/mol) (FECS)"],
            plotting_df["uncertainty (kcal/mol) (EXPT)"],
            plotting_df["uncertainty (kcal/mol) (FECS)"],
            statistic=statistic,
            include_true_uncertainty=False,
            include_pred_uncertainty=False,
        )
        stats_data.append(
            {
                "Statistic": statistic,
                "value": s["mle"],
                "lower bound": s["low"],
                "upper bound": s["high"],
            }
        )
    stats_df = pd.DataFrame(stats_data)
    # create a format for numerical data in the tables
    number_format = bokeh.models.widgets.tables.NumberFormatter(format="0.0000")
    stats_format = {col: number_format for col in stats_df.columns}
    stats_format["Statistic"] = "html"
    # construct the report
    layout = panel.Column(
        panel.Row(
            panel.pane.Bokeh(fig),
            panel.widgets.Tabulator(
                stats_df,
                show_index=False,
                selectable=False,
                disabled=True,
                formatters=stats_format,
                configuration={"columnDefaults": {"headerSort": False}},
            ),
        ),
        panel.widgets.Tabulator(
            # use full data frame including nans for table
            dataframe,
            show_index=False,
            disabled=True,
            formatters={
                "SMILES": "html",
                "Molecule": "html",
                "DG (kcal/mol) (FECS)": number_format,
                "uncertainty (kcal/mol) (FECS)": number_format,
                "DG (kcal/mol) (EXPT)": number_format,
                "uncertainty (kcal/mol) (EXPT)": number_format,
                "prediction error (kcal/mol)": number_format,
            },
            configuration={"rowHeight": 300},
        ),
        sizing_mode="stretch_width",
        scroll=True,
    )
    return layout


def create_relative_report(dataframe: pd.DataFrame) -> panel.Column:
    """
    Create a cinnabar style interactive report for the dataframe of relative free energy predictions, this dataframe
    should include experimental data already.

    Args:
        dataframe: The dataframe of relative predictions to construct the report for.

    Returns:
        A panel column containing an interactive plot and table of the free energy predictions.
    """

    mols, combined_smiles, titles = [], [], []
    for _, data in dataframe.iterrows():
        smiles = ".".join([data["SMILES_A"], data["SMILES_B"]])
        combined_smiles.append(smiles)
        mols.append(draw_mol(smiles=smiles))
        titles.append((data["labelA"], data["labelB"]))
    dataframe["Molecules"] = mols
    dataframe["labels"] = titles
    dataframe["smiles"] = combined_smiles
    # create a plotting dataframe which drops rows with nans
    plotting_df = dataframe.dropna(axis=0, inplace=False, ignore_index=True)
    # create the DG plot
    fig = plotmol_relative(
        calculated=plotting_df["DDG (kcal/mol) (FECS)"],
        experimental=plotting_df["DDG (kcal/mol) (EXPT)"],
        smiles=plotting_df["smiles"],
        titles=plotting_df["labels"],
        calculated_uncertainty=plotting_df["uncertainty (kcal/mol) (FECS)"],
        experimental_uncertainty=plotting_df["uncertainty (kcal/mol) (EXPT)"],
    )
    # calculate the bootstrapped stats using cinnabar
    stats_data = []
    for statistic in ["RMSE", "MUE", "R2", "rho"]:
        s = stats.bootstrap_statistic(
            plotting_df["DDG (kcal/mol) (EXPT)"],
            plotting_df["DDG (kcal/mol) (FECS)"],
            plotting_df["uncertainty (kcal/mol) (EXPT)"],
            plotting_df["uncertainty (kcal/mol) (FECS)"],
            statistic=statistic,
            include_true_uncertainty=False,
            include_pred_uncertainty=False,
        )
        stats_data.append(
            {
                "Statistic": statistic,
                "value": s["mle"],
                "lower bound": s["low"],
                "upper bound": s["high"],
            }
        )
    stats_df = pd.DataFrame(stats_data)
    # create a format for numerical data in the tables
    number_format = bokeh.models.widgets.tables.NumberFormatter(format="0.0000")
    stats_format = {col: number_format for col in stats_df.columns}
    stats_format["Statistic"] = "html"
    # construct the report
    layout = panel.Column(
        panel.Row(
            panel.pane.Bokeh(fig),
            panel.widgets.Tabulator(
                stats_df,
                show_index=False,
                selectable=False,
                disabled=True,
                formatters=stats_format,
                configuration={"columnDefaults": {"headerSort": False}},
            ),
        ),
        panel.widgets.Tabulator(
            dataframe.drop(columns=["labels", "smiles"]),
            show_index=False,
            disabled=True,
            formatters={
                "SMILES_A": "html",
                "SMILES_B": "html",
                "Molecules": "html",
                "DDG (kcal/mol) (FECS)": number_format,
                "uncertainty (kcal/mol) (FECS)": number_format,
                "DDG (kcal/mol) (EXPT)": number_format,
                "uncertainty (kcal/mol) (EXPT)": number_format,
                "prediction error (kcal/mol)": number_format,
            },
            configuration={"rowHeight": 300},
        ),
        sizing_mode="stretch_width",
        scroll=True,
    )
    return layout


def download_cdd_data(protocol_name: str) -> pd.DataFrame:
    """
    A wrapper method to download CDD protocol data, mainly used to tuck imports.

    Args:
        protocol_name: The name of the CDD protocol to extract experimental data for.

    Returns:
        A dataframe of the extracted and formatted experimental data.
    """
    from asapdiscovery.data.services.cdd.cdd_api import CDDAPI
    from asapdiscovery.data.services.services_config import CDDSettings
    from asapdiscovery.data.util.utils import parse_fluorescence_data_cdd

    settings = CDDSettings()
    cdd_api = CDDAPI.from_settings(settings=settings)

    ic50_data = cdd_api.get_ic50_data(protocol_name=protocol_name)
    # format the data to add the pIC50 and error
    formatted_data = parse_fluorescence_data_cdd(
        mol_df=ic50_data, assay_name=protocol_name
    )

    return formatted_data
