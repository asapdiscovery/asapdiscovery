import base64
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


def dg_to_pic50_dataframe(df):
    """
    Given a wrangled `FEMap` dataframe of absolute DG values (kcal/mol), replace 'kcal/mol' columns with
    pIC50 values.
    """
    # we have a decent amount of columns to get through, so replacing all numeric columns
    # with their respective pIC50 values will be much cleaner.
    for column in df._get_numeric_data().columns:
        # replace the kcal/mol values with pIC50s.
        df[column] = dG_to_pIC50(df[column].values)

        # just need to rename the column now. No clean way of doing this AFAIK.
        if "DDG (kcal/mol)" in column:
            df = df.rename(
                columns={
                    column: f"{column.replace('DDG (kcal/mol)', 'relative pIC50')}"
                }
            )
        elif "DG (kcal/mol)" in column:
            df = df.rename(
                columns={column: f"{column.replace('DG (kcal/mol)', 'pIC50')}"}
            )
        elif "uncertainty (kcal/mol)" in column:
            df = df.rename(
                columns={
                    column: f"{column.replace('uncertainty (kcal/mol)', 'pIC50 uncertainty')}"
                }
            )
        elif "prediction error (kcal/mol)" in column:
            df = df.rename(
                columns={
                    column: f"{column.replace('prediction error (kcal/mol)', 'prediction error (pIC50)')}"
                }
            )

    return df


def add_smiles_to_df(dataframe: pd.DataFrame, ligands: list) -> pd.DataFrame:
    """
    Given a wrangled DF containing either `label` (absolute) or `labelA` and `labelB` (relative),
    add SMILES for each row.

    Args:
        dataframe: The pandas dataframe we should add the smiles to.
        ligands: The list of ligands from which we should extract the smiles based on matching by name.
    Returns:
        The dataframe with added smiles column(s)
    """
    # get the experimental data for SMILES per compound name.
    expt_smiles = {ligand.name: ligand.smiles for ligand in ligands}

    # get the SMILES. Relative dataframe is a bit more involved than absolute.
    if "labelA" in dataframe.columns:
        smiles_a = []
        smiles_b = []
        for labelA, labelB in dataframe[["labelA", "labelB"]].values:
            smiles_a.append(expt_smiles[labelA])
            smiles_b.append(expt_smiles[labelB])
        dataframe["SMILES_A"] = smiles_a
        dataframe["SMILES_B"] = smiles_b

    elif "label" in dataframe.columns:
        smiles = [expt_smiles[label] for label in dataframe["label"].values]
        dataframe["SMILES"] = smiles

    return dataframe


def extract_experimental_data(
    reference_csv: str, assay_units: Literal["pIC50", "IC50"]
) -> dict[str, tuple[unit.Quantity, unit.Quantity]]:
    """
    Extract the experimental data from the given csv file, this assumes the csv has been downloaded from cdd.
    Where the molecule identifier is under column 'Molecule Name' and the experimental data is pIC50 / IC50
    TODO make more general

    Args:
        reference_csv: The name of the csv file with the experimental data
        assay_units: The assay units of 'pIC50' or 'IC50' that the experimental data is given in.

    Returns:
        A dictionary of molecule names and tuples of
        experimental data and its associated uncertainty converted to Gibbs free energy in kcal/mol.
    """
    experimental_data = {}
    assay_tag = assay_units + "_Mean"
    exp_data = pd.read_csv(reference_csv).fillna(0)
    # work out the columns for the ref data and the uncertainty
    assay_endpoint_tag, assay_endpoint_confidence_tag = None, None
    for col in exp_data.columns:
        if col.endswith(assay_tag):
            assay_endpoint_tag = col
        elif col.endswith(f"{assay_tag} Standard Deviation (±)"):
            assay_endpoint_confidence_tag = col
    if assay_endpoint_tag is None or assay_endpoint_confidence_tag is None:
        raise RuntimeError(
            f"Could not determine the assay tag from the provided units {assay_units}."
        )

    if assay_units == "pIC50":
        converter = pic50_to_dg
        units = unit.dimensionless
    else:
        converter = ki_to_dg
        units = unit.molar

    for _, row in exp_data.iterrows():
        # get the data.
        name = row["Molecule Name"]
        exp_value = row[assay_endpoint_tag]
        uncertainty = row[assay_endpoint_confidence_tag]

        dg, ddg = converter(exp_value * units, uncertainty * units)
        experimental_data[name] = (dg, ddg)

    return experimental_data


def add_absolute_expt(
    dataframe: pd.DataFrame,
    experimental_data: dict[str, tuple[unit.Quantity, unit.Quantity]],
):
    """
    Edit the dataframe inplace by adding experimental data provided to it.

    Args:
        dataframe: The dataframe of absolute free energy predictions to add the experimental data to.
        experimental_data: A dictionary of experimental free energies in units of kcal/mol to add to the dataframe.
    """
    experimental_col, uncertainty_col = [], []
    for mol_name in dataframe["label"].values:
        experimental_col.append(get_magnitude(experimental_data[mol_name][0]))
        uncertainty_col.append(get_magnitude(experimental_data[mol_name][1]))
    dataframe["DG (kcal/mol) (EXPT)"] = experimental_col
    dataframe["uncertainty (kcal/mol) (EXPT)"] = uncertainty_col


def add_relative_expt(
    dataframe: pd.DataFrame,
    experimental_data: dict[str, tuple[unit.Quantity, unit.Quantity]],
):
    """
    Edit the relative dataframe in place by adding experimental data provided to it.

    Args:
        dataframe: The dataframe of relative free energy predictions to add the experimental data to.
        experimental_data: A dictionary of experimental free energies in units of kcal/mol to add to the dataframe.
    """
    experimental_col, uncertainty_col = [], []
    for _, row in dataframe.iterrows():
        label_a, label_b = row[["labelA", "labelB"]]

        # compute experimental DDG for this edge
        ddg = get_magnitude(experimental_data[label_b][0]) - get_magnitude(
            experimental_data[label_a][0]
        )
        # take the average uncertainty between measurements for this edge.
        delta_ddg = np.mean(
            [
                get_magnitude(experimental_data[label_a][1]),
                get_magnitude(experimental_data[label_b][1]),
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a `cinnabar` `FEMap` add the experimental reference data and generate and return:
    1. a Pandas DataFrame that has all absolute predictions and measurements (DG in kcal/mol) and (pIC50)
    2. a pd DF that has all relative predictions and measurements (DDG in kcal/mol) and PIC50

    Args:
        fe_map: The cinnabar FEMap which has all calculated edges present and the absolute estimates.
        ligands: The list of openfe ligands which are part of the network.
        assay_units: The units of the experimental data, which should be extracted from the reference dataset.
        reference_dataset: The name of the cdd csv file which contains the experimental data.

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

    # add experimental data if available
    if reference_dataset:
        experimental_data = extract_experimental_data(
            reference_csv=reference_dataset, assay_units=assay_units
        )
        add_absolute_expt(dataframe=absolute_df, experimental_data=experimental_data)
        add_relative_expt(dataframe=relative_df, experimental_data=experimental_data)

        absolute_df = shift_and_add_prediction_error(df=absolute_df, point_type="DG")
        relative_df = shift_and_add_prediction_error(df=relative_df, point_type="DDG")

    # now also generate the pCI50 absolute dataframe.
    # absolute_df = dg_to_pic50_dataframe(absolute_df)
    # relative_df_wrangled_pic50 = dg_to_pic50_dataframe(relative_df_wrangled)

    # finally add SMILES to each dataframe.
    for df in [absolute_df, relative_df]:
        add_smiles_to_df(df, ligands)

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
    """
    # add drawn molecule as a column
    mols = [draw_mol(smiles) for smiles in dataframe["SMILES"]]
    dataframe["Molecule"] = mols
    # create the DG plot
    fig = plotmol_absolute(
        calculated=dataframe["DG (kcal/mol) (FECS)"],
        experimental=dataframe["DG (kcal/mol) (EXPT)"],
        smiles=dataframe["SMILES"],
        titles=dataframe["label"],
        calculated_uncertainty=dataframe["uncertainty (kcal/mol) (FECS)"],
        experimental_uncertainty=dataframe["uncertainty (kcal/mol) (EXPT)"],
    )
    # calculate the bootstrapped stats using cinnabar
    stats_data = []
    for statistic in ["RMSE", "MUE", "R2", "rho"]:
        s = stats.bootstrap_statistic(
            dataframe["DG (kcal/mol) (EXPT)"],
            dataframe["DG (kcal/mol) (FECS)"],
            dataframe["uncertainty (kcal/mol) (EXPT)"],
            dataframe["uncertainty (kcal/mol) (FECS)"],
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
            dataframe,
            show_index=False,
            selectable="checkbox",
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
    # create the DG plot
    fig = plotmol_relative(
        calculated=dataframe["DDG (kcal/mol) (FECS)"],
        experimental=dataframe["DDG (kcal/mol) (EXPT)"],
        smiles=combined_smiles,
        titles=titles,
        calculated_uncertainty=dataframe["uncertainty (kcal/mol) (FECS)"],
        experimental_uncertainty=dataframe["uncertainty (kcal/mol) (EXPT)"],
    )
    # calculate the bootstrapped stats using cinnabar
    stats_data = []
    for statistic in ["RMSE", "MUE", "R2", "rho"]:
        s = stats.bootstrap_statistic(
            dataframe["DDG (kcal/mol) (EXPT)"],
            dataframe["DDG (kcal/mol) (FECS)"],
            dataframe["uncertainty (kcal/mol) (EXPT)"],
            dataframe["uncertainty (kcal/mol) (FECS)"],
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
            dataframe,
            show_index=False,
            selectable="checkbox",
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
