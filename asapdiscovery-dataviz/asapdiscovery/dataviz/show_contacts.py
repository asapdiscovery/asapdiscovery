#!/usr/bin/python

"""PyMOL plugin that provides show_contacts command and GUI
for highlighting good and bad polar contacts. Factored out of
clustermols by Matthew Baumgartner.
The advantage of this package is it requires many fewer dependencies.
"""

import os  # noqa: F401
import sys  # noqa: F401

DEBUG = 1


def show_contacts(
    pymol_instance,
    selection,
    selection2,
    result="contacts",
    cutoff=3.6,
    bigcutoff=4.0,
    SC_DEBUG=DEBUG,
):
    """
    USAGE

    show_contacts pymol_instance, selection, selection2, [result=contacts],[cutoff=3.6],[bigcutoff=4.0]

    Show various polar contacts, the good, the bad, and the ugly.

    Edit MPB 6-26-14: The distances are heavy atom distances, so I upped the default cutoff to 4.0

    Returns:
    True/False -  if False, something went wrong
    """
    if SC_DEBUG > 4:
        print("Starting show_contacts")
        print('selection = "' + selection + '"')
        print('selection2 = "' + selection2 + '"')

    result = pymol_instance.cmd.get_legal_name(result)

    # if the group of contacts already exist, delete them
    pymol_instance.cmd.delete(result)

    # ensure only N and O atoms are in the selection
    all_don_acc1 = selection + " and (donor or acceptor)"
    all_don_acc2 = selection2 + " and  (donor or acceptor)"

    if SC_DEBUG > 4:
        print('all_don_acc1 = "' + all_don_acc1 + '"')
        print('all_don_acc2 = "' + all_don_acc2 + '"')

    # if theses selections turn out not to have any atoms in them, pymol throws cryptic errors when calling the dist function like:
    # 'Selector-Error: Invalid selection name'
    # So for each one, manually perform the selection and then pass the reference to the distance command and at the end, clean up the selections
    # the return values are the count of the number of atoms
    all1_sele_count = pymol_instance.cmd.select("all_don_acc1_sele", all_don_acc1)
    all2_sele_count = pymol_instance.cmd.select("all_don_acc2_sele", all_don_acc2)

    # print out some warnings
    if DEBUG > 3:
        if not all1_sele_count:
            print("Warning: all_don_acc1 selection empty!")
        if not all2_sele_count:
            print("Warning: all_don_acc2 selection empty!")

    ########################################
    allres = result + "_all"
    if all1_sele_count and all2_sele_count:
        pymol_instance.cmd.distance(
            allres, "all_don_acc1_sele", "all_don_acc2_sele", bigcutoff, mode=0
        )
        pymol_instance.cmd.set("dash_radius", "0.05", allres)
        pymol_instance.cmd.set("dash_color", "purple", allres)
        pymol_instance.cmd.hide("labels", allres)

    ########################################
    # compute good polar interactions according to pymol
    polres = result + "_polar"
    if all1_sele_count and all2_sele_count:
        pymol_instance.cmd.distance(
            polres, "all_don_acc1_sele", "all_don_acc2_sele", cutoff, mode=2
        )  # hopefully this checks angles? Yes
        pymol_instance.cmd.set("dash_radius", "0.126", polres)

    ########################################
    # When running distance in mode=2, the cutoff parameter is ignored if set higher then the default of 3.6
    # so set it to the passed in cutoff and change it back when you are done.
    old_h_bond_cutoff_center = pymol_instance.cmd.get(
        "h_bond_cutoff_center"
    )  # ideal geometry
    old_h_bond_cutoff_edge = pymol_instance.cmd.get(
        "h_bond_cutoff_edge"
    )  # minimally acceptable geometry
    pymol_instance.cmd.set("h_bond_cutoff_center", bigcutoff)
    pymol_instance.cmd.set("h_bond_cutoff_edge", bigcutoff)

    # compute possibly suboptimal polar interactions using the user specified distance
    pol_ok_res = result + "_polar_ok"
    if all1_sele_count and all2_sele_count:
        pymol_instance.cmd.distance(
            pol_ok_res, "all_don_acc1_sele", "all_don_acc2_sele", bigcutoff, mode=2
        )
        pymol_instance.cmd.set("dash_radius", "0.06", pol_ok_res)

    # now reset the h_bond cutoffs
    pymol_instance.cmd.set("h_bond_cutoff_center", old_h_bond_cutoff_center)
    pymol_instance.cmd.set("h_bond_cutoff_edge", old_h_bond_cutoff_edge)

    ########################################

    onlyacceptors1 = selection + " and (acceptor and !donor)"
    onlyacceptors2 = selection2 + " and (acceptor and !donor)"
    onlydonors1 = selection + " and (!acceptor and donor)"
    onlydonors2 = selection2 + " and (!acceptor and donor)"

    # perform the selections
    onlyacceptors1_sele_count = pymol_instance.cmd.select(
        "onlyacceptors1_sele", onlyacceptors1
    )
    onlyacceptors2_sele_count = pymol_instance.cmd.select(
        "onlyacceptors2_sele", onlyacceptors2
    )
    onlydonors1_sele_count = pymol_instance.cmd.select("onlydonors1_sele", onlydonors1)
    onlydonors2_sele_count = pymol_instance.cmd.select("onlydonors2_sele", onlydonors2)

    # print out some warnings
    if SC_DEBUG > 2:
        if not onlyacceptors1_sele_count:
            print("Warning: onlyacceptors1 selection empty!")
        if not onlyacceptors2_sele_count:
            print("Warning: onlyacceptors2 selection empty!")
        if not onlydonors1_sele_count:
            print("Warning: onlydonors1 selection empty!")
        if not onlydonors2_sele_count:
            print("Warning: onlydonors2 selection empty!")

    accres = result + "_aa"
    if onlyacceptors1_sele_count and onlyacceptors2_sele_count:
        aa_dist_out = pymol_instance.cmd.distance(
            accres, "onlyacceptors1_sele", "onlyacceptors2_sele", cutoff, 0
        )

        if aa_dist_out < 0:
            print(
                "\n\nCaught a pymol selection error in acceptor-acceptor selection of show_contacts"
            )
            print("accres:", accres)
            print("onlyacceptors1", onlyacceptors1)
            print("onlyacceptors2", onlyacceptors2)
            return False

        pymol_instance.cmd.set("dash_color", "red", accres)
        pymol_instance.cmd.set("dash_radius", "0.125", accres)

    ########################################

    donres = result + "_dd"
    if onlydonors1_sele_count and onlydonors2_sele_count:
        dd_dist_out = pymol_instance.cmd.distance(
            donres, "onlydonors1_sele", "onlydonors2_sele", cutoff, 0
        )

        # try to catch the error state
        if dd_dist_out < 0:
            print("\n\nCaught a pymol selection error in dd selection of show_contacts")
            print("donres:", donres)
            print("onlydonors1", onlydonors1)
            print("onlydonors2", onlydonors2)
            print(
                "pymol_instance.cmd.distance('"
                + donres
                + "', '"
                + onlydonors1
                + "', '"
                + onlydonors2
                + "', "
                + str(cutoff)
                + ", 0)"
            )
            return False

        pymol_instance.cmd.set("dash_color", "red", donres)
        pymol_instance.cmd.set("dash_radius", "0.125", donres)

    ##################################################
    # find the buried unpaired atoms of the receptor #
    ##################################################

    # initialize the variable for when CALC_SASA is False
    unpaired_atoms = ""

    # Group
    pymol_instance.cmd.group(
        result,
        "%s %s %s %s %s %s"
        % (polres, allres, accres, donres, pol_ok_res, unpaired_atoms),
    )

    # Clean up the selection objects
    # if the show_contacts debug level is high enough, don't delete them.
    if SC_DEBUG < 5:
        pymol_instance.cmd.delete("all_don_acc1_sele")
        pymol_instance.cmd.delete("all_don_acc2_sele")
        pymol_instance.cmd.delete("onlyacceptors1_sele")
        pymol_instance.cmd.delete("onlyacceptors2_sele")
        pymol_instance.cmd.delete("onlydonors1_sele")
        pymol_instance.cmd.delete("onlydonors2_sele")

    return True
