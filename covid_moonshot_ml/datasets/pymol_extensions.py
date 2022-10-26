# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:34:28 2020

@author: alexpayne
"""

import yaml
from pymol import cmd, stored

## Very useful script
def make_selection(
    name="chainA", selection="chain A", structure_list="all", delete=False
):
    if structure_list == "all":
        structure_list = cmd.get_object_list()
    else:
        ## This is ugly.
        ## It reminds me that I'm expecting a list, e.g. [7bv1, 7bv2]
        ## The structures should not be in quotes
        structure_list = (
            structure_list.replace("[", "").replace("]", "").split(",")
        )
        print(structure_list)

    for structure in structure_list:
        selname = f"{structure}_{name}"
        full_selection = f"{structure} and ({selection})"
        print(f"Making {selname}: {full_selection}")
        cmd.select(selname, full_selection)

        ## Nifty way to get rid of a bunch of selections I just made if made by accident
        if delete:
            cmd.delete(selname)
            print("############ Deleted #############")


def sel_from_file(file_name, structure_list="all", delete=False):
    with open(file_name) as file:
        seldict = yaml.full_load(file)
    if structure_list == "all":
        structure_list = cmd.get_object_list()
    else:
        ## This is ugly.
        ## It reminds me that I'm expecting a list, e.g. [7bv1, 7bv2]
        ## The structures should not be in quotes
        structure_list = (
            structure_list.replace("[", "").replace("]", "").split(",")
        )
        print(structure_list)
    for structure in structure_list:
        for name, selection in seldict.items():
            make_selection(name, selection, structure, delete)


def align_all(sel_str, reference):
    for model in cmd.get_names(type="selections", selection=sel_str):
        if not str(model) == "sele":
            print(model)
            cmd.align(model, reference)


cmd.extend("make_selection", make_selection)
cmd.extend("sel_from_file", sel_from_file)
cmd.extend("align_all", align_all)
