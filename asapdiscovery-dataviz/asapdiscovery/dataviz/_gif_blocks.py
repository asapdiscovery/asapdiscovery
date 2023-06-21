# SARS2
view_coords_sars2 = (
    -0.729172230,
    -0.243629232,
    -0.639491022,
    -0.022380522,
    -0.925491273,
    0.378106207,
    -0.683962226,
    0.290017933,
    0.669392288,
    0.000049770,
    0.000011081,
    -71.979034424,
    15.792427063,
    -2.869046450,
    16.934762955,
    25.658096313,
    118.312721252,
    -20.000000000,
)

# MERS
view_coord_mers = (
    -0.635950804,
    -0.283323288,
    -0.717838645,
    -0.040723491,
    -0.916550398,
    0.397835642,
    -0.770651817,
    0.282238036,
    0.571343124,
    0.000061535,
    0.000038342,
    -58.079559326,
    8.052228928,
    0.619271040,
    21.864795685,
    -283.040344238,
    399.190032959,
    -20.000000000,
)

# 7ENE
view_coords_7ene = (
    0.710110664,
    0.317291290,
    -0.628544748,
    -0.485409290,
    -0.426031262,
    -0.763462067,
    -0.510019004,
    0.847245276,
    -0.148514912,
    0.000047840,
    0.000042381,
    -58.146995544,
    -1.610392451,
    -9.301478386,
    57.779785156,
    13.662354469,
    102.618484497,
    -20.000000000,
)


# 272
view_coords_272 = (
    0.909307361,
    0.223530963,
    -0.350987315,
    0.331670791,
    -0.898707509,
    0.286908478,
    -0.251302391,
    -0.377300352,
    -0.891342342,
    0.000005919,
    0.000006526,
    -76.573043823,
    4.865715981,
    -5.167206764,
    33.649829865,
    35.030181885,
    118.116012573,
    -20.000000000,
)


# set colorings of subpockets by resn. This may change over time.
# SARS2
pocket_dict_sars2 = {
    "subP1": "140-145+163+172",
    "subP1_prime": "25-27",
    "subP2": "41+49+54",
    "subP3_4_5": "165-168+189-192",
    "sars_unique": "25+49+142+164+168+169+181+186+188+190+191",
}

# MERS
pocket_dict_mers = {
    "subP1": "143+144+145+146+147+148+166+175",
    "subP1_prime": "25+26+27",
    "subP2": "41+49+54",
    "subP3_4_5": "168+169+170+171+192+193+194+195",
    "sars_unique": "25+49+145+167+171+172+184+189+191+193+194",
}

color_dict = {
    "subP1": "yellow",
    "subP1_prime": "orange",
    "subP2": "skyblue",
    "subP3_4_5": "aquamarine",
}
# TODO: pick color-blind-friendly scheme, e.g. using https://colorbrewer2.org/#type=qualitative&scheme=Pastel1&n=4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
