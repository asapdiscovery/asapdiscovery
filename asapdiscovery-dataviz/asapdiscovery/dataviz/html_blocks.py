visualisation_header = """\
<div id="viewport"
role="NGL"
data-proteins='[{
   "type": "data", "value": "mike_combined", "isVariable": true, "loadFx": "loadmike_combined"
}]'
data-backgroundcolor="white"></div>
<script type="text/javascript">document.getElementById("blue").style.color = "blue";</script>
<script src="https://code.jquery.com/jquery-3.4.1.min.js"
              integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo="
              crossorigin="anonymous" type="text/javascript"></script>
<script src="https://unpkg.com/ngl@2.0.0-dev.34/dist/ngl.js" type="text/javascript"></script>
<script src="https://michelanglo.sgc.ox.ac.uk/michelanglo.js" type="text/javascript"></script>
"""


def _indent(indent_me: str) -> str:
    return indent_me.replace("\n", "\n" + "    ")


def _vis_core(pdb_body: str) -> str:
    core = f"""<script type="text/javascript">
    var mike_combined =     `{_indent(pdb_body)}    
        `.replace(/^ +/gm, '');

</script>


"""
    return core


visualisation_tail = """\
<script type="text/javascript">


function loadmike_combined (protein) {
    var stage=NGL.getStage('viewport'); //alter if not using multiLoader.

    //define colors
    var nonCmap = {'N': '0x3333ff', 'O': '0xff4c4c', 'H': '0xe5e5e5', 'S': '0xe5c53f', 'Cl': '0x1ff01f', 'Na': '0xab5cf2'};
    var sermap={};
    var chainmap={'A': '0x33ff33', 'B': '0x33ff33', 'C': '0xffa5d8'};
    var resmap={};
    var schemeId = NGL.ColormakerRegistry.addScheme(function (params) {
        this.atomColor = function (atom) {
            chainid=atom.chainid;
            if (! isNaN(parseFloat(chainid))) {chainid=atom.chainname} // hack for chainid/chainIndex/chainname issue if the structure is loaded from string.
            if (atom.serial in sermap)  {return +sermap[atom.serial]}
            else if (atom.element in nonCmap) {return +nonCmap[atom.element]}
            else if (atom.element in nonCmap) {return +nonCmap[atom.element]}
            else if (chainid+atom.resno in resmap) {return +resmap[chainid+atom.resno]}
            else if (chainid in chainmap) {return +chainmap[chainid]}
            else {return 0x7b7d7d} //black as the darkest error!
        };
    });

    //representations
    
    protein.removeAllRepresentations();

        // Show the ligand.
        let sticks = new NGL.Selection( 'ligand' );
        protein.addRepresentation( 'licorice', {color: schemeId, sele: sticks.string, opacity: 1.0} );

"""


def make_core_html(pdb_body: str) -> str:
    return visualisation_header + _vis_core(pdb_body) + visualisation_tail


orient_tail_7ene = """\
    //orient
    stage.viewerControls.orient((new NGL.Matrix4).fromArray([-47.411552767443936, 18.160442129079684, 45.864598141236, 0.0, 33.734451290915786, -34.476765704662284, 48.52362967346835, 0.0, 35.99080588204288, 56.238428207759625, 14.936708193081358, 0.0, 1.14170241355896, 9.264076232910156, -58.5212287902832, 1.0]));
    stage.setParameters({ cameraFov: 20.0, fogNear: 45.0}); //clipFar: 44.402000427246094, clipNear: -44.39164352416992
}



</script> """


orient_tail_mers = """\
    //orient
    stage.viewerControls.orient((new NGL.Matrix4).fromArray([27.892651485688475, -17.496836682191315, 27.842153246574526, 0.0, 4.645958152833828, -34.046145151871315, -26.049992638068943, 0.0, 32.55357881447708, 19.850633421900966, -20.137968094041753, 0.0, -9.070898056030273, 0.7458584308624268, -23.088354110717773, 1.0]));
    stage.setParameters({ cameraFov: 20.0, fogNear: 45.0}); //clipFar: 67.24873352050781, clipNear: -67.247802734375
}



</script> """


orient_tail_sars2 = """\
    //orient
    stage.viewerControls.orient((new NGL.Matrix4).fromArray([32.700534186404866, -23.332587195321253, 28.280957904108163, 0.0, 2.9287225438600046, -36.11192126726678, -33.179842577844965, 0.0, 36.54640101226278, 23.77111378788311, -22.645942287727394, 0.0, -9.279577255249023, 1.1916427612304688, -23.425792694091797, 1.0]));
    stage.setParameters({ cameraFov: 20.0, fogNear: 45.0}); //clipFar: 28.732975006103516, clipNear: -28.735034942626953
}



</script> """


#TODO: change to proper name once it has one
orient_tail_272 = """\
    //orient
    stage.viewerControls.orient((new NGL.Matrix4).fromArray([27.379894279306995, -38.3384651476274, 35.99831312046376, 0.0, -2.131373802248433, -41.360394640088316, -42.42809094617655, 0.0, 52.546711765211285, 18.298874302466288, -20.47814494185161, 0.0, -13.712335586547852, 4.014099597930908, -21.26261329650879, 1.0]));
    stage.setParameters({ cameraFov: 20.0, fogNear: 45.0}); //clipFar: 6315.492679595947, clipNear: -6315.499507904053
}



</script> """




colour_mers = """\
        // Define the binding pocket.
        const data = {
        'color_dict': {
            'subP1': 'yellow',
            'subP1_prime': 'orange',
            'subP2': 'skyblue',
            'subP3_4_5': 'aquamarine'
        },
        'pocket_dict': {
            'subP1' : '143+144+145+146+147+148+166+175',
            'subP1_prime' : '25+26+27',
            'subP2' : '41+49+54',
            'subP3_4_5' : '168+169+170+171+192+193+194+195',
            'sars_unique' : '25+49+145+167+171+172+184+189+191+193+194',
        }
        }
        // Color the BP by subpocket definitions.
        const othercolor = 'white'; // resi not selected
        const uncolored = 'gainsboro'; // sars_unique is not assigned a color.
        let selecol = Object.entries(data.pocket_dict).map(([name, pymol_sele]) => [data.color_dict[name] || uncolored, pymol_sele.replace(/\+/g, ' or ')]);
        selecol.push([othercolor, '*']);  // default
        const pocket_scheme = NGL.ColormakerRegistry.addSelectionScheme(selecol);
        protein.addRepresentation( 'surface', {color: pocket_scheme, sele: 'not ligand', opacity: 0.8, side: 'front',} );
        // Show sticks for residues. Just binding pocket AAs would be ideal here at some point.
        let cartoon = new NGL.Selection( '*' );
        myData.current_cartoonScheme = protein.addRepresentation( 'licorice', {color: schemeId, sele: cartoon.string, smoothSheet: true, opacity: 1.0} );

        // Add interactions (contacts). Some inter-residue contacts are okay.
        function getNeighbors(protein, sele, radius) {
            const neigh_atoms = protein.structure.getAtomSetWithinSelection( sele, radius );
            const resi_atoms = protein.structure.getAtomSetWithinGroup( neigh_atoms );
            return resi_atoms.toSeleString()
        };
        const neigh_sele = getNeighbors(protein, 'ligand', 2);
        protein.addRepresentation( 'contact', {sele: neigh_sele});

"""


colour_sars2 = """\
        // Define the binding pocket.
        const data = {
        'color_dict': {
            'subP1': 'yellow',
            'subP1_prime': 'orange',
            'subP2': 'skyblue',
            'subP3_4_5': 'aquamarine'
        },
        'pocket_dict': {
            'subP1' : '140-145+163+172',
            'subP1_prime' : '25-27',
            'subP2' : '41+49+54',
            'subP3_4_5' : '165-168+189-192',
            'sars_unique' : '25+49+142+164+168+169+181+186+188+190+191',
        }
        }
        // Color the BP by subpocket definitions.
        const othercolor = 'white'; // resi not selected
        const uncolored = 'gainsboro'; // sars_unique is not assigned a color.
        let selecol = Object.entries(data.pocket_dict).map(([name, pymol_sele]) => [data.color_dict[name] || uncolored, pymol_sele.replace(/\+/g, ' or ')]);
        selecol.push([othercolor, '*']);  // default
        const pocket_scheme = NGL.ColormakerRegistry.addSelectionScheme(selecol);
        protein.addRepresentation( 'surface', {color: pocket_scheme, sele: 'not ligand', opacity: 0.8, side: 'front',} );
        // Show sticks for residues. Just binding pocket AAs would be ideal here at some point.
        let cartoon = new NGL.Selection( '*' );
        myData.current_cartoonScheme = protein.addRepresentation( 'licorice', {color: schemeId, sele: cartoon.string, smoothSheet: true, opacity: 1.0} );

        // Add interactions (contacts). Some inter-residue contacts are okay.
        function getNeighbors(protein, sele, radius) {
            const neigh_atoms = protein.structure.getAtomSetWithinSelection( sele, radius );
            const resi_atoms = protein.structure.getAtomSetWithinGroup( neigh_atoms );
            return resi_atoms.toSeleString()
        };
        const neigh_sele = getNeighbors(protein, 'ligand', 2);
        protein.addRepresentation( 'contact', {sele: neigh_sele});

"""

colour_7ene = colour_sars2
