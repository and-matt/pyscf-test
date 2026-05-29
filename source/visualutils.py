"""
Tools to visualize orbitals and molecular structures.
"""

import os
import webbrowser
import py3Dmol
from pyscfutils import xyz_string, cartesian_modevec


def view_orbital(mol_obj, cube_filename, iso=0.025, alpha=0.9, html=True):
    """
    Visualize orbital from pyscf mol object and cube file. 
    Generate view with py3Dmol which can be displayed as view.show().
    Optionally save orbital as html and visualize it in an external browser.
    This helps keep the notebook nice and lightweight.
    """
    with open(cube_filename) as f:
        cube_data = f.read()
    view = py3Dmol.view(width=600, height=300)
    symbols = mol_obj.elements
    coords = mol_obj.atom_coords(unit="Angstrom")
    view.addModel(xyz_string(symbols, coords), "xyz")
    view.setStyle({"model":0}, {"stick":{"colorscheme":"grayCarbon"}})
    view.addVolumetricData(cube_data, "cube", {
        "isoval": iso,
        "color": "blue",
        "opacity": alpha
    })
    view.addVolumetricData(cube_data, "cube", {
        "isoval": -iso,
        "color": "red",
        "opacity": alpha
    })
    view.setViewStyle({"style": "outline"})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.rotate(-45, "x")
    view.zoom(1.5)
    if html:
        htmlview = view._make_html()
        html_filename = cube_filename.split(".")[0] + ".html"
        with open(html_filename, "w") as f:
            f.write(htmlview)
        webbrowser.open("file://" + os.path.realpath(html_filename))
    return view


def view_vibration(mol_obj, modevec, amplitude=0.5, equilibrium=False):
    """
    Visualize distorted geometries along a vibrational mode.
    Generate view with py3Dmol which can be displayed as view.show().
    """
    view = py3Dmol.view(width=600, height=300)
    symbols = mol_obj.elements
    coords = mol_obj.atom_coords(unit="Angstrom")
    coords_pve = coords + amplitude * cartesian_modevec(mol_obj, modevec)
    coords_nve = coords - amplitude * cartesian_modevec(mol_obj, modevec)
    # Draw distorted structures with thinner sticks for better rendering
    radius = 0.25 # angstrom, default value in py3Dmol
    # Positive displacement
    view.addModel(xyz_string(symbols, coords_pve), "xyz")
    view.setStyle({"model":0},
                  {"stick":{"color":"blue", "radius":0.95*radius}})
    # Negative displacement
    view.addModel(xyz_string(symbols, coords_nve), "xyz")
    view.setStyle({"model":1}, 
                  {"stick":{"color":"red", "radius":0.95*radius}})
    # Equilibrium
    if equilibrium:
        view.addModel(xyz_string(symbols, coords), "xyz")
        view.setStyle({"model":2},
                      {"stick":{"colorscheme":"grayCarbon", "radius":radius}})
    view.setViewStyle({"style": "outline"})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.rotate(-45, "x")
    view.zoom(1.5)
    return view