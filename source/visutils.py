"""
Tools for visualizing orbitals and molecular structures.
"""

import os
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import py3Dmol
from dftutils import xyz_string
from ephutils import cartesian_modevec


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


def view_vibration_distorted(mol_obj, modevec, amplitude=0.5, equilibrium=False):
    """
    Visualize distorted geometries along a vibrational mode.
    Generate view with py3Dmol which can be displayed as view.show().
    """
    view = py3Dmol.view(width=600, height=300)
    symbols = mol_obj.elements
    coords = mol_obj.atom_coords(unit="Angstrom")
    mass = mol_obj.atom_mass_list()
    coords_pve = coords + amplitude * cartesian_modevec(modevec, mass)
    coords_nve = coords - amplitude * cartesian_modevec(modevec, mass)
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


def view_vibration_animation(mol_obj, modevec, amplitude=0.5, period=1000, fps=60):
    """
    Visualize vibrational animation along a vibrational mode.
    Generate view with py3Dmol which can be displayed as view.show().
    """
    view = py3Dmol.view(width=600, height=300)
    symbols = mol_obj.elements
    coords = mol_obj.atom_coords(unit="Angstrom")
    mass = mol_obj.atom_mass_list()
    displacement = amplitude * cartesian_modevec(modevec, mass)
    xyz_frames = ""
    # calculate number of frames from period (ms) and fps
    nframes = round(period / 1000 * fps)
    times = np.linspace(0, 2*np.pi, nframes+1)[:-1]
    for t in times:
        coords_distorted = coords + np.sin(t) * displacement
        xyz_frames += xyz_string(symbols, coords_distorted, filename=False)
    view.addModelsAsFrames(xyz_frames, "xyz")
    view.animate({"loop" : "forward", 
                  "reps" : 0,
                  "interval" : period / nframes})
    view.setStyle({"stick":{"colorscheme":"grayCarbon"}})
    view.setViewStyle({"style": "outline"})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.rotate(-45, "x")
    view.zoom(1.5)
    return view


def view_vibration_arrows(mol_obj, modevec, amplitude=5.0, threshold=0.5):
    """
    Visualize distortion along a vibrational mode using arrows.
    Generate view with py3Dmol which can be displayed as view.show().
    """
    view = py3Dmol.view(width=600, height=300)
    symbols = mol_obj.elements
    coords = mol_obj.atom_coords(unit="Angstrom")
    view.addModel(xyz_string(symbols, coords), "xyz")
    mass = mol_obj.atom_mass_list()
    displacement = amplitude * cartesian_modevec(modevec, mass)
    for r, dr in zip(coords, displacement):
        if np.linalg.norm(dr) > threshold:
            start = {"x": float(r[0]),
                     "y": float(r[1]),
                     "z": float(r[2])}
            end = {"x": float(r[0] + dr[0]),
                   "y": float(r[1] + dr[1]),
                   "z": float(r[2] + dr[2])}
            view.addArrow({"start": start,
                           "end": end,
                           "radius": 0.1,
                           "color": "red"})
    view.setStyle({"stick":{"colorscheme":"grayCarbon"}})
    view.setViewStyle({"style": "outline"})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.rotate(-45, "x")
    view.zoom(1.5)
    return view


def view_multiconfig_vibronic_coupling(mat, frequency=None, coupling=None):
    """Plot e-ph coupling matrix in the singlet multiconfigurational subspace."""
    state_labels = [r'SE$_+$', r'SE$_-$', r'CT$_+$', r'CT$_-$', r'TT']
    # Set range for diverging colormap
    vmin = min(mat.min(), 0)
    vmax = max(mat.max(), 0)
    v = max(np.abs(vmin), np.abs(vmax))
    norm = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=+v)
    # Plot
    fig, ax = plt.subplots()
    p = ax.imshow(mat, cmap='bwr', norm=norm)
    fig.colorbar(p, ax=ax)
    ax.set_xticks(ticks=list(range(5)), labels=state_labels)
    ax.set_yticks(ticks=list(range(5)), labels=state_labels)
    if (frequency is not None) and (coupling is not None):
        f, c = round(frequency), round(coupling)
        ax.set_title(fr"$\omega$ = {f} cm$^{{-1}}$, $w$ = {c} cm$^{{-1}}$")
    return fig


