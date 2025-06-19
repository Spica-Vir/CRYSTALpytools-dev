#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The module for `TOPOND <https://www.crystal.unito.it/topond.html>`_ topological
analysis of electron density
"""
import numpy as np
from warnings import warn

from CRYSTALpytools import units


class ScalarField():
    """
    Basic TOPOND scalar field class, containing a nY\*nX (nZ\*nY\*nX) data
    array for 2D (3D) fields. Call the property-specific child classes below to
    use.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BC, BA (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        type (str): 'RHOO', 'SPDE', 'LAPP', 'LAPM', 'GRHO', 'KKIN', 'GKIN',
            'VIRI', 'ELFB', quantities to plot.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, struc, type, unit):
        self.data = np.array(data, dtype=float)
        self.base = np.array(base, dtype=float)
        self.dimension = self.data.ndim
        self.structure = struc
        self.unit = unit
        self.type = type.upper()
        self.subtracted = False # Hidden. For plotting.

    @classmethod
    def from_file(cls, file, output, type, source):
        """
        Generate an object from output files.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            type (str): 'RHOO', 'SPDE', 'LAPP', 'LAPM', 'GRHO', 'KKIN', 'GKIN',
                'VIRI', 'ELFB', quantities to plot.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (ScalarField)
        """
        if source == 'crystal':
            from CRYSTALpytools.crystal_io import Properties_output
            obj = Properties_output(output).read_topond(file, type)
        else:
            raise Exception("Unknown file format. Source = '{}'.".format(source))
        return obj

    def plot_2D(self, unit, levels, lineplot, contourline, isovalues,
                colorplot, colormap, cbar_label, a_range, b_range, edgeplot,
                x_ticks, y_ticks, figsize, overlay, **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            2D periodicity (``a_range`` and ``b_range``), though available for
            the ``ScalarField`` class, is not suggested as TOPOND plotting
            window does not always commensurate with periodic boundary. The
            ``Trajectory`` class has no 2D periodicity so if ``overlay`` is not
            None, ``a_range``, ``b_range`` and ``edgeplot`` will be disabled.

        3 styles are available:

        1. ``lineplot=True`` and ``colorplot=True``: The color-filled contour
            map with black contour lines.  
        2. ``lineplot=False`` and ``colorplot=True``: The color-filled contour
            map.  
        3. ``lineplot=True`` and ``colorplot=False``: The color coded contour
            line map.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-3}`, 'a.u.' for
                Bohr :math:`^{-3}`.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D.
            lineplot (bool): Plot contour lines.
            contourline (list): nLevel\*3 contour line styles. Useful only if
                ``lineplot=True``. For every line, color, line style and line
                width are defined in sequence.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        from CRYSTALpytools.base.plotbase import plot_2Dscalar, GridRotation2D
        import matplotlib.pyplot as plt

        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower():
            self._set_unit(unit)

        try:
            # dimen
            if self.dimension != 2: raise Exception('Not a 2D scalar field object.')

            # levels
            levels = np.array(levels, ndmin=1, dtype=float)
            if levels.shape[0] == 1:
                levels = np.linspace(np.min(self.data), np.max(self.data), int(levels[0]))
            if levels.ndim > 1: raise ValueError('Levels must be a 1D array.')

            # contour line styles
            if lineplot == False:
                contourline = None
            # colormap
            if colorplot == False:
                colormap = None

            # overlay
            if np.all(overlay!=None):
                if not isinstance(overlay, Trajectory):
                    raise Exception("The overlaied layer must be a topond.Trajectory class or its child classes.")

                diff_base = np.abs(overlay.base-self.base)
                if np.any(diff_base>1e-3):
                    raise Exception("The plotting base of surface and trajectory are different.")
                a_range = [0., 1.]; b_range=[0., 1.] # no periodicity for Traj

            # plot
            ## layout
            if 'fig' not in kwargs.keys():
                fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
                ax_index = 0
            else:
                if 'ax_index' not in kwargs.keys():
                    raise ValueError("Indices of axes must be set when 'fig' is passed.")
                ax_index = int(kwargs['ax_index'])
                fig = kwargs['fig']
                ax = fig.axes[ax_index]

            ## surf first
            if unit.lower() == 'angstrom': pltbase = self.base
            else: pltbase = units.angstrom_to_au(self.base)
            fig = plot_2Dscalar(
                fig, ax, self.data, pltbase, levels, contourline, isovalues, colormap,
                cbar_label, a_range, b_range, False, edgeplot, x_ticks, y_ticks
            )

            ## plot traj
            if np.all(overlay!=None):
                kwargs['fig'] = fig; kwargs['ax_index'] = ax_index
                kwargs['y_ticks'] = y_ticks; kwargs['x_ticks'] = x_ticks
                kwargs['figsize'] = figsize; kwargs['unit'] = unit
                kwargs['title'] = None
                fig = overlay.plot_2D(**kwargs)

        except Exception as e:
            self._set_unit(uold)
            raise e

        self._set_unit(uold)
        return fig

    def plot_3D(self, unit, isovalue, volume_3d, contour_2d, interp, interp_size,
                grid_display_range, **kwargs):
        """
        Visualize **2D or 3D** scalar fields with atomic structures using
        `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        Explanations of input/output arguments are detailed in specific classes.
        """
        from CRYSTALpytools.base.plotbase import plot_GeomScalar
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower():
            self._set_unit(unit)

        try:
            # structure
            if self.structure == None:
                raise Exception("Structure information is required for 3D plots.")

            # Plot figure
            inputs = dict(struc=self.structure,
                          base=self.base,
                          data=self.data,
                          isovalue=isovalue,
                          volume_3d=volume_3d,
                          contour_2d=contour_2d,
                          interp=interp,
                          interp_size=interp_size,
                          grid_display_range=grid_display_range)
            keys = ['colormap', 'opacity', 'transparent', 'color', 'line_width',
                    'vmax', 'vmin', 'title', 'orientation', 'nb_labels', 'label_fmt'
                    'atom_color', 'bond_color', 'atom_bond_ratio', 'cell_display',
                    'cell_color', 'cell_linewidth', 'display_range', 'special_bonds']
            for k, v in zip(kwargs.keys(), kwargs.values()):
                if k in keys: inputs[k] = v

            fig = plot_GeomScalar(**inputs)

        except Exception as e:
            self._set_unit(uold)
            raise e

        self._set_unit(uold)
        return fig

    def subtract(self, *args):
        """
        Subtracting data of the same type from the object.

        Args:
            \*args (str|ScalarField): File names or ``ScalarField`` objects.
                Must be of the same type (check the attribute ``type``).
        Returns:
            self (ScalarField) : Data difference
        """
        from CRYSTALpytools.crystal_io import Properties_output

        for i in args:
            if isinstance(i, str):
                obj = Properties_output().read_topond(i, type=self.type)
            elif isinstance(i, ScalarField):
                obj = i
            else:
                raise TypeError('Inputs must be file name strings or Surf objects.')

            # type
            if self.type != obj.type:
                raise TypeError('Input is not the same type as object.')
            # base vector
            if not np.all(np.abs(self.base-obj.base)<1e-6):
                raise ValueError('Inconsistent base vectors between input and object.')
            # dimensionality
            if self.dimension != obj.dimension:
                raise ValueError('Inconsistent dimensionality between input and object.')
            # mesh grid
            if self.data.shape != obj.data.shape:
                raise ValueError('Inconsistent mesh grid between input and object.')
            # subtract
            self.data = self.data - obj.data

        self.subtracted = True # Hidden. For plotting.
        return self

    def substract(self, *args):
        """An old typo"""
        return self.subtract(*args)

    def _set_unit(self, unit):
        """"Of no practical use."""
        pass


class Trajectory():
    """
    Basic TOPOND trajectory plot class. Call the property-specific child classes
    below to use.

    Args:
        wtraj (list[int]): 1\*nPath, weight of the path, int 0 to 3.
        traj (list[array]): 1\*nPath, list of paths. Every array is a nPoint\*3
            3D ref framework coordinates of points on the path.
        base (array): 3\*3 Cartesian coordinates of the 3 points defining
            vectors BC and BA.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    Returns:
        self (Trajectory): Import attributes listed below.
        trajectory (list): nPath\*2 list of trajectory(2nd) and its weight(1st).
        cpoint (array): nCpoint\*3 array of critical point coordinates in 3D ref framework.
    """
    def __init__(self, wtraj, traj, base, struc, type, unit):
        if len(wtraj) != len(traj):
            raise ValueError('Inconsistent lengths of input trajectory and its weight.')

        self.base = np.array(base, dtype=float)
        self.structure = struc
        self.type = type.upper()
        self.unit = unit
        self.cpoint = []; self.trajectory = []
        for w, t in zip(wtraj, traj):
            if len(t) == 1: self.cpoint.append(t[0])
            else: self.trajectory.append([w, t])
        self.cpoint = np.array(self.cpoint)

    @classmethod
    def from_file(cls, file, output, type, source):
        """
        Generate an object from output files.

        Args:
            file (str): The trajectory data.
            output (str): Output file for the geometry and periodicity.
            type (str): 'TRAJGRAD', 'TRAJMOLG', quantities to plot.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (Trajectory)
        """
        if source == 'crystal':
            from CRYSTALpytools.crystal_io import Properties_output
            obj = Properties_output(output).read_topond(file, type)
        else:
            raise Exception("Unknown file format. Source = '{}'.".format(source))
        return obj

    def plot_2D(
        self, unit, cpt_marker, cpt_color, cpt_size, traj_weight, traj_color,
        traj_linestyle, traj_linewidth, x_ticks, y_ticks, figsize, overlay, **kwargs):
        """
        Get TOPOND trajectory in a 2D plot.

        .. note::

            2D periodicity (``a_range`` and ``b_range``) is not available for
            the ``Trajectory`` class. If ``overlay`` is not None, ``a_range``
            and ``b_range`` and ``edgeplot`` will be disabled for the
            ``ScalarField`` object.

        Args:
            unit (str): unit (str): Plot unit. 'Angstrom' for :math:`\\AA`,
                'a.u.' for Bohr.
            cpt_marker (str): Marker of critical point scatter.
            cpt_color (str): Marker color of critical point scatter.
            cpt_size (float|int): Marker size of critical point scatter.
            traj_weight (int|list): Weight(s) of the plotted trajectories (0 to 3).
            traj_color (str|list): 1\*nTraj_weight list of plotted trajectory
                colors. Use string for the same color.
            traj_linestyle (str|list): 1\*nTraj_weight list of plotted trajectory
                styles. Use string for the same style.
            traj_linewidth (str): Line width of 2D trajectory plot.
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|ScalarField): Overlapping a 2D plot from the
                ``topond.ScalarField`` object if not None.
            \*\*kwargs : Other arguments passed to ``ScalarField.plot_2D()``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.
        Returns:
            fig (Figure): Matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        from CRYSTALpytools.base.plotbase import GridRotation2D
        from CRYSTALpytools.base.plotbase import _plot_label_preprocess

        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower():
            self._set_unit(unit)

        try:
            # overlay
            if np.all(overlay!=None):
                if not isinstance(overlay, ScalarField):
                    raise Exception("The overlaied layer must be a topond.ScalarField class or its child classes.")

                diff_base = np.abs(overlay.base-self.base)
                if np.any(diff_base>1e-3):
                    raise Exception("The plotting base of surface and trajectory are different.")

            # plot setups
            traj_weight = np.array(traj_weight, ndmin=1, dtype=int)
            if type(traj_color) == str:
                if traj_color.lower() == 'default':
                    traj_color = []
                    for wt in traj_weight:
                        if wt == 3: traj_color.append(cpt_color)
                        else: traj_color.append('r')
            if type(traj_linestyle) == str:
                if traj_linestyle.lower() == 'default':
                    traj_linestyle = []
                    for wt in traj_weight:
                        if wt == 3: traj_linestyle.append('-')
                        else: traj_linestyle.append(':')
            if type(traj_linewidth) == str:
                if traj_linewidth.lower() == 'default':
                    traj_linewidth = []
                    for wt in traj_weight:
                        if wt == 3: traj_linewidth.append(1.0)
                        else: traj_linewidth.append(0.5)
            bands = np.zeros([traj_weight.shape[0], 1, 1, 1]) # dummy bands
            cmds = _plot_label_preprocess(bands, None, traj_color, traj_linestyle, traj_linewidth)

            # plot
            ## layout
            if 'fig' not in kwargs.keys():
                fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
                ax_index = 0
            else:
                if 'ax_index' not in kwargs.keys():
                    raise ValueError("Indices of axes must be set when 'fig' is not None.")
                fig = kwargs['fig']
                ax_index = int(kwargs['ax_index'])
                ax = fig.axes[ax_index]

            ## Get bottom surf figure first
            if np.all(overlay!=None) and isinstance(overlay, ScalarField):
                kwargs['a_range'] = [0., 1.]; kwargs['b_range'] = [0., 1.] # no periodicity
                kwargs['edgeplot'] = False; kwargs['figsize'] = figsize
                kwargs['fig'] = fig; kwargs['ax_index'] = ax_index
                kwargs['x_ticks'] = x_ticks; kwargs['y_ticks'] = y_ticks
                kwargs['unit'] = unit; kwargs['title'] = None

                fig = overlay.plot_2D(**kwargs)
                ax = fig.axes[ax_index]

            ## rotate the trajectory to plotting plane, base defined in OXY
            if unit.lower() == 'angstrom': pltbase = self.base
            else: pltbase = units.angstrom_to_au(self.base)

            rot, disp = GridRotation2D(np.vstack([pltbase[1], pltbase[2], pltbase[0]]))
            pltbase = rot.apply(pltbase)
            xmx = np.linalg.norm(pltbase[2, :]-pltbase[1, :])
            ymx = np.linalg.norm(pltbase[0, :]-pltbase[1, :])

            # Plot critical points
            if cpt_marker is not None:
                cpt = (rot.apply(self.cpoint) - pltbase[1]).round(4)
                cpt = cpt[np.where((cpt[:,0]>=0.)&\
                                   (cpt[:,0]<=xmx)&\
                                   (cpt[:,1]>=0.)&\
                                   (cpt[:,1]<=ymx)&\
                                   (np.abs(cpt[:,2])<1e-3))]
                if len(cpt) > 0:
                    cpt += pltbase[1]
                    ax.scatter(cpt[:, 0], cpt[:, 1], marker=cpt_marker, c=cpt_color, s=cpt_size)

            # Plot trajectory
            allwt = np.array([i[0] for i in self.trajectory], dtype=int)
            alltraj = [i[1] for i in self.trajectory]
            for wt, clr, stl, lwid in zip(traj_weight, cmds[1], cmds[2], cmds[3]):
                idx = np.where(allwt==wt)[0]
                for i in idx:
                    plttraj = (rot.apply(alltraj[i]) - pltbase[1]).round(4)
                    plttraj = plttraj[np.where((plttraj[:,0]>=0.)&\
                                               (plttraj[:,0]<=xmx)&\
                                               (plttraj[:,1]>=0.)&\
                                               (plttraj[:,1]<=ymx)&\
                                               (np.abs(plttraj[:,2])<1e-3))]
                    if len(plttraj) == 0: continue

                    plttraj += pltbase[1]
                    ax.plot(plttraj[:, 0], plttraj[:, 1], color=clr[0],
                            linestyle=stl[0], linewidth=lwid[0])

            ax.set_aspect(1.0)
            ax.set_xlim(pltbase[1, 0], pltbase[2, 0])
            ax.set_ylim(pltbase[1, 1], pltbase[0, 1])

        except Exception as e:
            self._set_unit(uold)
            raise e

        self._set_unit(uold)
        return fig

    def plot_3D(self, unit, cpt_marker, cpt_color, cpt_size,
                traj_weight, traj_color, traj_linewidth, **kwargs):
        """
        Visualize trajectories in 3D space with atomic structures using
        `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        Explanations of input/output arguments are detailed in specific classes.
        """
        from CRYSTALpytools.geometry import CStructure
        from CRYSTALpytools.base.plotbase import GridRotation2D
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower():
            self._set_unit(unit)

        try:
            # structure
            if self.structure == None:
                raise Exception("Structure information is required for 3D plots.")
            else:
                if not isinstance(self.structure, CStructure):
                    raise Exception("Structure information must be in CRYSTALpytools CStructure class for 3D plots.")

            # plot setups
            traj_weight = np.array(traj_weight, ndmin=1, dtype=int)
            if type(traj_color) == str:
                if traj_color.lower() == 'default':
                    traj_color = []
                    for wt in traj_weight:
                        if wt == 3: traj_color.append(cpt_color)
                        else: traj_color.append((1.,0.,0.))
            else:
                traj_color = np.array(traj_color, ndmin=2, dtype=float)
                if traj_color.shape[0] == 1 and traj_color.shape[1] == 3:
                    traj_color = traj_color.repeat(traj_weight.shape[0]).reshape([-1, 3]).T
                if traj_color.shape[0] != traj_weight.shape[0] or traj_color.shape[1] != 3:
                    raise Exception("Errors in trajectory color settings.")

            if type(traj_linewidth) == str:
                if traj_linewidth.lower() == 'default':
                    traj_linewidth = []
                    for wt in traj_weight:
                        if wt == 3: traj_linewidth.append(3.0)
                        else: traj_linewidth.append(1.5)
            else:
                traj_linewidth = np.array(traj_linewidth, ndmin=1, dtype=float)
                if traj_linewidth.shape[0] == 1:
                    traj_linewidth = traj_linewidth.repeat(traj_weight.shape[0])
                if traj_linewidth.shape[0] != traj_weight.shape[0]:
                    raise Exception("Errors in trajectory linewidth settings.")

            # Plot figure
            ## Geometry
            if 'display_range' in kwargs.keys():
                display_range = np.array(kwargs['display_range'], dtype=float)
            else:
                display_range = np.array([[0,1], [0,1], [0,1]], dtype=float)

            if np.any(self.structure.pbc==False):
                idir = np.where(self.structure.pbc==False)[0]
                display_range[idir] = [0., 1.]

            idx = np.where(display_range[:,1]-display_range[:,0]<1e-4)[0]
            if len(idx) > 0:
                direct = ['x', 'y', 'z'][idx[0]]
                raise Exception("Structure display range error along {} axis!\n{} min = {:.2f}, {} max = {:.2f}. No data is displayed.".format(
                    direct, direct, display_range[idx[0], 0], direct, display_range[idx[0], 1]))

            keywords = dict(show_the_scene=False, display_range=display_range)
            keys = ['atom_color', 'bond_color', 'atom_bond_ratio', 'cell_display',
                    'cell_color', 'cell_linewidth', 'special_bonds']
            for k, v in zip(kwargs.keys(), kwargs.values()):
                if k in keys: keywords[k] = v

            fig = self.structure.visualize(**keywords)
            ## Critical points
            if cpt_marker is not None:
                pltcpt = self.cpoint
                if len(self.cpoint) > 0:
                    pts = mlab.points3d(self.cpoint[:,0], self.cpoint[:,1], self.cpoint[:,2],
                                        figure=fig, color=tuple(cpt_color), mode=cpt_marker,
                                        scale_factor=cpt_size)

            ## trajectories
            allwt = np.array([i[0] for i in self.trajectory], dtype=int)
            alltraj = [i[1] for i in self.trajectory]
            for wt, clr, lwid in zip(traj_weight, traj_color, traj_linewidth):
                idx = np.where(allwt==wt)[0]
                for i in idx:
                    mlab.plot3d(alltraj[i][:,0], alltraj[i][:,1], alltraj[i][:,2],
                                figure=fig, color=tuple(clr), line_width=lwid,
                                tube_radius=None)
        except Exception as e:
            self._set_unit(uold)
            raise e

        self._set_unit(uold)
        return fig

    def _set_unit(self, unit):
        """"Of no practical use."""
        pass


class ChargeDensity(ScalarField):
    """
    The charge density object from TOPOND. Unit: 'Angstrom' for
    :math:`\\AA^{-3}` and 'a.u.' for Bohr :math:`^{-3}`.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BA, BC (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, struc=None, unit='Angstrom'):
        super().__init__(data, base, struc, 'RHOO', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``ChargeDensity`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (ChargeDensity)
        """
        return super().from_file(file, output, 'RHOO', source)

    def plot_2D(
        self, unit='Angstrom', levels=None, lineplot=True, linewidth=1.0,
        isovalues='%.2f', colorplot=False, colormap='jet', cbar_label='default',
        a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-3}`, 'a.u.' for
                Bohr :math:`^{-3}`. X and y axis scales are changed correspondingly.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. None for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'default' for default values and 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80, 200],
                                  dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # contour line styles
        blimit = -1e-6; rlimit = 1e-6
        if colorplot == False: cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else: cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for i in levels:
            if i < blimit: contourline.append([cb, stlb, linewidth])
            elif i > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if cbar_label=='default':
            if unit.lower() == 'angstrom': ustr = r'$|e|/\AA^{-3}$'
            else: ustr = r'$|e|/Bohr^{-3}$'
            if self.subtracted == False: cbar_label=r'$\rho$ ({})'.format(ustr)
            else: cbar_label=r'$\Delta\rho$ ({})'.format(ustr)

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Charge Density'
                else: title = 'Charge Density + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                unit='Angstrom',
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** charge densities with atomic structures using
        `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            unit (str): Plot unit (only for length units here). 'Angstrom' for
                :math:`\\AA^{-3}`, 'a.u.' for Bohr:math:`^{-3}`.
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. None for default isovalues (0.01, 1, 100
                or -1, 0, 1 for charge differences).
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # default isovalues
        if isovalue is None:
            if self.subtracted == False:
                isovalue = np.array([0.01, 1., 100.], dtype=float)
            else:
                isovalue = np.array([-1, 0, 1], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default isovalue are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=grid_display_range,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``ChargeDensity`` object.

        Args:
            unit (str): ''Angstrom', :math:`\\AA^{-3}`; 'a.u.', Bohr :math:`^{-3}`.
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        dprops = ['data'] # density units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        for d in dprops:
            newattr = getattr(self, d) / cst**3
            setattr(self, d, newattr)
        return self


class SpinDensity(ScalarField):
    """
    The spin density object from TOPOND. Unit: 'Angstrom' for :math:`\\AA^{-3}`
    and 'a.u.' for Bohr :math:`^{-3}`.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BC, BA (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, struc=None, unit='Angstrom'):
        super().__init__(data, base, struc, 'SPDE', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``SpinDensity`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (SpinDensity)
        """
        return super().from_file(file, output, 'SPDE', source)

    def plot_2D(
        self, unit='Angstrom', levels=None, lineplot=True, linewidth=1.0,
        isovalues='%.4f', colorplot=False, colormap='jet', cbar_label='default',
        a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-3}`, 'a.u.' for
                Bohr :math:`^{-3}`. X and y axis scales are changed correspondingly.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. 'None' for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                               -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                               0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                               0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # contour line styles
        blimit = -1e-6; rlimit = 1e-6
        if colorplot == False: cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else: cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for i in levels:
            if i < blimit: contourline.append([cb, stlb, linewidth])
            elif i > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if cbar_label=='default':
            if unit.lower() == 'angstrom': ustr = r'$|e|/\AA^{-3}$'
            else: ustr = r'$|e|/Bohr^{-3}$'
            if self.subtracted == False: cbar_label=r'$\rho$ ({})'.format(ustr)
            else: cbar_label=r'$\Delta\rho$ ({})'.format(ustr)

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Spin Density'
                else: title = 'Spin Density + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                unit='Angstrom',
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** spin densities with atomic structures using
        `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-3}`, 'a.u.' for
                Bohr :math:`^{-3}`.
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. None for default isovalues (-1, 0, 1).
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # default isovalues
        if isovalue is None:
            isovalue = np.array([-1, 0., 1], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default isovalue are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=grid_display_range,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``SpinDensity`` object.

        Args:
            unit (str): ''Angstrom', :math:`\\AA^{-3}`; 'a.u.', Bohr :math:`^{-3}`.
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        dprops = ['data'] # density units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        for d in dprops:
            newattr = getattr(self, d) / cst**3
            setattr(self, d, newattr)
        return self


class Gradient(ScalarField):
    """
    The charge density gradient object from TOPOND. Unit: 'Angstrom' for
    :math:`\\AA^{-4}` and 'a.u.' for Bohr :math:`^{-4}`.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BA, BC (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, struc=None, unit='Angstrom'):
        super().__init__(data, base, struc, 'GRHO', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``Gradient`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (Gradient)
        """
        return super().from_file(file, output, 'GRHO', source)

    def plot_2D(
        self, unit='Angstrom', levels=None, lineplot=True, linewidth=1.0,
        isovalues='%.2f', colorplot=False, colormap='jet', cbar_label='default',
        a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-4}`, 'a.u.' for
                Bohr:math:`^{-4}`. X and y axis scales are changed correspondingly.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. 'None' for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80, 200],
                                  dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # contour line styles
        blimit = -1e-6; rlimit = 1e-6
        if colorplot == False: cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else: cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for i in levels:
            if i < blimit: contourline.append([cb, stlb, linewidth])
            elif i > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if cbar_label=='default':
            if unit.lower() == 'angstrom': ustr = r'$|e|/\AA^{-4}$'
            else: ustr = r'$|e|/Bohr^{-4}$'
            if self.subtracted == False: cbar_label=r'$\nabla\rho$ ({})'.format(ustr)
            else: cbar_label=r'$\Delta(\nabla\rho)$ ({})'.format(ustr)

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Density Gradient'
                else: title = 'Density Gradient + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                unit='Angstrom',
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** charge density gradient with atomic structures
        using `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            unit (str): Plot unit (only for length units here). 'Angstrom' for
                :math:`\\AA^{-4}`, 'a.u.' for Bohr:math:`{-4}`.
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. None for default isovalues (0.01, 1, 100
                or -1, 0, 1 for differences)
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # default isovalues
        if isovalue is None:
            if self.subtracted == False:
                isovalue = np.array([0.01, 1., 100.], dtype=float)
            else:
                isovalue = np.array([-1, 0, 1], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default isovalue are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=grid_display_range,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``Gradient`` object.

        Args:
            unit (str): ''Angstrom', :math:`\\AA^{-4}`; 'a.u.', Bohr :math:`^{-4}`.
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        gprops = ['data'] # density gradient units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        for g in gprops:
            newattr = getattr(self, g) / cst**4
            setattr(self, g, newattr)
        return self


class Laplacian(ScalarField):
    """
    The Laplacian object from TOPOND. Unit: 'Angstrom' for :math:`\\AA^{-5}`
    and 'a.u.' for Bohr :math:`^{-5}`.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BC, BA (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, struc=None, unit='Angstrom'):
        super().__init__(data, base, struc, 'LAPP', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``Laplacian`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

            It is suggested to name the file with 'LAPP' or 'LAPM'. Otherwise
            it will be read as 'LAPP'. Data is always saved as :math:`\\nabla^{2}\\rho`.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (Laplacian)
        """
        if 'LAPM' in file.upper(): type = 'LAPM'
        elif 'LAPP' in file.upper(): type = 'LAPP'
        else:
            warn("Type of data not available from the file name. Using 'LAPP'.",
                 stacklevel=2)
            type = 'LAPP'
        return super().from_file(file, output, type, source)

    def plot_2D(
        self, unit='Angstrom', plot_lapm=False, levels=None, lineplot=True,
        linewidth=1.0, isovalues='%.2f', colorplot=False, colormap='jet',
        cbar_label='default', a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-5}`, 'a.u.' for
                Bohr:math:`^{-5}`. X and y axis scales are changed correspondingly.
            plot_lapm (bool): Whether to plot :math:`-\\nabla^{2}\\rho`.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. 'None' for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # lapm
        if plot_lapm == True:
            self.data = -self.data

        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                               -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                               0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                               0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # contour line styles
        blimit = -1e-6; rlimit = 1e-6
        if colorplot == False:
            cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else:
            cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for i in levels:
            if i < blimit: contourline.append([cb, stlb, linewidth])
            elif i > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if plot_lapm == True: pm = '-'
        else: pm= '+'
        if cbar_label=='default':
            if unit.lower() == 'angstrom': ustr = r'$|e|/\AA^{-5}$'
            else: ustr = r'$|e|/Bohr^{-5}$'
            if self.subtracted == False: cbar_label=r'{}$\nabla^2\rho$ ({})'.format(pm, ustr)
            else: cbar_label=r'$\Delta({}\nabla^2\rho)$ ({})'.format(pm, ustr)

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Density Laplacian ({})'.format(pm)
                else: title = 'Density Laplacian ({}) + {}'.format(pm, titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)

        # restore laplacian
        if plot_lapm == True:
            self.data = -self.data
        return fig

    def plot_3D(self,
                unit='Angstrom',
                plot_lapm=False,
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** charge density Laplacian with atomic structures
        using `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            unit (str): Plot unit (only for length units here). 'Angstrom' for
                :math:`\\AA^{-5}`, 'a.u.' for Bohr:math:`^{-5}`.
            plot_lapm (bool): Whether to plot :math:`-\\nabla^{2}\\rho`.
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. 'None' for default isovalues (-1, 0, 1)
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # LAPM
        if plot_lapm == True: self.data = -self.data

        # default isovalues
        if isovalue is None:
            isovalue = np.array([-1, 0, 1], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default isovalue are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # Plot
        try:
            fig = super().plot_3D(unit,
                                  isovalue=isovalue,
                                  volume_3d=volume_3d,
                                  contour_2d=contour_2d,
                                  interp=interp,
                                  interp_size=interp_size,
                                  grid_display_range=grid_display_range,
                                  **kwargs)
            if plot_lapm == True: self.data = -self.data

        except Exception as e:
            if plot_lapm == True: self.data = -self.data
            raise e

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``Laplacian`` object.

        Args:
            unit (str): ''Angstrom', :math:`\\AA^{-5}`; 'a.u.', Bohr :math:`^{-5}`.
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        laprops = ['data'] # laplacian units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        for la in laprops:
            newattr = getattr(self, la) / cst**5
            setattr(self, la, newattr)
        return self


class HamiltonianKE(ScalarField):
    """
    The Hamiltonian kinetic energy density object from TOPOND. Unit: 'Angstrom'
    for eV.:math:`\\AA^{-3}` and 'a.u.' for Hartree.Bohr :math:`^{-3}`.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BA, BC (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, dimen, struc=None, unit='Angstrom'):
        super().__init__(data, base, struc, 'KKIN', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``HamiltonianKE`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (HamiltonianKE)
        """
        return super().from_file(file, output, 'KKIN', source)

    def plot_2D(
        self, unit='Angstrom', levels=None, lineplot=True, linewidth=1.0,
        isovalues='%.2f', colorplot=False, colormap='jet', cbar_label='default',
        a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit. 'Angstrom' for eV.:math:`\\AA^{-3}`, 'a.u.'
                for Hartree.Bohr :math:`^{-3}`. X and y axis scales are changed
                correspondingly.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. 'None' for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80, 200],
                                  dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # contour line styles
        blimit = -1e-6; rlimit = 1e-6
        if colorplot == False: cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else: cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for i in levels:
            if i < blimit: contourline.append([cb, stlb, linewidth])
            elif i > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if cbar_label=='default':
            if unit.lower() == 'angstrom': ustr = r'$eV/\AA^{-3}$'
            else: ustr = r'$Hartree/Bohr^{-3}$'
            if self.subtracted == False: cbar_label=r'$E_k$ ({})'.format(ustr)
            else: cbar_label=r'$\Delta E_k$ ({})'.format(ustr)

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Hamiltonian Kinetic Energy'
                else: title = 'Hamiltonian Kinetic Energy + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                unit='Angstrom',
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** Hamiltonian kinetic energy density with atomic
        structures using `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_
        (*not installed by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            unit (str): Plot unit. 'Angstrom' for eV.:math:`\\AA^{-3}`, 'a.u.'
                for Hartree.Bohr :math:`^{-3}`.
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. None for default isovalues (0.01, 1, 100
                or -1, 0, 1 for differences)
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # default isovalues
        if isovalue is None:
            if self.subtracted == False:
                isovalue = np.array([0.01, 1., 100.], dtype=float)
            else:
                isovalue = np.array([-1, 0, 1], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default isovalue are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=grid_display_range,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``HamiltonianKE`` object.

        Args:
            unit (str): ''Angstrom', eV.:math:`\\AA^{-3}`; 'a.u.',
                Hartree.Bohr :math:`^{-3}`.
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            ecst = units.H_to_eV(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            ecst = units.eV_to_H(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        eprops = ['data'] # energy density units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        for g in gprops:
            newattr = getattr(self, g) / cst**3 * ecst
            setattr(self, g, newattr)
        return self


class LagrangianKE(ScalarField):
    """
    The Lagrangian kinetic energy density object from TOPOND. Unit: 'Angstrom'
    for eV.:math:`\\AA^{-3}` and 'a.u.' for Hartree.Bohr :math:`^{-3}`.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BC, BA (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, struc=None, unit='Angstrom'):
        super().__init__(data, base, struc, 'GKIN', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``LagrangianKE`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (LagrangianKE)
        """
        return super().from_file(file, output, 'GKIN', source)

    def plot_2D(
        self, unit='Angstrom', levels=None, lineplot=True, linewidth=1.0,
        isovalues='%.2f', colorplot=False, colormap='jet', cbar_label='default',
        a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit. 'Angstrom' for eV.:math:`\\AA^{-3}`, 'a.u.'
                for Eh.Bohr:math:`^{-3}`. X and y axis scales are changed
                correspondingly.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. 'None' for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8,
                                   2, 4, 8, 20, 40, 80, 200], dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # contour line styles
        blimit = -1e-6; rlimit = 1e-6
        if colorplot == False: cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else: cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for i in levels:
            if i < blimit: contourline.append([cb, stlb, linewidth])
            elif i > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if cbar_label=='default':
            if unit.lower() == 'angstrom': ustr = r'$eV/\AA^{-3}$'
            else: ustr = r'$Hartree/Bohr^{-3}$'
            if self.subtracted == False: cbar_label=r'$E_k$ ({})'.format(ustr)
            else: cbar_label=r'$\Delta E_k$ ({})'.format(ustr)

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Lagrangian Kinetic Energy'
                else: title = 'Lagrangian Kinetic Energy + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                unit='Angstrom',
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** Lagrangian kinetic energy density with atomic
        structures using `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_
        (*not installed by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            unit (str): Plot unit. 'Angstrom' for eV.:math:`\\AA^{-3}`, 'a.u.'
                for Eh.Bohr:math:`^{-3}`.
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. None for default isovalues (0.01, 1, 100
                or -1, 0, 1 for differences)
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # default isovalues
        if isovalue is None:
            if self.subtracted == False:
                isovalue = np.array([0.01, 1., 100.], dtype=float)
            else:
                isovalue = np.array([-1, 0, 1], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default isovalue are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=grid_display_range,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``LagrangianKE`` object.

        Args:
            unit (str): ''Angstrom', eV.:math:`\\AA^{-3}`; 'a.u.',
                Hartree.Bohr :math:`^{-3}`.
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            ecst = units.H_to_eV(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            ecst = units.eV_to_H(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        eprops = ['data'] # energy density units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        for g in gprops:
            newattr = getattr(self, g) / cst**3 * ecst
            setattr(self, g, newattr)
        return self


class VirialField(ScalarField):
    """
    The Virial field density object from TOPOND. Unit: 'Angstrom'
    for eV.:math:`\\AA^{-4}` and 'a.u.' for Hartree.Bohr :math:`^{-4}`.

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BA, BC (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).
    """
    def __init__(self, data, base, struc=None, unit='Angstrom'):
        super().__init__(data, base, struc, 'VIRI', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``VirialField`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (VirialField)
        """
        return super().from_file(file, output, 'VIRI', source)

    def plot_2D(
        self, unit='Angstrom', levels=None, lineplot=True, linewidth=1.0,
        isovalues='%.2f', colorplot=False, colormap='jet', cbar_label='default',
        a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit. 'Angstrom' for eV.:math:`\\AA^{-4}`, 'a.u.' for
                Hartree.Bohr :math:`^{-4}`. X and y axis scales are changed
                correspondingly.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. 'None' for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            if self.subtracted == False:
                levels = -np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8,
                                    2, 4, 8, 20, 40, 80, 200], dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # contour line styles
        blimit = -1e-6; rlimit = 1e-6
        if colorplot == False: cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else: cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for i in levels:
            if i < blimit: contourline.append([cb, stlb, linewidth])
            elif i > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if cbar_label=='default':
            if unit.lower() == 'angstrom': ustr = r'$eV/\AA^{-3}$'
            else: ustr = r'$Hartree/Bohr^{-3}$'
            if self.subtracted == False: cbar_label=r'$VF$ ({})'.format(ustr)
            else: cbar_label=r'$\Delta VF$ ({})'.format(ustr)

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Virial Field'
                else: title = 'Virial Field + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                unit='Angstrom',
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** Virial Field with atomic structures using
        `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            unit (str): Plot unit (only for length units here). 'Angstrom' for
                eV.:math:`\\AA^{-4}`, 'a.u.' for Eh.Bohr:math:`^{-4}`.
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. None for default isovalues (0.01, 1, 100
                or -1, 0, 1 for differences)
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # default isovalues
        if isovalue is None:
            if self.subtracted == False:
                isovalue = np.array([0.01, 1., 100.], dtype=float)
            else:
                isovalue = np.array([-1, 0, 1], dtype=float)
            if unit.lower() != 'angstrom':
                warn("Unit must be 'Angstrom' when the default isovalue are set. Using Angstrom-eV units.", stacklevel=2)
                unit = 'Angstrom'

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=grid_display_range,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``VirialField`` object.

        Args:
            unit (str): ''Angstrom', eV.:math:`\\AA^{-3}`; 'a.u.',
                Hartree.Bohr :math:`^{-3}`.
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            ecst = units.H_to_eV(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            ecst = units.eV_to_H(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        eprops = ['data'] # energy density units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        for g in gprops:
            newattr = getattr(self, g) / cst**4 * ecst
            setattr(self, g, newattr)
        return self


class ELF(ScalarField):
    """
    The electron localization object from TOPOND. Dimensionless. Unit: 'Angstrom'
    for :math:`\\AA` and 'a.u.' for Bohr (only for plot base vectors).

    Args:
        data (array): 2D (3D) Plot data. (nZ\*)nY\*nX.
        base (array): 3(4)\*3 Cartesian coordinates of the 3(4) points defining
            base vectors BC, BA (2D) or OA, OB, OC (3D). The sequence is (O),
            A, B, C.
        struc (CStructure): Extended Pymatgen Structure object.
    """
    def __init__(self, data, base, struc=None, unit='dimensionless'):
        super().__init__(data, base, struc, 'ELFB', unit)

    @classmethod
    def from_file(cls, file, output=None, source='crystal'):
        """
        Generate a ``ELF`` object from a single file.

        .. note::

            Output is not mandatory for 2D plottings. But it is highly
            recommended to be added for geometry information and other methods.

        Args:
            file (str): The scalar field data.
            output (str): Supplementary output file to help define the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (ELF)
        """
        return super().from_file(file, output, 'ELFB', source)

    def plot_2D(
        self, unit='Angstrom', levels=None, lineplot=True, linewidth=1.0,
        isovalues='%.2f', colorplot=False, colormap='jet', cbar_label='default',
        a_range=[0., 1.], b_range=[0., 1.], edgeplot=False,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None,
        **kwargs):
        """
        Plot 2D contour lines, color maps or both for the 2D data set. The user
        can also get the overlapped plot of ``ScalarField`` and ``Trajectory``
        classes.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.ScalarField``.

        Args:
            unit (str): Plot unit, only for x and y axis scales. 'Angstrom' for
                :math:`\\AA`, 'a.u.' for Bohr.
            levels (array|int): Set levels of colored / line contour plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D. 'None' for default levels.
            lineplot (bool): Plot contour lines.
            contourline (list): Width of contour lines. Other styles are pre-
                set and cannot be changed.
            isovalues (str|None): Add isovalues to contour lines and set their
                formats. Useful only if ``lineplot=True``. None for not adding
                isovalues.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if
                ``colorplot=True``.
            cbar_label (str): Label of colorbar. Useful only if
                ``colorplot=True``. 'None' for no labels.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            edgeplot (bool): Whether to add cell boundaries represented by the
                original base vectors (not inflenced by a/b range).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for default values and
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|Trajectory): Overlapping a 2D plot from the
                ``topond.Trajectory`` object if not None.
            \*\*kwargs : Other arguments passed to ``topond.Trajectory``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.

        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # default levels
        if type(levels) == str and levels.lower() == 'default': levels=None # Compatibility
        if levels is None:
            if self.subtracted == False:
                levels = np.linspace(0, 1, 21)
                blimit = 0.5-1e-6; rlimit = 0.5+1e-6
            else:
                levels = np.array([-1.5, -1.0, -0.5, -0.1, -0.05, -0.01, -0.005, -0.001,
                                   0,
                                   0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5], dtype=float)
                blimit = -1e-6; rlimit = 1e-6

        # contour line styles
        if colorplot == False: cb = 'b'; stlb = 'dotted'; cr = 'r'; stlr = '-'
        else: cb = 'k'; stlb = 'dotted'; cr = 'k'; stlr = '-'

        contourline = []
        for l in levels:
            if l < blimit: contourline.append([cb, stlb, linewidth])
            elif l > rlimit: contourline.append([cr, stlr, linewidth])
            else: contourline.append(['k', '-', linewidth*2])

        # cbar label
        if cbar_label=='default':
            if self.subtracted == False: cbar_label=r'ELF'
            else: cbar_label=r'$\Delta$ ELF'

        # plot
        fig = super().plot_2D(unit, levels, lineplot, contourline, isovalues,
                              colorplot, colormap, cbar_label, a_range, b_range,
                              edgeplot, x_ticks, y_ticks, figsize, overlay, **kwargs)

        # label and title
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if overlay is None: title = 'Electron Localization'
                else: title = 'Electron Localization + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
                grid_display_range=[[0,1], [0,1], [0,1]],
                show_the_scene=True,
                **kwargs):
        """
        Visualize **2D or 3D** electron localization function with atomic
        structures using `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_
        (*not installed by default*).

        * For 2D charge/spin densities, plot 2D heatmap with/without contour lines.
        * For 3D charge/spin densities, plot 3D isosurfaces or volumetric data.

        Args:
            isovalue (float|array): Isovalues of 3D/2D contour plots. A number
                or an array for user-defined values of isosurfaces, **must be
                consistent with ``unit``**. None for default isovalues (0.2 0.5 0.8
                or -0.1, 0, 0.1 for differences)
            volume_3d (bool): *3D only*. Display 3D volumetric data instead of
                isosurfaces. ``isovalue`` is disabled.
            contour_2d (bool): *2D only* Display 2D black contour lines over
                colored contour surfaces.
            interp (str): Interpolate data to smoothen the plot. 'no interp' or
                'linear', 'nearest', 'slinear', 'cubic'. please refer to
                `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
                 The interpolated data is not saved.
            interp_size (list[int]|int): The new size of interpolated data
                (list) or a scaling factor. *Valid only when ``interp`` is not
                'no interp'*.
            grid_display_range (array): 3\*2 array defining the displayed
                region of the data grid. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and
                return None. Otherwise return the scene object.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            colormap (turple|str): Colormap of isosurfaces/heatmaps. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            color (turple): Color of contour lines. *'contour_2d=True' only*.
            line_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            vmax (float): Maximum value of colormap.
            vmin (float): Minimum value of colormap.
            title (str): Colorbar title.
            orientation (str): Orientation of colorbar, 'horizontal' or 'vertical'.
            nb_labels (int): The number of labels to display on the colorbar.
            label_fmt (str): The string formater for the labels, e.g., '%.1f'.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # default isovalues
        if isovalue is None:
            if self.subtracted == False:
                isovalue = np.array([0.2, 0.5, 0.8], dtype=float)
            else:
                isovalue = np.array([-0.1, 0, 0.1], dtype=float)

        # Plot
        fig = super().plot_3D(unit=self.unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=grid_display_range,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """Of no practical use, since ELF is dimensionless."""
        self.unit = 'dimensionless'
        return self


class GradientTraj(Trajectory):
    """
    The charge density gradient trajectory object from TOPOND. Unit: 'Angstrom'
    for :math:`\\AA` and 'a.u.' for Bohr.

    Args:
        wtraj (list[int]): 1\*nPath, weight of the path, int 0 to 3.
        traj (list[array]): 1\*nPath, list of paths. Every array is a nPoint\*3
            3D ref framework coordinates of points on the path.
        base (array): 3\*3 Cartesian coordinates of the 3 points defining
            vectors BA and BC.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).

    Returns:
        self (GradientTraj): Import attributes: ``trajectory``, nPath\*2 list of
            trajectory and its weight; ``cpoint``, nCpoint\*3 array of critical
            point coordinates in 3D ref framework.
    """
    def __init__(self, wtraj, traj, base, struc, unit='Angstrom'):
        super().__init__(wtraj, traj, base, struc, 'TRAJGRAD', unit)

    @classmethod
    def from_file(cls, file, output, source='crystal'):
        """
        Generate a ``GradientTraj`` object from 'TRAJGRAD.DAT' file and the
        standard screen output of CRYSTAL 'properties' calculation.

        .. note::

            Output file is mandatory.

        Args:
            file (str): The trajectory data.
            output (str): Output file for the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (GradientTraj)
        """
        return super().from_file(file, output, 'TRAJGRAD', source)

    def plot_2D(
        self, unit='Angstrom', traj_color='r', traj_linestyle=':', traj_linewidth=0.5,
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None, **kwargs):
        """
        Plot charge density gradient trajectory in a 2D plot.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.Trajectory``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA`, 'a.u.' for Bohr.
            traj_color (str): Line color of 2D trajectory plot.
            traj_linestyl (str): Line style of 2D trajectory plot.
            traj_linewidth (str): Line width of 2D trajectory plot.
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for proeprty plotted.
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|ScalarField): Overlapping a 2D plot from the
                ``topond.ScalarField`` object if not None.
            \*\*kwargs : Other arguments passed to ``ScalarField.plot_2D()``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.
        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # plot
        fig = super().plot_2D(unit, None, 'k', None, 0, traj_color,
                              traj_linestyle, traj_linewidth, x_ticks, y_ticks,
                              figsize, overlay, fig, ax_index, **kwargs)
        # labels and titles
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'RHOO' : 'Charge Density',
                      'SPDE' : 'Spin Density',
                      'GRHO' : 'Density Gradient',
                      'LAPP' : 'Density Laplacian',
                      'KKIN' : 'Hamiltonian Kinetic Eenergy',
                      'GKIN' : 'Lagrangian Kinetic Eenergy',
                      'VIRI' : 'Virial Field',
                      'ELFB' : 'Electron Localization'}
            if title == 'default':
                if overlay is None: title = 'Gradient Trajectory'
                else: title = '{} + Gradient Trajectory'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)

        return fig

    def plot_3D(self,
                traj_color=(1., 0., 0.),
                traj_linewidth=1.5,
                show_the_scene=True,
                **kwargs):
        """
        Plot 3D chemical graph with atomic structures. The plot unit is 'Angstrom'.
        Unit conversion is automatically performed.

        Args:
            traj_color (tuple): 1\*3 tuple of plotted trajectory colors.
            traj_linewidth (float): Linewidth of trajectories.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # Plot
        fig = super().plot_3D(unit='Angstrom',
                              cpt_marker=None,
                              cpt_color=(0.35, 0.35, 0.35),
                              cpt_size=None,
                              traj_weight=0,
                              traj_color=traj_color,
                              traj_linewidth=traj_linewidth,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``GradientTraj`` object.

        Args:
            unit (str): 'Angstrom' or 'a.u.'
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = ['cpoint'] # length units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        # special: trajectory
        self.trajectory = [[i[0], i[1]*cst] for i in self.trajectory]
        return self


class ChemicalGraph(Trajectory):
    """
    The molecular / crystal graph object from TOPOND. Unit: 'Angstrom' for
    :math:`\\AA` and 'a.u.' for Bohr. Also compatible with chemical graph + 
    gradient trajectories.

    Args:
        wtraj (list[int]): 1\*nPath, weight of the path, int 0 to 3.
        traj (list[array]): 1\*nPath, list of paths. Every array is a nPoint\*3
            3D ref framework coordinates of points on the path.
        base (array): 3\*3 Cartesian coordinates of the 3 points defining
            vectors BA and BC.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'Angstrom' (case insensitive).

    Returns:
        self (ChemicalGraph): Import attributes: ``trajectory``, nPath\*2 list
            of trajectory and its weight; ``cpoint``, nCpoint\*3 array of
            critical point coordinates in 3D ref framework.
    """
    def __init__(self, wtraj, traj, base, struc, unit='Angstrom'):
        super().__init__(wtraj, traj, base, struc, 'TRAJGRAD', unit)

    @classmethod
    def from_file(cls, file, output, source='crystal'):
        """
        Generate a ``ChemicalGraph`` object from 'TRAJMOLG.DAT' file and the
        standard screen output of CRYSTAL 'properties' calculation.

        .. note::

            Output file is mandatory.

        Args:
            file (str): The trajectory data.
            output (str): Output file for the geometry and periodicity.
            source (str): Currently not used. Saved for future development.
        Returns:
            cls (ChemicalGraph)
        """
        return super().from_file(file, output, 'TRAJMOLG', source)

    def plot_2D(
        self, unit='Angstrom', cpt_marker='o', cpt_color='k', cpt_size=10,
        traj_weight=[0,1,2,3], traj_color='default', traj_linestyle='default', traj_linewidth='default',
        x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None, **kwargs):
        """
        Plot crystal / molecular graph in a 2D plot.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.Trajectory``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA`, 'a.u.' for Bohr.
            cpt_marker (str): Marker of critical point scatter. 'None' to turn
                off critical point plotting.
            cpt_color (str): Marker color of critical point scatter.
            cpt_size (float|int): Marker size of critical point scatter.
            traj_weight (int|list): Weight(s) of the plotted trajectories (0 to 3).
            traj_color (str|list): 1\*nTraj_weight list of plotted trajectory
                colors. Use string for the same color. 'default' for default
                styles, 'cpt_color' for weight = 3 and 'r' for others.
            traj_linestyle (str|list): 1\*nTraj_weight list of plotted trajectory
                styles. Use string for the same style. 'default' for default
                styles. 'solid' for weight = 3 and 'dotted' for others.
            traj_linewidth (str): Line width of 2D trajectory plot. 'default'
                for default styles. 1.0 for weight = 3 and 0.5 for others.
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for proeprty plotted.
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|ScalarField): Overlapping a 2D plot from the
                ``topond.ScalarField`` object if not None.
            \*\*kwargs : Other arguments passed to ``ScalarField.plot_2D()``, and
                *developer only* arguments, ``fig`` and ``ax_index``, to pass
                figure object and axis index.
        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # plot
        fig = super().plot_2D(unit, cpt_marker, cpt_color, cpt_size, traj_weight,
                              traj_color, traj_linestyle, traj_linewidth,
                              x_ticks, y_ticks, figsize, overlay, **kwargs)
        # labels and titles
        if 'ax_index' in kwargs.keys():
            iax = kwargs['ax_index']
        else:
            iax = 0

        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if title is not None:
            titles = {'RHOO' : 'Charge Density',
                      'SPDE' : 'Spin Density',
                      'GRHO' : 'Density Gradient',
                      'LAPP' : 'Density Laplacian',
                      'KKIN' : 'Hamiltonian Kinetic Eenergy',
                      'GKIN' : 'Lagrangian Kinetic Eenergy',
                      'VIRI' : 'Virial Field',
                      'ELFB' : 'Electron Localization'}
            if title == 'default':
                if overlay is None: title = 'Chemical Graph'
                else: title = '{} + Chemical Graph'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)

        return fig

    def plot_3D(self,
                cpt_marker='sphere',
                cpt_color=(0.35, 0.35, 0.35),
                cpt_size=0.5,
                traj_weight=3,
                traj_color='default',
                traj_linewidth='default',
                show_the_scene=True,
                **kwargs):
        """
        Plot 3D chemical graph with atomic structures. The plot unit is 'Angstrom'.
        Unit conversion is automatically performed.

        Args:
            cpt_marker (string): The mode of the glyphs. Passed to ``mode`` of
                mayavi ``mlab.points3d()``. Accepted values: '2darrow',
                '2dcircle', '2dcross', '2ddash', '2ddiamond', '2dhooked_arrow',
                '2dsquare', '2dthick_arrow', '2dthick_cross', '2dtriangle',
                '2dvertex’, 'arrow', 'axes', 'cone', 'cube', 'cylinder', 'point',
                'sphere'. 'None' to turn off critical point plotting.
            cpt_color (tuple): 1\*3 1D tuple for the color of critical points.
            cpt_size (float): Size of critical points.
            traj_weight (int|list): Weight(s) of the plotted trajectories (0 to 3).
            traj_color (tuple|str): nTraj_weight\*3 list of plotted trajectory
                colors. Use 1\*3 1D tuple for the same color. 'default' for default
                styles, 'cpt_color' for weight = 3 and '(1,0,0)' for others.
            traj_linewidth (float): Linewidth of trajectories. 'default' for default
                styles, 3.0 for weight = 3 and 1.5 for others.
            \*\*kwargs: Optional keywords passed to MayaVi or ``CStructure.visualize()``.
                Allowed keywords are listed below.
            atom_color (str): Color map of atoms. 'jmol' or 'cpk'.
            bond_color (turple): Color of bonds, in a 1\*3 RGB turple from 0 to 1.
            atom_bond_ratio (str): 'balls', 'large', 'medium', 'small' or
                'sticks'. The relative sizes of balls and sticks.
            cell_display (bool): Display lattice boundaries (at \[0., 0., 0.\] only).
            cell_color (turple): Color of lattice boundaries, in a 1\*3 RGB turple from 0 to 1.
            cell_linewidth (float): Linewidth of plotted lattice boundaries.
            display_range (array): 3\*2 array defining the displayed region of
                the structure. Fractional coordinates a, b, c are used but only
                the periodic directions are applied.
            scale (float): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            azimuth: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            elevation: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            distance: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
                By default set to 'auto'.
            focalpoint: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
            roll: See `mlab.view() <https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view>`_.
        Returns:
            fig: MayaVi scence object, if ``show_the_scene=False``.
        """
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MayaVi is required for this functionality, which is not in the default dependency list of CRYSTALpytools.')

        # Plot
        fig = super().plot_3D(unit='Angstrom',
                              cpt_marker=cpt_marker,
                              cpt_color=cpt_color,
                              cpt_size=cpt_size,
                              traj_weight=traj_weight,
                              traj_color=traj_color,
                              traj_linewidth=traj_linewidth,
                              **kwargs)

        # Final setups
        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: keywords[k] = v
        mlab.view(**keywords)
        mlab.gcf().scene.parallel_projection = True

        if show_the_scene == False:
            return fig
        else:
            mlab.show()
            return

    def _set_unit(self, unit):
        """
        Set units of data of ``ChemicalGraph`` object.

        Args:
            unit (str): 'Angstrom' or 'a.u.'
        """
        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'angstrom':
            cst = units.au_to_angstrom(1.)
            self.unit = 'Angstrom'
        elif unit.lower() == 'a.u.':
            cst = units.angstrom_to_au(1.)
            self.unit = 'a.u.'
        else:
            raise ValueError('Unknown unit.')

        lprops = ['cpoint'] # length units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        # special: trajectory
        self.trajectory = [[i[0], i[1]*cst] for i in self.trajectory]
        return self

