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
        self.type = type
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
        from CRYSTALpytools.base.plotbase import plot_2Dscalar
        import matplotlib.pyplot as plt

        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower():
            self._set_unit(unit)

        # dimen
        if self.dimension != 2:
            self._set_unit(uold)
            raise Exception('Not a 2D scalar field object.')

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
                self._set_unit(uold)
                raise ValueError("The overlaied layer must be a topond.Trajectory class or its child classes.")

            overlay._set_unit(unit)
            diff_base = np.abs(overlay.base-self.base)
            if np.any(diff_base>1e-3):
                self._set_unit(uold)
                raise Exception("The plotting base of surface and trajectory are different.")
            a_range = [0., 1.]; b_range=[0., 1.] # no periodicity for Traj

        # plot
        ## layout
        if 'fig' not in kwargs.keys():
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
            ax_index = 0
        else:
            if 'ax_index' not in kwargs.keys():
                self._set_unit(uold)
                raise ValueError("Indices of axes must be set when 'fig' is passed.")
            ax_index = int(kwargs['ax_index'])
            fig = kwargs['fig']
            ax = fig.axes[ax_index]
        ## surf first
        if unit.lower() == 'angstrom': pltbase = self.base
        else: pltbase = units.angstrom_to_au(self.base)
        try:
            fig = plot_2Dscalar(
                fig, ax, self.data, pltbase, levels, contourline, isovalues, colormap,
                cbar_label, a_range, b_range, False, edgeplot, x_ticks, y_ticks
            )
        except Exception as e:
            self._set_unit(uold)
            raise e

        ## plot traj
        if np.all(overlay!=None):
            kwargs['fig'] = fig; kwargs['ax_index'] = ax_index
            kwargs['y_ticks'] = y_ticks; kwargs['x_ticks'] = x_ticks
            kwargs['figsize'] = figsize; kwargs['unit'] = overlay.unit # convert unit in wrappers!!!
            try:
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

        # structure
        if self.structure == None:
            self._set_unit(uold)
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
                'cell_color', 'cell_linewidth', 'display_range', 'scale',
                'special_bonds']
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k in keys: inputs[k] = v

        try:
            fig = plot_GeomScalar(**inputs)
        except Exception as e:
            self._set_unit(uold)
            raise e
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
        """"no practical use."""
        pass


class Trajectory():
    """
    Basic TOPOND trajectory plot class. Call the property-specific child classes
    below to use.
    """
    def plot_2D(
        self, cpt_marker='o', cpt_color='k', cpt_size=10,
        traj_color='r', traj_linestyle=':', traj_linewidth=0.5, x_ticks=5,
        y_ticks=5, figsize=[6.4, 4.8], overlay=None, fig=None, ax_index=None,
        **kwargs):
        """
        Get TOPOND trajectory in a 2D plot.

        .. note::

            2D periodicity (``a_range`` and ``b_range``) is not available for
            the ``Trajectory`` class. If ``overlay`` is not None, ``a_range``
            and ``b_range`` and ``edgeplot`` will be disabled for the
            ``ScalarField`` object.

        Args:
            cpt_marker (str): Marker of critical point scatter.
            cpt_color (str): Marker color of critical point scatter.
            cpt_size (float|int): Marker size of critical point scatter.
            traj_color (str): Line color of 2D trajectory plot.
            traj_linestyl (str): Line style of 2D trajectory plot.
            traj_linewidth (str): Line width of 2D trajectory plot.
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            overlay (None|ScalarField): Overlapping a 2D plot from the
                ``topond.ScalarField`` object if not None.
            fig (Figure|None): *Developers only*, matplotlib Figure class..
            ax_index (list[int]): *Developer Only*, indices of axes in ``fig.axes``.
            \*\*kwargs : Other arguments passed to ``ScalarField.plot_2D()``.
        Returns:
            fig (Figure): Matplotlib Figure object
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import copy
        from CRYSTALpytools.base.plotbase import GridRotation2D

        # overlay
        if np.all(overlay!=None) and isinstance(overlay, ScalarField):
            diff_base = np.abs(overlay.base-self.base)
            if np.any(diff_base>1e-3):
                raise Exception("The plotting base of surface and trajectory are different.")

        # plot
        ## layout
        if np.all(fig==None):
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
            ax_index = 0
        else:
            if np.all(ax_index==None):
                raise ValueError("Indices of axes must be set when 'fig' is not None.")
            ax_index = int(ax_index)
            ax = fig.axes[ax_index]

        ## Get bottom surf figure first
        if np.all(overlay!=None) and isinstance(overlay, ScalarField):
            kwargs['a_range'] = [0., 1.]; kwargs['b_range'] = [0., 1.] # no periodicity
            kwargs['edgeplot'] = False; kwargs['figsize'] = figsize
            kwargs['fig'] = fig; kwargs['ax_index'] = ax_index;
            kwargs['x_ticks'] = x_ticks; kwargs['y_ticks'] = y_ticks
            kwargs['unit'] = overlay.unit # convert unit in wrappers!!!

            fig = overlay.plot_2D(**kwargs)
            ax = fig.axes[ax_index]

        ## rotate the trajectory to plotting plane, base defined in OAB
        rot, disp = GridRotation2D(np.vstack([self.base[1], self.base[2], self.base[0]]))
        ## plot TRAJ
        baserot = rot.apply(self.base)
        xmx = np.linalg.norm(baserot[2, :]-baserot[1, :])
        ymx = np.linalg.norm(baserot[0, :]-baserot[1, :])
        extra_width = {1 : 0., 2 : 0., 3 : 0.5} # extra linewidth for critical path
        for wt, traj in self.trajectory:
            traj = rot.apply(traj)
            # plot CPT
            if len(traj) == 1:
                v = traj[0] - baserot[1]
                if v[0]>=0 and v[0]<xmx and v[1]>=0 and v[1]<ymx and np.abs(v[2])<=1e-3:
                    ax.scatter(traj[0,0], traj[0,1], marker=cpt_marker, c=cpt_color, s=cpt_size)
            # plot TRAJ
            else:
                plttraj = [] # traj in plot plane
                for v in traj:
                    v = v - baserot[1]
                    if v[0]>=0 and v[0]<xmx and v[1]>=0 and v[1]<ymx and np.abs(v[2])<=1e-3:
                        plttraj.append(v+baserot[1])

                if len(plttraj) == 0:
                    continue
                plttraj = np.array(plttraj)
                if wt != 0: # Molegraph path
                    ax.plot(plttraj[:, 0], plttraj[:, 1], color=cpt_color,
                            linestyle='-', linewidth=traj_linewidth+extra_width[wt])
                else: # other pathes
                    ax.plot(plttraj[:, 0], plttraj[:, 1], color=traj_color,
                            linestyle=traj_linestyle, linewidth=traj_linewidth)

        ax.set_aspect(1.0)
        ax.set_xlim(baserot[1, 0], baserot[1, 0]+xmx)
        ax.set_ylim(baserot[1, 1], baserot[1, 1]+ymx)
        return fig


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
        self, unit='Angstrom', levels='default', lineplot=True, linewidth=1.0,
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
                user-defined levels. 1D. 'default' for default levels.
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
        # unit
        if np.all(levels=='default') and unit.lower() != 'angstrom':
            warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.",
                 stacklevel=2)
            unit = 'Angstrom'

        # default levels
        if np.all(levels=='default'):
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80, 200],
                                  dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Charge Density'
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        self, unit='Angstrom', levels='default', lineplot=True, linewidth=1.0,
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
                user-defined levels. 1D. 'default' for default levels.
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
        # unit
        if np.all(levels=='default') and unit.lower() != 'angstrom':
            warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.",
                 stacklevel=2)
            unit = 'Angstrom'

        # default levels
        if np.all(levels=='default'):
            levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Spin Density'
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        self, unit='Angstrom', levels='default', lineplot=True, linewidth=1.0,
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
                user-defined levels. 1D. 'default' for default levels.
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
        # unit
        if np.all(levels=='default') and unit.lower() != 'angstrom':
            warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.",
                 stacklevel=2)
            unit = 'Angstrom'

        # default levels
        if np.all(levels=='default'):
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80, 200],
                                  dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Density Gradient'
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        self, unit='Angstrom', plot_lapm=False, levels='default', lineplot=True,
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
                user-defined levels. 1D. 'default' for default levels.
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
        # unit
        if np.all(levels=='default') and unit.lower() != 'angstrom':
            warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.",
                 stacklevel=2)
            unit = 'Angstrom'

        # lapm
        if plot_lapm == True:
            self.data = -self.data

        # default levels
        if np.all(levels=='default'):
            levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                               -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                               0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                               0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Density Laplacian ({})'.format(pm)
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        try:
            fig = super().plot_3D(unit,
                                  isovalue=isovalue,
                                  volume_3d=volume_3d,
                                  contour_2d=contour_2d,
                                  interp=interp,
                                  interp_size=interp_size,
                                  grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        self, unit='Angstrom', levels='default', lineplot=True, linewidth=1.0,
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
                user-defined levels. 1D. 'default' for default levels.
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
        # unit
        if np.all(levels=='default') and unit.lower() != 'angstrom':
            warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.",
                 stacklevel=2)
            unit = 'Angstrom'

        # default levels
        if np.all(levels=='default'):
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80, 200],
                                  dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Hamiltonian Kinetic Energy'
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        self, unit='Angstrom', levels='default', lineplot=True, linewidth=1.0,
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
                user-defined levels. 1D. 'default' for default levels.
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
        # unit
        if np.all(levels=='default') and unit.lower() != 'angstrom':
            warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.",
                 stacklevel=2)
            unit = 'Angstrom'

        # default levels
        if np.all(levels=='default'):
            if self.subtracted == False:
                levels = np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8,
                                   2, 4, 8, 20, 40, 80, 200], dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Lagrangian Kinetic Energy'
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        self, unit='Angstrom', levels='default', lineplot=True, linewidth=1.0,
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
                user-defined levels. 1D. 'default' for default levels.
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
        # unit
        if np.all(levels=='default') and unit.lower() != 'angstrom':
            warn("Unit must be 'Angstrom' when the default levels are set. Using Angstrom-eV units.",
                 stacklevel=2)
            unit = 'Angstrom'

        # default levels
        if np.all(levels=='default'):
            if self.subtracted == False:
                levels = -np.array([0.02, 0.04, 0.08, 0.2, 0.4, 0.8,
                                    2, 4, 8, 20, 40, 80, 200], dtype=float)
            else:
                levels = np.array([-80, -40, -20, -8, -4, -2, -0.8, -0.4, -0.2,
                                   -0.08, -0.04, -0.02, -0.008, -0.004, -0.002,
                                   0, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
                                   0.2, 0.4, 0.8, 2, 4, 8, 20, 40, 80], dtype=float)
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Virial Field'
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        fig = super().plot_3D(unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        self, unit='Angstrom', levels='default', lineplot=True, linewidth=1.0,
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
                user-defined levels. 1D. 'default' for default levels.
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
        if np.all(levels=='default'):
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
        fig = super().plot_2D(self.unit, levels, lineplot, contourline, isovalues,
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

        if np.all(title!=None):
            titles = {'TRAJGRAD' : 'Gradient Trajectory',
                      'TRAJMOLG' : 'Chemical Graph'}
            if title == 'default':
                if np.all(overlay==None): title = 'Electron Localization'
                else: title = 'Electron Localization + {}'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self,
                isovalue=None,
                volume_3d=False,
                contour_2d=False,
                interp='no interp',
                interp_size=1,
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
                consistent with ``unit``**. By default half between max and min
                values.
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

        # Plot
        fig = super().plot_3D(unit=self.unit,
                              isovalue=isovalue,
                              volume_3d=volume_3d,
                              contour_2d=contour_2d,
                              interp=interp,
                              interp_size=interp_size,
                              grid_display_range=[[0,1], [0,1], [0,1]], # disable grid periodicity
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
        if len(wtraj) != len(traj):
            raise ValueError('Inconsistent lengths of input trajectory and its weight.')

        self.trajectory = [[int(wtraj[i]), np.array(traj[i], dtype=float)]
                           for i in range(len(wtraj))]
        self.base = np.array(base, dtype=float)
        self.structure = struc
        self.type = 'TRAJGRAD'
        self.unit = unit
        cpt = []
        for i in self.trajectory:
            if len(i[1]) == 1:
                cpt.append(i[1][0])
        self.cpoint = np.array(cpt)

    @classmethod
    def from_file(cls, file, output):
        """
        Generate a ``GradientTraj`` object from 'TRAJGRAD.DAT' file and the
        standard screen output of CRYSTAL 'properties' calculation.

        .. note::

            Output file is mandatory.

        Args:
            file (str): 'TRAJGRAD.DAT' file by TOPOND.
            output (str): Standard output of Properties calculation, used to
                get geometry and orientation of the plane.
        Returns:
            cls (ChargeDensity)
        """
        from CRYSTALpytools.crystal_io import Properties_output

        return Properties_output(output).read_topond(file, 'TRAJGRAD')

    def plot_2D(
        self, unit='Angstrom', cpt_marker='o', cpt_color='k', cpt_size=10,
        traj_color='r', traj_linestyle=':', traj_linewidth=0.5, x_ticks=5,
        y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None, fig=None,
        ax_index=None, **kwargs):
        """
        Plot charge density gradient trajectory in a 2D plot.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.Trajectory``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-3}`, 'a.u.' for
                Bohr :math:`^{-3}`.
            cpt_marker (str): Marker of critical point scatter.
            cpt_color (str): Marker color of critical point scatter.
            cpt_size (float|int): Marker size of critical point scatter.
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
            fig (Figure|None): *Developers only*, matplotlib Figure class..
            ax_index (list[int]): *Developer Only*, indices of axes in ``fig.axes``.
            \*\*kwargs : Other arguments passed to ``ScalarField.plot_2D()``.
        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower():
            self._set_unit(unit)
        if np.all(overlay!=None):
            overlay._set_unit(unit)

        # axis index
        if np.all(fig!=None) and np.all(ax_index!=None):
            iax = int(ax_index)
        else:
            iax = 0

        # plot
        fig = super().plot_2D(cpt_marker, cpt_color, cpt_size, traj_color,
                              traj_linestyle, traj_linewidth, x_ticks, y_ticks,
                              figsize, overlay, fig, ax_index, **kwargs)
        # labels and titles
        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if np.all(title!=None):
            titles = {'SURFRHOO' : 'Charge Density',
                      'SURFSPDE' : 'Spin Density',
                      'SURFGRHO' : 'Density Gradient',
                      'SURFLAPP' : 'Density Laplacian',
                      'SURFKKIN' : 'Hamiltonian Kinetic Eenergy',
                      'SURFGKIN' : 'Lagrangian Kinetic Eenergy',
                      'SURFVIRI' : 'Virial Field',
                      'SURFELFB' : 'Electron Localization'}
            if title == 'default':
                if np.all(overlay==None): title = 'Gradient Trajectory'
                else: title = '{} + Gradient Trajectory'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)

        self._set_unit(uold)
        if np.all(overlay!=None): overlay._set_unit(uold)
        return fig

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

        lprops = ['base', 'cpoint'] # length units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        # special: trajectory
        self.trajectory = [[i[0], i[1]*cst] for i in self.trajectory]
        return self


class ChemicalGraph(Trajectory):
    """
    The molecular / crystal graph object from TOPOND. Unit: 'Angstrom' for
    :math:`\\AA` and 'a.u.' for Bohr.

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
        if len(wtraj) != len(traj):
            raise ValueError('Inconsistent lengths of input trajectory and its weight.')

        self.trajectory = [[int(wtraj[i]), np.array(traj[i], dtype=float)]
                           for i in range(len(wtraj))]
        self.base = np.array(base, dtype=float)
        self.structure = struc
        self.type = 'TRAJMOLG'
        self.unit = unit
        cpt = []
        for i in self.trajectory:
            if len(i[1]) == 1:
                cpt.append(i[1][0])
        self.cpoint = np.array(cpt)

    @classmethod
    def from_file(cls, file, output):
        """
        Generate a ``ChemicalGraph`` object from 'TRAJMOLG.DAT' file and the
        standard screen output of CRYSTAL 'properties' calculation.

        .. note::

            Output file is mandatory.

        Args:
            file (str): 'TRAJMOLG.DAT' file by TOPOND.
            output (str): Standard output of Properties calculation, used to
                get geometry and orientation of the plane.
        Returns:
            cls (ChargeDensity)
        """
        from CRYSTALpytools.crystal_io import Properties_output

        return Properties_output(output).read_topond(file, 'TRAJMOLG')

    def plot_2D(
        self, unit='Angstrom', cpt_marker='o', cpt_color='k', cpt_size=10,
        traj_color='r', traj_linestyle=':', traj_linewidth=0.5, x_ticks=5,
        y_ticks=5, title='default', figsize=[6.4, 4.8], overlay=None, fig=None,
        ax_index=None, **kwargs):
        """
        Plot crystal / molecular graph in a 2D plot.

        .. note::

            For more information of plot setups, please refer its parent class,
            ``topond.Trajectory``.

        Args:
            unit (str): Plot unit. 'Angstrom' for :math:`\\AA^{-3}`, 'a.u.' for
                Bohr :math:`^{-3}`.
            cpt_marker (str): Marker of critical point scatter.
            cpt_color (str): Marker color of critical point scatter.
            cpt_size (float|int): Marker size of critical point scatter.
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
            fig (Figure|None): *Developers only*, matplotlib Figure class..
            ax_index (list[int]): *Developer Only*, indices of axes in ``fig.axes``.
            \*\*kwargs : Other arguments passed to ``ScalarField.plot_2D()``.
        Returns:
            fig (Figure): Matplotlib Figure object
        """
        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower():
            self._set_unit(unit)
        if np.all(overlay!=None):
            overlay._set_unit(unit)

        # axis index
        if np.all(fig!=None) and np.all(ax_index!=None):
            iax = int(ax_index)
        else:
            iax = 0

        # plot
        fig = super().plot_2D(cpt_marker, cpt_color, cpt_size, traj_color,
                              traj_linestyle, traj_linewidth, x_ticks, y_ticks,
                              figsize, overlay, fig, ax_index, **kwargs)
        # labels and titles
        if unit.lower() == 'angstrom':
            fig.axes[iax].set_xlabel(r'$\AA$')
            fig.axes[iax].set_ylabel(r'$\AA$')
        else:
            fig.axes[iax].set_xlabel(r'$Bohr$')
            fig.axes[iax].set_ylabel(r'$Bohr$')

        if np.all(title!=None):
            titles = {'SURFRHOO' : 'Charge Density',
                      'SURFSPDE' : 'Spin Density',
                      'SURFGRHO' : 'Density Gradient',
                      'SURFLAPP' : 'Density Laplacian',
                      'SURFKKIN' : 'Hamiltonian Kinetic Eenergy',
                      'SURFGKIN' : 'Lagrangian Kinetic Eenergy',
                      'SURFVIRI' : 'Virial Field',
                      'SURFELFB' : 'Electron Localization'}
            if title == 'default':
                if np.all(overlay==None): title = 'Chemical Graph'
                else: title = '{} + Chemical Graph'.format(titles[overlay.type.upper()])
            fig.axes[iax].set_title(title)

        self._set_unit(uold)
        if np.all(overlay!=None): overlay._set_unit(uold)
        return fig

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

        lprops = ['base', 'cpoint'] # length units
        for l in lprops:
            newattr = getattr(self, l) * cst
            setattr(self, l, newattr)
        # special: trajectory
        self.trajectory = [[i[0], i[1]*cst] for i in self.trajectory]
        return self


