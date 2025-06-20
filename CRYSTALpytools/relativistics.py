#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The module for '2c-SCF', i.e., 2-component SCF, and relativistics.
"""
import numpy as np
from warnings import warn

from CRYSTALpytools import units
from CRYSTALpytools.electronics import ChargeDensity as ChgDens

class ChargeDensity(ChgDens):
    """
    Same as ``electronics.ChargeDensity``, the spin density. But its dimension
    is kept to be commensurate with other keywords of the 'PROPS2COMP' block.

    .. note::

        Its ``type`` attribute is 'ECHG' rather than 'DENSITY'

    """
    @classmethod
    def from_file(cls, file, output, source='crystal'):
        """
        Generate a ``ChargeDensity`` object from formatted output.

        Args:
            file (str): File name of fort.25 or CUBE files.
            output (str): Screen output of 'properties' calculation.
            source (str): Currently useless. Must be 'crystal'.
        Returns:
            cls (ChargeDensity)
        """
        if source.lower() == 'crystal':
            from CRYSTALpytools.crystal_io import Properties_output
            return Properties_output(output).read_relativistics(file, type='DENSITY')


class VectorField():
    """
    The basic vector field object, containing a nY\*nX\*3 (nZ\*nY\*nX\*3) data
    array for 2D (3D) fields. Call the property-specific child classes below in
    use.
    """
    @classmethod
    def from_file(cls, file, output, source, prop):
        """ Generate corresponding object from formatted output.

        Args:
            file (str): File name of fort.25 or CUBE files.
            output (str): Screen output of 'properties' calculation.
            source (str): Currently useless. Must be 'crystal'.
            prop (str): 'MAGNETIZ', 'ORBCURDENS' or 'SPICURDENS'.
        Returns:
            obj: Depending on ``prop``.
        """
        if prop.upper() not in ['MAGNETIZ', 'ORBCURDENS', 'SPICURDENS']:
            raise Exception(f"Unknown property: '{prop}'. Use 'MAGNETIZ', 'ORBCURDENS' or 'SPICURDENS'.")

        if source.lower() == 'crystal':
            from CRYSTALpytools.crystal_io import Properties_output
            return Properties_output(output).read_relativistics(file, type=prop.upper())

    def plot_2D(self, unit, levels, quiverplot, quiverscale, colorplot, colormap,
                cbar_label, a_range, b_range, rectangle, edgeplot,
                x_ticks, y_ticks, figsize, **kwargs):
        """ Plot 2D vector field. Only accept integer ``ax_index`` inputs, i.e.,
        only plots on a single axis.

        3 styles are available:

        1. ``quiverplot=True`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors. The black arrows indicates both
            the directions and norms of in-plane prjections.  
        2. ``quiverplot=True`` and ``colorplot=False``: The arrows are colored
            to indicate the directions and norms of in-plane prjections.  
        3. ``quiverplot=False`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors, similar to the 2D scalar map.

        Please refer to the child classes for input arguments.

        Returns:
            fig (Figure): Matplotlib Figure class.
        """
        from CRYSTALpytools.base.plotbase import plot_2Dscalar, plot_2Dvector
        import matplotlib.pyplot as plt

        # dimen
        if self.dimension != 2: raise Exception('Not a 2D vector field object.')

        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower: self._set_unit(unit)

        try:
            # levels
            ## get norm
            vnorm = np.linalg.norm(self.data, axis=2)
            levels = np.array(levels, ndmin=1, dtype=float)
            if levels.shape[0] == 1:
                levels = np.linspace(np.min(vnorm), np.max(vnorm), int(levels[0]))
            if levels.ndim > 1: raise ValueError('Levels must be a 1D array.')

            # plot
            if 'fig' not in kwargs.keys():
                fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
            else:
                if 'ax_index' not in kwargs.keys():
                    raise ValueError("Indices of axes must be set when 'fig' is not None.")
                fig = kwargs['fig']
                ax = fig.axes[int(kwargs['ax_index'])]

            # plot colormap
            if colorplot == True:
                fig = plot_2Dscalar(fig, ax, vnorm, self.base, levels, None, None,
                                    colormap, cbar_label, a_range, b_range,
                                    rectangle, edgeplot, x_ticks, y_ticks)
            # plot quiver
            if quiverplot == True:
                if colorplot == True:
                    cxlim = ax.get_xlim()
                    cylim = ax.get_ylim() # in case of white edges
                    fig = plot_2Dvector(fig, ax, self.data, self.base, quiverscale,
                                        'k', levels, colormap, cbar_label, a_range,
                                        b_range, rectangle, False, x_ticks, y_ticks)
                    ax.set_xlim(cxlim)
                    ax.set_ylim(cylim)
                else:
                    fig = plot_2Dvector(fig, ax, self.data, self.base, quiverscale,
                                        'colored', levels, colormap, cbar_label,
                                        a_range, b_range, rectangle, edgeplot, x_ticks, y_ticks)

        except Exception as e:
            self._set_unit(uold)
            raise e

        self._set_unit(uold)
        return fig


    def plot_3D(self, unit, vec_plot=True, scal_plot=False, **kwargs):
        """ Visualize **2D or 3D** vector fields with atomic structures using
        `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        Explanations of input/output arguments are detailed in specific classes.
        """
        from CRYSTALpytools.base.plotbase import plot_GeomScalar, plot_GeomVector

        # unit
        uold = self.unit
        if self.unit.lower() != unit.lower: self._set_unit(unit)

        # map input keywords
        struc_key = ['atom_color', 'bond_color', 'atom_bond_ratio', 'cell_display',
                     'cell_color', 'cell_linewidth', 'display_range', 'special_bonds']

        ## except struc, base, data of base.plotbase.plot_GeomScalar()
        scalar_key = ['isovalue', 'volume_3d', 'contour_2d', 'interp', 'interp_size',
                      'opacity', 'transparent', 'isoline_color', 'isoline_width',
                      'scal_grid_display_range', 'scal_colormap',  'scal_vmax',
                      'scal_vmin', 'scal_title', 'scal_orientation', 'scal_nb_labels',
                      'scal_label_fmt']
        scalar_basekey = dict(
            isovalue='isovalue', volume_3d='volume_3d', contour_2d='contour_2d',
            interp='interp', interp_size='interp_size', opacity='opacity',
            transparent='transparent', isoline_color='color', isoline_width='line_width',
            scal_grid_display_range='grid_display_range', scal_colormap='colormap',
            scal_vmax='vmax', scal_vmin='vmin', scal_title='title',
            scal_orientation='orientation', scal_nb_labels='nb_labels',
            scal_label_fmt='label_fmt')

        ## except struc, base, data of base.plotbase.plot_GeomVector()
        vector_key = ['vec_grid_display_range', 'vec_colormap', 'vec_linewidth',
                      'vec_linescale', 'vec_vmax', 'vec_vmin', 'vec_title',
                      'vec_orientation', 'vec_nb_labels', 'vec_label_fmt']
        vector_basekey = dict(
            vec_grid_display_range='grid_display_range', vec_colormap='colormap',
            vec_linewidth='line_width', vec_linescale='scale_factor', vec_vmax='vmax',
            vec_vmin='vmin', vec_title='title', vec_orientation='orientation',
            vec_nb_labels='nb_labels', vec_label_fmt='label_fmt')

        try:
            if scal_plot == True:
                # defaults
                plot_in = dict(isovalue=None, volume_3d=False, contour_2d=False,
                               interp='no interp', interp_size=1,
                               grid_display_range=[[0,1], [0,1], [0,1]],
                               colormap='jet')
                for k, v in kwargs.items():
                    if k in struc_key:
                        plot_in[k] = v
                    elif k in scalar_key:
                        plot_in[scalar_basekey[k]] = v
                # plot norm
                fig = plot_GeomScalar(struc=self.structure, base=self.base,
                                      data=np.linalg.norm(self.data, axis=-1), **plot_in)

            if vec_plot == True:
                if scal_plot == True:
                    # defaults
                    plot_in = dict(fig=fig, grid_display_range=[[0,1], [0,1], [0,1]],
                                   colormap=(0., 0., 0.))
                    for k, v in kwargs.items():
                        if k in vector_key:
                            plot_in[vector_basekey[k]] = v

                    # plot vec
                    fig = plot_GeomVector(struc=None, base=self.base,
                                          data=self.data, **plot_in)
                else:
                    # defaults
                    plot_in = dict(grid_display_range=[[0,1], [0,1], [0,1]], colormap='jet')
                    for k, v in kwargs.items():
                        if k in struc_key:
                            plot_in[k] = v
                        elif k in vector_key:
                            plot_in[vector_basekey[k]] = v

                    # plot vec
                    fig = plot_GeomVector(struc=self.structure, base=self.base,
                                          data=self.data, **plot_in)

        except Exception as e:
            self._set_unit(uold)
            raise e

        self._set_unit(uold)
        return fig


    def _set_unit(self, unit):
        """"Of no practical use."""
        pass


class Magnetization(VectorField):
    """
    The class for magnetization. Unit: 'SI' (length: :math:`\\AA`,
    magnetization: A/m).

    Args:
        data (array): nY\*nX\*3 (nZ\*nY\*nX\*3) array of magnetization vectors
            in 2D (3D) plane.
        base (array): 3\*3 Cartesian coordinates of the 3 points defining
            vectors BA and BC (2D) or 3 base vectors (3D)
        dimen (int): Dimensionality of the plot.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'SI' (case insensitive).
    """
    def __init__(self, data, base, dimen, struc, unit='SI'):
        self.data = np.array(data, dtype=float)
        self.base = np.array(base, dtype=float)
        self.dimension = int(dimen)
        self.structure = struc
        self.unit = unit
        self.type = 'MAGNETIZ'

    @classmethod
    def from_file(cls, file, output, source='crystal'):
        """
        Generate a ``Magentization`` object from CRYSTAL formatted output unit
        and standard screen output (mandatory).

        Args:
            file (str): File name of fort.25 or CUBE (in development) files.
            output (str): Screen output of 'properties' calculation.
            source (str): Currently useless. Must be 'crystal'.
        Returns:
            cls (Magnetization)
        """
        return super().from_file(file=file, output=output, source=source, prop='MAGNETIZ')

    def plot_2D(self, unit='SI', levels=100, quiverplot=True, quiverscale=1.0,
                colorplot=True, colormap='jet', cbar_label='default',
                a_range=[0.,1.], b_range=[0.,1.], rectangle=False, edgeplot=False,
                x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], **kwargs):
        """
        Plot 2D magnetization field.

        3 styles are available:

        1. ``quiverplot=True`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors. The black arrows indicates both
            the directions and norms of in-plane prjections.  
        2. ``quiverplot=True`` and ``colorplot=False``: The arrows are colored
            to indicate the directions and norms of in-plane prjections.  
        3. ``quiverplot=False`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors, similar to the 2D scalar map.

        Args:
            unit (str): Plot unit. 'SI' for :math:`\\AA` and A/m. 'a.u.' for
                Bohr and a.u. magnetization.
            levels (int|array): Set levels of colored contour/quiver plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D.
            quiverplot (bool): Plot 2D field of arrows.
            quiverscale (float): Tune the length of arrows. Useful only if
                ``quiverplot=True``.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if ``colorplot=True``.
            cbar_label (str|None): Label of colorbar. 'default' for unit.
                'None' for no label. Useful only if ``colorplot=True``.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            rectangle (bool): If :math:`a, b` are non-orthogonal, plot a
                rectangle region and reset :math:`b`. If used together with
                ``b_range``, that refers to the old :math:`b` (i.e., expansion first).
            edgeplot (bool): Whether to add cell edges represented by the
                original base vectors (not inflenced by a/b range or rectangle
                options).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for proeprty plotted.
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            \*\*kwargs : Other arguments passed to ``axes.quiver()`` function
                to set arrow styles.

        Returns:
            fig (Figure): Matplotlib Figure class.
        """
        # cbar_label
        if isinstance(cbar_label, str):
            if cbar_label.lower() == 'default':
                if unit.lower() == 'si':
                    cbar_label = 'Unit: A/m'
                else:
                    cbar_label = 'Unit: a.u.'
        else:
            cbar_label = None

        # force ax_index to be 1D array
        if 'ax_index' in kwargs.keys():
            ax_index = np.array(kwargs['ax_index'], ndmin=1, dtype=int)
            if len(ax_index) != 0:
                raise Exception("Magnetization does not support multiple subplots.")
        else:
            ax_index = [0]
        kwargs['ax_index'] = ax_index[0]

        fig = super().plot_2D(unit, levels, quiverplot, quiverscale, colorplot,
                              colormap, cbar_label, a_range, b_range, rectangle,
                              edgeplot, x_ticks, y_ticks, figsize, **kwargs)

        # title and axis labels
        for iax in ax_index:
            if unit.lower() == 'si':
                fig.axes[iax].set_xlabel(r'$\AA$')
                fig.axes[iax].set_ylabel(r'$\AA$')
            else:
                fig.axes[iax].set_xlabel('Bohr')
                fig.axes[iax].set_ylabel('Bohr')
            if isinstance(title, str):
                if title.lower() == 'default':
                    title = 'Magnetization'
                fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self, unit='SI', vec_plot=True, scal_plot=False, **kwargs):
        """ Visualize **2D or 3D** magnetization fields with atomic structures using
        `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        Args:
            unit (str): 'SI' or 'a.u.'.
            vec_plot (bool): Plot quivers.
            scal_plot (bool): Plot vector norms as 2D/3D scalar fields.
            \*\*kwargs: Optional keywords, listed below.
            vec_linewidth (float): Width of arrows.
            vec_linescale (float): Length scale of arrows.
            vec_grid_display_range (array): 3\*2 array defining the displayed
                region of the vector field. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            vec_colormap (turple|str): Colormap of vector field. Or a 1\*3
                RGB turple from 0 to 1 to define colors of vectors.
            vec_vmax (float): Maximum value of vector field's colormap.
            vec_vmin (float): Minimum value of vector field's colormap.
            vec_title (str): Vector field's colorbar title.
            vec_orientation (str): Orientation of vector field's colorbar, 'horizontal' or 'vertical'.
            vec_nb_labels (int): The number of labels to display on the vector field's colorbar.
            vec_label_fmt (str): The string formater for the labels of vector field's colorbar, e.g., '%.1f'.
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
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            isoline_color (turple): Color of contour lines. *'contour_2d=True' only*.
            isoline_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            scal_grid_display_range (array): 3\*2 array defining the displayed
                region of the scalar field. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            scal_colormap (turple|str): Colormap of scalar field. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            scal_vmax (float): Maximum value of scalar field's colormap.
            scal_vmin (float): Minimum value of scalar field's colormap.
            scal_title (str): Scalar field's colorbar title.
            scal_orientation (str): Orientation of scalar field's colorbar, 'horizontal' or 'vertical'.
            scal_nb_labels (int): The number of labels to display on the scalar field's colorbar.
            scal_label_fmt (str): The string formater for the labels of scalar field's colorbar, e.g., '%.1f'.
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
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and return None.
                Otherwise return the scene object.
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
        fig = super().plot_3D(unit, vec_plot=vec_plot, scal_plot=scal_plot, **kwargs)

        # Final setups
        if 'show_the_scene' in kwargs.keys():
            show_the_scene = kwargs['show_the_scene']
        else:
            show_the_scene = True

        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in kwargs.items():
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
        Set units of data of ``Magnetization`` object.

        Args:
            unit (str): 'SI', length: :math:`\\AA`, magnetization: A/m.
                'a.u.', all in a.u..
        """
        from CRYSTALpytools.units import angstrom_to_au, au_to_angstrom, ampere_to_au, au_to_ampere

        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'si':
            self.unit = 'SI'
            mcst = au_to_ampere(1.) * angstrom_to_au(1.)*1e10
            lcst = au_to_angstrom(1.)
        elif unit.lower() == 'a.u.':
            mcst = ampere_to_au(1.) * au_to_angstrom(1.)*1e10
            lcst = angstrom_to_au(1.)
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        mprops = ['data'] # magnetization units
        for l in lprops:
            if hasattr(self, l):
                newattr = getattr(self, l) * lcst
                setattr(self, l, newattr)
        for m in mprops:
            if hasattr(self, m):
                newattr = getattr(self, m) * mcst
                setattr(self, m, newattr)
        return self


class OrbitalCurrentDensity(VectorField):
    """
    The class for orbital current density. Unit: 'SI' (length: :math:`\\AA`,
    orbital current density: A/m :math:`^{2}`).

    Args:
        data (array): nY\*nX\*3 (nZ\*nY\*nX\*3) array of orbnital current vectors
            in 2D (3D) plane.
        base (array): 3\*3 Cartesian coordinates of the 3 points defining
            vectors BA and BC (2D) or 3 base vectors (3D)
        dimen (int): Dimensionality of the plot.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'SI' (case insensitive).
    """
    def __init__(self, data, base, dimen, struc, unit='SI'):
        self.data = np.array(data, dtype=float)
        self.base = np.array(base, dtype=float)
        self.dimension = int(dimen)
        self.structure = struc
        self.unit = unit
        self.type = 'ORBCURDENS'

    @classmethod
    def from_file(cls, file, output, source='crystal'):
        """
        Generate a ``OrbitalCurrentDensity`` object from CRYSTAL formatted
        output unit and standard screen output (mandatory).

        Args:
            file (str): File name of fort.25 or CUBE (in development) files.
            output (str): Screen output of 'properties' calculation.
            source (str): Currently useless. Must be 'crystal'.
        Returns:
            cls (OrbitalCurrentDensity)
        """
        return super().from_file(file=file, output=output, source=source, prop='ORBCURDENS')


    def plot_2D(self, unit='SI', levels=100, quiverplot=True, quiverscale=1.0,
                colorplot=True, colormap='jet', cbar_label='default',
                a_range=[0., 1], b_range=[0., 1], rectangle=False, edgeplot=False,
                x_ticks=5, y_ticks=5, title='default', figsize=[6.4, 4.8], **kwargs):
        """
        Plot 2D orbital current density field.

        3 styles are available:

        1. ``quiverplot=True`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors. The black arrows indicates both
            the directions and norms of in-plane prjections.  
        2. ``quiverplot=True`` and ``colorplot=False``: The arrows are colored
            to indicate the directions and norms of in-plane prjections.  
        3. ``quiverplot=False`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors, similar to the 2D scalar map.

        Args:
            unit (str): Plot unit. 'SI' for :math:`\\AA` and A/m :math:`^{2}`.
                'a.u.' for Bohr and a.u. current density.
            levels (int|array): Set levels of colored contour/quiver plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D.
            quiverplot (bool): Plot 2D field of arrows.
            quiverscale (float): Tune the length of arrows. Useful only if
                ``quiverplot=True``.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if ``colorplot=True``.
            cbar_label (str|None): Label of colorbar. 'default' for unit.
                'None' for no label. Useful only if ``colorplot=True``.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (y, or BA) in
                fractional coordinate.
            rectangle (bool): If :math:`a, b` are non-orthogonal, plot a
                rectangle region and reset :math:`b`. If used together with
                ``b_range``, that refers to the old :math:`b` (i.e., expansion first).
            edgeplot (bool): Whether to add cell edges represented by the
                original base vectors (not inflenced by a/b range or rectangle
                options).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for proeprty plotted.
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            \*\*kwargs : Other arguments passed to ``axes.quiver()`` function
                to set arrow styles.

        Returns:
            fig (Figure): Matplotlib Figure class.
        """
        # cbar_label
        if isinstance(cbar_label, str):
            if cbar_label.lower() == 'default':
                if unit.lower() == 'si':
                    cbar_label = r'Unit: A/m$^{2}$'
                else:
                    cbar_label = 'Unit: a.u.'
        else:
            cbar_label = None

        # force ax_index to be 1D array
        if 'ax_index' in kwargs.keys():
            ax_index = np.array(kwargs['ax_index'], ndmin=1, dtype=int)
            if len(ax_index) != 1:
                raise Exception("Oribtal current density does not support multiple subplots.")
        else:
            ax_index = [0]
        kwargs['ax_index'] = ax_index[0]

        fig = super().plot_2D(unit, levels, quiverplot, quiverscale, colorplot,
                              colormap, cbar_label, a_range, b_range, rectangle,
                              edgeplot, x_ticks, y_ticks, figsize, **kwargs)

        # title and axis labels
        for iax in ax_index:
            if unit.lower() == 'si':
                fig.axes[iax].set_xlabel(r'$\AA$')
                fig.axes[iax].set_ylabel(r'$\AA$')
            else:
                fig.axes[iax].set_xlabel('Bohr')
                fig.axes[iax].set_ylabel('Bohr')
            if isinstance(title, str):
                if title.lower() == 'default':
                    title = 'Orbital Current Density'
                fig.axes[iax].set_title(title)
        return fig

    def plot_3D(self, unit='SI', vec_plot=True, scal_plot=False, **kwargs):
        """ Visualize **2D or 3D** orbital current density with atomic structures
        using `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        Args:
            unit (str): 'SI' or 'a.u.'.
            vec_plot (bool): Plot quivers.
            scal_plot (bool): Plot vector norms as 2D/3D scalar fields.
            \*\*kwargs: Optional keywords, listed below.
            vec_linewidth (float): Width of arrows.
            vec_linescale (float): Length scale of arrows.
            vec_grid_display_range (array): 3\*2 array defining the displayed
                region of the vector field. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            vec_colormap (turple|str): Colormap of vector field. Or a 1\*3
                RGB turple from 0 to 1 to define colors of vectors.
            vec_vmax (float): Maximum value of vector field's colormap.
            vec_vmin (float): Minimum value of vector field's colormap.
            vec_title (str): Vector field's colorbar title.
            vec_orientation (str): Orientation of vector field's colorbar, 'horizontal' or 'vertical'.
            vec_nb_labels (int): The number of labels to display on the vector field's colorbar.
            vec_label_fmt (str): The string formater for the labels of vector field's colorbar, e.g., '%.1f'.
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
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            isoline_color (turple): Color of contour lines. *'contour_2d=True' only*.
            isoline_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            scal_grid_display_range (array): 3\*2 array defining the displayed
                region of the scalar field. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            scal_colormap (turple|str): Colormap of scalar field. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            scal_vmax (float): Maximum value of scalar field's colormap.
            scal_vmin (float): Minimum value of scalar field's colormap.
            scal_title (str): Scalar field's colorbar title.
            scal_orientation (str): Orientation of scalar field's colorbar, 'horizontal' or 'vertical'.
            scal_nb_labels (int): The number of labels to display on the scalar field's colorbar.
            scal_label_fmt (str): The string formater for the labels of scalar field's colorbar, e.g., '%.1f'.
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
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and return None.
                Otherwise return the scene object.
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
        fig = super().plot_3D(unit, vec_plot=vec_plot, scal_plot=scal_plot, **kwargs)

        # Final setups
        if 'show_the_scene' in kwargs.keys():
            show_the_scene = kwargs['show_the_scene']
        else:
            show_the_scene = True

        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in kwargs.items():
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
        Set units of data of ``OrbitalCurrentDensity`` object.

        Args:
            unit (str): 'SI', length: :math:`\\AA`, orbital current density:
                A/m :math:`^{2}`. 'a.u.', all in a.u..
        """
        from CRYSTALpytools.units import angstrom_to_au, au_to_angstrom, ampere_to_au, au_to_ampere

        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'si':
            self.unit = 'SI'
            mcst = au_to_ampere(1.) * (angstrom_to_au(1.)*1e10)**2
            lcst = au_to_angstrom(1.)
        elif unit.lower() == 'a.u.':
            mcst = ampere_to_au(1.) * (au_to_angstrom(1.)*1e10)**2
            lcst = angstrom_to_au(1.)
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        mprops = ['data'] # current density units
        for l in lprops:
            if hasattr(self, l):
                newattr = getattr(self, l) * lcst
                setattr(self, l, newattr)
        for m in mprops:
            if hasattr(self, m):
                newattr = getattr(self, m) * mcst
                setattr(self, m, newattr)
        return self


class SpinCurrentDensity(VectorField):
    """
    The class for spin current density. Unit: 'SI' (length: :math:`\\AA`,
    spin current density: A/m :math:`^{2}`).

    Args:
        data_x (array): nY\*nX\*3 (nZ\*nY\*nX\*3) array of spin current vectors
            :math:`J^{x}` in 2D (3D) plane.
        data_y (array): nY\*nX\*3 (nZ\*nY\*nX\*3) array of spin current vectors
            :math:`J^{y}` in 2D (3D) plane.
        data_z (array): nY\*nX\*3 (nZ\*nY\*nX\*3) array of spin current vectors
            :math:`J^{z}` in 2D (3D) plane.
        base (array): 3\*3 Cartesian coordinates of the 3 points defining
            vectors BA and BC (2D) or 3 base vectors (3D)
        dimen (int): Dimensionality of the plot.
        struc (CStructure): Extended Pymatgen Structure object.
        unit (str): In principle, should always be 'SI' (case insensitive).
    """
    def __init__(self, data_x, data_y, data_z, base, dimen, struc, unit='SI'):
        self.data_x = np.array(data_x, dtype=float)
        self.data_y = np.array(data_y, dtype=float)
        self.data_z = np.array(data_z, dtype=float)
        self.base = np.array(base, dtype=float)
        self.dimension = int(dimen)
        self.structure = struc
        self.unit = unit
        self.type = 'SPICURDENS'

    @classmethod
    def from_file(cls, file, output, source='crystal'):
        """
        Generate a ``SpinCurrentDensity`` object from CRYSTAL formatted output
        unit and standard screen output (mandatory).

        Args:
            file (str): File name of fort.25 or CUBE (in development) files.
            output (str): Screen output of 'properties' calculation.
            source (str): Currently useless. Must be 'crystal'.
        Returns:
            cls (SpinCurrentDensity)
        """
        return super().from_file(file=file, output=output, source=source, prop='SPICURDENS')

    def plot_2D(self, unit='SI', direction=['x','y','z'], levels=100,
                quiverplot=True, quiverscale=1.0, colorplot=True, colormap='jet',
                cbar_label='default', a_range=[0.,1.], b_range=[0.,1.],
                rectangle=False, edgeplot=False, x_ticks=5, y_ticks=5,
                title='default', figsize=[6.4, 4.8], **kwargs):
        """
        Plot 2D orbital current density field.

        3 styles are available:

        1. ``quiverplot=True`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors. The black arrows indicates both
            the directions and norms of in-plane prjections.  
        2. ``quiverplot=True`` and ``colorplot=False``: The arrows are colored
            to indicate the directions and norms of in-plane prjections.  
        3. ``quiverplot=False`` and ``colorplot=True``: The color-filled contour
            illustrates the norm of vectors, similar to the 2D scalar map.

        Args:
            unit (str): Plot unit. 'SI' for :math:`\\AA` and A/m :math:`^{2}`.
                'a.u.' for Bohr and a.u. current density.
            direction (str|list): Direction of spin-current to plot, in 'x', 'y' or 'z'.
            levels (int|array): Set levels of colored contour/quiver plot. A
                number for linear scaled plot colors or an array for
                user-defined levels. 1D.
            quiverplot (bool): Plot 2D field of arrows.
            quiverscale (float): Tune the length of arrows. Useful only if
                ``quiverplot=True``.
            colorplot (bool): Plot color-filled contour plots.
            colormap (str): Matplotlib colormap option. Useful only if ``colorplot=True``.
            cbar_label (str|None): Label of colorbar. 'default' for unit.
                'None' for no label. Useful only if ``colorplot=True``.
            a_range (list): 1\*2 range of :math:`a` axis (x, or BC) in
                fractional coordinate.
            b_range (list): 1\*2 range of :math:`b` axis (x, or AB) in
                fractional coordinate.
            rectangle (bool): If :math:`a, b` are non-orthogonal, plot a
                rectangle region and reset :math:`b`. If used together with
                ``b_range``, that refers to the old :math:`b` (i.e., expansion first).
            edgeplot (bool): Whether to add cell edges represented by the
                original base vectors (not inflenced by a/b range or rectangle
                options).
            x_ticks (int): Number of ticks on x axis.
            y_ticks (int): Number of ticks on y axis.
            title (str|None): Plot title. 'default' for proeprty plotted.
                'None' for no title.
            figsize (list): Matplotlib figure size. Note that axes aspects are
                fixed to be equal.
            \*\*kwargs : Other arguments passed to ``axes.quiver()`` function
                to set arrow styles.

        Returns:
            fig (Figure): Matplotlib Figure class.
        """
        import matplotlib.pyplot as plt

        # cbar_label
        if isinstance(cbar_label, str):
            if cbar_label.lower() == 'default':
                if unit.lower() == 'si':
                    cbar_label = r'Unit: A/m$^{2}$'
                else:
                    cbar_label = 'Unit: a.u.'
        else:
            cbar_label = None

        # direction
        props = {'x' : 'data_x', 'y' : 'data_y', 'z' : 'data_z'}
        direction = np.array(direction, ndmin=1)
        if len(direction) > 3:
            raise ValueError("At maximum a 1*3 list of string should be specified.")
        for id in range(len(direction)):
            direction[id] = direction[id].lower()
            if direction[id] not in ['x', 'y', 'z']:
                raise ValueError(f"Unknown direction entry: '{direction[id]}'.")

        # force ax_index to be 1D array
        if 'ax_index' in kwargs.keys():
            ax_index = np.array(kwargs['ax_index'], ndmin=1, dtype=int)
            if len(ax_index) != len(direction):
                raise Exception("Inconsistent lengthes of 'direction 'and 'ax_index'.")
        else:
            ax_index = [i for i in range(len(direction))]
        kwargs['ax_index'] = ax_index

        if 'fig' not in kwargs.keys():
            fig, _ = plt.subplots(1, len(direction), figsize=figsize,
                                  sharex=True, sharey=True, layout='tight')
            kwargs['fig'] = fig

        for d, iax in zip(direction, ax_index):
            kwargs['ax_index'] = iax
            setattr(self, 'data', getattr(self, props[d])) # add data_x/y/z as 'data' attr
            fig = super().plot_2D(unit, levels, quiverplot, quiverscale, colorplot,
                                  colormap, cbar_label, a_range, b_range, rectangle,
                                  edgeplot, x_ticks, y_ticks, figsize, **kwargs)
            delattr(self, 'data')

        # title and axis labels
        for d, iax in zip(direction, ax_index):
            if unit.lower() == 'si':
                fig.axes[iax].set_xlabel(r'$\AA$')
                fig.axes[iax].set_ylabel(r'$\AA$')
            else:
                fig.axes[iax].set_xlabel('Bohr')
                fig.axes[iax].set_ylabel('Bohr')
            if isinstance(title, str):
                if title.lower() == 'default':
                    ntitle = f'Spin Current Density J{d}'
                else:
                    ntitle = title
                fig.axes[iax].set_title(ntitle)
        return fig

    def plot_3D(self, unit='SI', direction='x', vec_plot=True, scal_plot=False, **kwargs):
        """ Visualize **2D or 3D** spin current density with atomic structures
        using `MayaVi <https://docs.enthought.com/mayavi/mayavi/>`_ (*not installed
        by default*).

        Args:
            unit (str): 'SI' or 'a.u.'.
            direction (str): Direction of spin-current to plot, in 'x', 'y' or 'z'.
            vec_plot (bool): Plot quivers.
            scal_plot (bool): Plot vector norms as 2D/3D scalar fields.
            \*\*kwargs: Optional keywords, listed below.
            vec_linewidth (float): Width of arrows.
            vec_linescale (float): Length scale of arrows.
            vec_grid_display_range (array): 3\*2 array defining the displayed
                region of the vector field. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            vec_colormap (turple|str): Colormap of vector field. Or a 1\*3
                RGB turple from 0 to 1 to define colors of vectors.
            vec_vmax (float): Maximum value of vector field's colormap.
            vec_vmin (float): Minimum value of vector field's colormap.
            vec_title (str): Vector field's colorbar title.
            vec_orientation (str): Orientation of vector field's colorbar, 'horizontal' or 'vertical'.
            vec_nb_labels (int): The number of labels to display on the vector field's colorbar.
            vec_label_fmt (str): The string formater for the labels of vector field's colorbar, e.g., '%.1f'.
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
            opacity (float): Opacity from 0 to 1. For ``volume_3d=True``, that
                defines the opacity of the maximum value. The opacity of the
                minimum is half of it.
            transparent (bool): Scalar-dependent opacity. *Not for volume_3d=True*.
            isoline_color (turple): Color of contour lines. *'contour_2d=True' only*.
            isoline_width (float): Width of 2D contour lines. *'contour_2d=True' only*.
            scal_grid_display_range (array): 3\*2 array defining the displayed
                region of the scalar field. Fractional coordinates a, b, c are
                used but only the periodic directions are applied.
            scal_colormap (turple|str): Colormap of scalar field. Or a 1\*3
                RGB turple from 0 to 1 to define colors. *Not for volume_3d=True*.
            scal_vmax (float): Maximum value of scalar field's colormap.
            scal_vmin (float): Minimum value of scalar field's colormap.
            scal_title (str): Scalar field's colorbar title.
            scal_orientation (str): Orientation of scalar field's colorbar, 'horizontal' or 'vertical'.
            scal_nb_labels (int): The number of labels to display on the scalar field's colorbar.
            scal_label_fmt (str): The string formater for the labels of scalar field's colorbar, e.g., '%.1f'.
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
            special_bonds (dict): See :ref:`geometry.CStructure.get_bonds() <ref-CStrucGetBonds>`.
            show_the_scene (bool): Display the scene by ``mlab.show()`` and return None.
                Otherwise return the scene object.
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

        props = {'x' : 'data_x', 'y' : 'data_y', 'z' : 'data_z'}
        direction = str(direction).lower()
        if direction not in ['x', 'y', 'z']:
            raise ValueError(f"Unknown direction entry: '{direction}'.")

        # Plot
        setattr(self, 'data', getattr(self, props[direction]))
        fig = super().plot_3D(unit, vec_plot=vec_plot, scal_plot=scal_plot, **kwargs)
        delattr(self, 'data')

        # Final setups
        if 'show_the_scene' in kwargs.keys():
            show_the_scene = kwargs['show_the_scene']
        else:
            show_the_scene = True

        keys = ['azimuth', 'elevation', 'distance', 'focalpoint', 'roll']
        keywords = dict(figure=fig, distance='auto')
        for k, v in kwargs.items():
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
        Set units of data of ``SpinCurrentDensity`` object.

        Args:
            unit (str): 'SI', length: :math:`\\AA`, spin current density:
                A/m :math:`^{2}`. 'a.u.', all in a.u..
        """
        from CRYSTALpytools.units import angstrom_to_au, au_to_angstrom, ampere_to_au, au_to_ampere

        if unit.lower() == self.unit.lower():
            return self

        if unit.lower() == 'si':
            self.unit = 'SI'
            mcst = au_to_ampere(1.) * (angstrom_to_au(1.)*1e10)**2
            lcst = au_to_angstrom(1.)
        elif unit.lower() == 'a.u.':
            mcst = ampere_to_au(1.) * (au_to_angstrom(1.)*1e10)**2
            lcst = angstrom_to_au(1.)
        else:
            raise ValueError('Unknown unit.')

        lprops = [] # length units. Note: Base should be commensurate with structure and not updated here.
        mprops = ['data_x', 'data_y', 'data_z', 'data'] # current density units
        for l in lprops:
            if hasattr(self, l):
                newattr = getattr(self, l) * lcst
                setattr(self, l, newattr)
        for m in mprops:
            if hasattr(self, m):
                newattr = getattr(self, m) * mcst
                setattr(self, m, newattr)
        return self

