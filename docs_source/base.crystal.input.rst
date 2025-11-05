base.crystal.input module
=========================

CRYSTAL d12 file
----------------

.. _ref-base-crysd12:

The ``Crystal_inputBASE`` object is strictly structured by 'blocks', which, in general, is defined as keywords that are closed by 'END'. It is inherited from the :ref:`BlockBASE <ref-base-inputbase>` object and is inherited by the :ref:`Crystal_input <ref-io-crystal>` object. All the blocks are organized in layers and each corresponds to a list of keywords that can be called and set. The current structure of ``Crystal_inputBASE`` is listed below:

Layer 1: ``geom``, ``basisset``, ``scf``  

Layer 2: ``optgeom``, ``freqcalc``, ``hf3c``, ``hfsol3c``, ``dft``, ``dftd3``, ``gcp``, ``geom``, ``base``, ``geba``  

Layer 3: ``preoptgeom``  

For the usages of :ref:`BlockBASE <ref-base-inputbase>` and :ref:`Crystal_input <ref-io-crystal>` objects, please refer to the corresponding documentations.


Examples
~~~~~~~~

Note that methods listed below are for :ref:`Crystal_input <ref-io-crystal>` objects, which is typically consistent with ``Crystal_inputBASE``, except file read and write functions.

To set force convergence threshold of a optimization run:

.. code-block:: python

    >>> obj = Crystal_input()
    >>> obj.geom.optgeom.toldeg(0.0001)

By calling the 'block-like' attribute, a sub-block object will be automatically generated if no such object is saved in the upper block object. It also be initialized and deleted in a similar way as keyword commands.

.. code-block:: python

    >>> obj = Crystal_input()
    >>> obj.geom.optgeom() # Initialize OPTGEOM
    >>> obj.geom.optgeom(None) # Remove OPTGEOM

To set the values of keywords, their names are called as methods with a really rare exception (1 so far):

.. code-block:: python

    >>> obj.geom.freqcalc.preoptgeom() # Initialize PREOPTGEOM
    >>> obj.geom.freqcalc.preoptgeom.toldeg(0.0003)
    >>> obj.scf.toldee(9) # Set SCF TOLDEE = 9
    >>> obj.scf.toldee(None) # Clean the TOLDEE keyword and value
    >>> obj.scf.ppan() # Print PPAN keyword, without value

The only exception is given below. However, when doing text analysis and printing formatted string to files, the correct keyword is recognized and printed.

.. code-block:: python

    >>> obj.geom.molecule2() # MOLECULE keyword to extract molecule from lattice. Renamed to address the conflict with modelling keyword MOLECULE

Though one can set CRYSTAL input object by manually setting up all the attributes, it is also possible to read a template d12 file and do modifications.

.. code-block:: python

    >>> obj = Crystal_input('opt.d12')
    >>> obj.geom.optgeom(None) # Remove OPTGEOM block
    >>> obj.to_file('scf.d12') # Print it into file

It is also possible to set individual blocks by a string. This is achieved by simply assigning the string variable as input when calling the corresponding method.

.. code-block:: python

    >>> obj.scf.dft('SPIN\nEXCHANGE\nPBE\nCORRELAT\nP86\n')

For basis set, it is not a typical ``BlockBASE`` object (though it inherits ``BlockBASE``). When 'BASISSET' keyword is used, it is called in the same way as other blocks. When explicit definitions of basis set are used, it can be defined via formatted string, file, `Basis Set Exchange (BSE) <https://molssi-bse.github.io/basis_set_exchange/index.html>`_ and :ref:`BasisSetBASE <ref-base-basisset>` object. The ending line '99 0' is required.

.. code-block:: python

    >>> obj.basisset.basisset('def2-SVP')
    >>> obj.basisset.from_file('mybasis.txt')
    >>> obj.basisset.from_bse('6-311G*', [6, 1, 8]) # conventional atomic numbers are supported.

Wrapper methods (``bs_user()`` and ``bs_keyword()``) are added in :ref:`Crystal_input <ref-io-crystal>`. For details please checkt the manual :ref:`there <ref-io-crystal>`.

.. code-block:: python

    >>> obj.bs_keyword('def2-SVP')
    >>> obj.bs_user('mybasis.txt')
    >>> obj.bs_user('6-311G*', [6, 1, 8]) # bs_user accepts file, string and BSE variables

To examine the data in a block object, including the :ref:`Crystal_input <ref-io-crystal>` obj itself, call the ``data`` attribute.

.. code-block:: python

    >>> print(obj.data)


CRYSTAL d3 file
---------------

.. _ref-base-propd3:

The ``Properties_inputBASE`` object is strictly structured by 'blocks', which, in general, is defined as keywords that are closed by 'END'. It is inherited from the :ref:`BlockBASE <ref-base-inputbase>` object and is inherited by the :ref:`Properties_input <ref-io-crystal>` object. All the blocks are organized in layers and each corresponds to a list of keywords that can be called and set. The current structure of ``Properties_inputBASE`` is listed below:

Layer 1: Optional, repeated block (same calculation, another time) ``append1`` to ``append5``  

Layer 2: Data grid and DFT/MP2 correlation energy ``ECHG``, ``POTM``, ``CLAS``, ``EDFT/ENECOR``, ``ADFT/ACOR``

For the usages of :ref:`BlockBASE <ref-base-inputbase>` and :ref:`Properties_input <ref-io-crystal>` objects, please refer to the corresponding documentations.


Examples
~~~~~~~~

Note that methods listed below are for :ref:`Properties_input <ref-io-crystal>` objects, which is typically consistent with ``Properties_inputBASE``, except file read and write functions.

To set a band structure and a projected doss calculation:

.. code-block::

    >>> obj = Properties_input()
    >>> obj.band('Band calc title', 3, 6, 188, 203, 224, 1, 0,
                 [[[0, 0, 0], [3, 0, 0]], # A 3*2*3 list, for 3 line segments, 2 ending points of each segment, xyz for each point
                  [[3, 0, 0], [2, 2, 0]],
                  [[2, 2, 0], [0, 0, 0]]])
    >>> obj.newk(8, 16, 1, 0)
    >>> obj.doss(1, 600, 203, 224, 1, 12, 0, [[-1, 57], [-1, 64]]) # Project to atom 54 and atom 64

By calling the 'block-like' attribute, a sub-block object will be automatically generated if no such object is saved in the upper block object. It also be initialized and deleted in a similar way as keyword commands.

.. code-block::

    >>> obj = Properties_input()
    >>> obj.echg() # Initialize ECHG
    >>> obj.echg(None) # Remove ECHG
    >>> obj.echg(0, 95) # Charge density map, Npoint of MAPNET is 95 (default value 100)

To set the values of keywords, their names are called as methods:

.. code-block::

    >>> obj.echg.coordina([-2.498, 0., 1.696], [-2.498, 0., -1.696], [-1.249, -2.164, -1.696])
    >>> obj.echg.rectangu() # Print RECTANGU keyword, without value
    >>> obj.echg.margins(3, 3, 3, 3)

Though one can set CRYSTAL input object by manually setting up all the attributes, it is also possible to read a template d3 file and do modifications.

.. code-block::

    >>> obj = Properties_input('charge2d.d3')
    >>> obj.echg(None) # Remove ECHG block
    >>> obj.ech3(100) # Define ECH3 block
    >>> obj.ech3.range(-10, 10) # Range of Non-periodic direction
    >>> obj.to_file('charge3d.d3') # Print it into file

It is also possible to set individual blocks by a string. This is achieved by simply assigning the string variable as input when calling the corresponding method.

.. code-block::

    >>> obj.ech3('ECH3\n100\nRANGE\n-10\n10\n')

As stressed in the doc of :ref:`base.inputbase <ref-base-inputbase>`, repeated keywords not protected by sub-blocks are not permitted. That leads to problems when, for example, plotting the charge difference map to analyze bonds, where 'ECHG' is repeated twice in the main block. To address this, the following sub-block is called when the second 'ECHG' is used:

.. code-block::

    >>> obj = Properties_input()
    >>> obj.echg(0) # Set the first ECHG. MAPNET density = 100, default value is used.
    >>> obj.echg.coordina([-2.498, 0., 1.696], [-2.498, 0., -1.696], [-1.249, -2.164, -1.696])
    >>> obj.echg.rectangu()
    >>> obj.echg.margins(3, 3, 3, 3)
    >>> obj.append1() # Initialize the first appended calculation
    >>> obj.append1.pato(1, 0)
    >>> obj.append1.echg(0) # Set the second ECHG. MAPNET density = 100, default value is used.
    >>> obj.append1.echg.coordina([-2.498, 0., 1.696], [-2.498, 0., -1.696], [-1.249, -2.164, -1.696])
    >>> obj.append1.echg.rectangu()
    >>> obj.append1.echg.margins(3, 3, 3, 3)

5 appended calculations (``append1`` to ``append5``) at most can be added and the ``Properties_input`` object is the initial calculation, so the same keyword, which is not protected by subblocks, can repeat 6 times at most. Error is reported if it appears more than 6 times.


To examine the data in a block object, including the :ref:`Properties_input <ref-io-crystal>` obj itself, call the ``data`` attribute.

.. code-block::

    >>> print(obj.data)


.. automodule:: CRYSTALpytools.base.crystal.input
   :members:
   :private-members:
   :undoc-members:
   :show-inheritance:
