#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It is a test script to load Jupyter Notebooks and run examples there. Outputs
are compared with outputs saved in notebooks.
"""
def run_notebook(filename, nbdir, exact_out=False):
    """
    Read Jupyter Notebooks and run example codes.

    Args:
        filename (str): Name of jupyter notebook file
        nbdir (str): 'examples/' directory
        exact_out (bool): Whether to match the exact outputs from cells.
            Suggested for file I/O and format convertion only.
    Returns:
        status (int): 0 for no error; 1 for exact matching of output error; 2
            for errors in running code block
    """
    import io, os, sys, json, warnings

    os.chdir(nbdir) # working directory moved to examples/

    file = open(filename, 'r')
    data = json.load(file)
    file.close()

    # redirect screen output to obj
    printout = io.StringIO()
    sys.stdout = printout

    err_msg = []
    err_code = [0,]
    prepend_imports = [] # load environments. In case it is missing from the running cell

    for icell, cell in enumerate(data['cells']):
        if cell['cell_type'] != 'code':
            continue

        ############################################### Turn it to error when published
        if cell['execution_count'] == None:
            warnings.warn('Cell {:d} of Notebook examples/{}: Reference run missing. Please report to developers.'.format(icell+1, filename))
            continue

        tmpcode = open('tmpcode.py', 'w') # write tmp script
        for p in prepend_imports:
            tmpcode.write('%s' % p)
        for iline, line in enumerate(cell['source']):
            tmpcode.write('%s' % line)
            if 'import' in line:
                prepend_imports.append(line)
        # If the cell runs
        try:
            execfile('tmpcode.py')
        except Exception as e:
            tmpcode.close()
            tmpcode = open('tmpcode.py', 'r')
            codeinfo = tmpcode.read()
            tmpcode.close()

            err_code.append(2)
            err_msg.append("""
Error occurs in Cell {:d}, Notebook 'examples/{}'.
  It can either be a logic error with the developed code, or your environment setups.
  Make sure that all the dependencies of CRYSTALpytools are installed properly.
  If you keep getting the same error, please open an issue in
  'https://github.com/crystal-code-tools/CRYSTALpytools/issues'.

Error massage:

{}

Code:

{}
""".format(icell+1, filename, e.__str__(), codeinfo))
            continue

        # If output matches ref
        if exact_out == True:
            ref = ''
            if len(cell['outputs']) == 0:
                continue

            for out in cell['outputs'][0]['text']:
                ref += out
            if ref != printout.getvalue():
                err_code.append(1)
                err_msg.append("""
Exact matching between the current and reference outputs of Cell {:d}, Notebook 'examples/{}' fails.
  Please take extra care of your CRYSTALpytools distribution and testing files.
  Either of them might be out-of-date or have been modified.
  Plese make sure that versions of CRYSTALpytools and testing scripts are consistent.

Output of current run:

{}
""".format(icell+1, filename, printout.getvalue()))

        tmpcode.close()

    if os.path.exists('tmpcode.py'):
        os.remove('tmpcode.py')
    status = max(err_code)
    return status, err_msg


def run_all():
    # Main function
    import sys, os, warnings

    # exact matching of output
    exact_match = ['convert.ipynb', 'crystal_io.ipynb', 'thermo_file_readwrite.ipynb']

    whereami = os.path.abspath(os.getcwd()) # CRYSTALpytools/
    moduledir = os.path.dirname(whereami) # CRYSTALpytools/../

    try:
        import CRYSTALpytools
    except ImportError:
        warnings.warn('CRYSTALpytools does not exist in default path, the one in current directory will be used. Please make sure that is what you want.')
        sys.path.append(moduledir)
        import CRYSTALpytools

    if os.path.exists('{}/examples'.format(moduledir)) == False:
        raise FileNotFoundError('{}/examples directory does not exist. No testing available.'.format(moduledir))
    else:
        nbdir = '{}/examples'.format(moduledir)  # Notebook dir


    ferr = open('errors.log', 'w')
    ferr.write('******************************************************************************\n')
    for file in os.listdir(nbdir):
        if '.ipynb' in file:
            if file in exact_match:
                err, msg = run_notebook(file, nbdir, True)
            else:
                err, msg = run_notebook(file, nbdir, False)
            ferr.write('%s%s%s%4i\n' % ('Runtime status code of example code in examples/', file, ':', err))
            ferr.write('%s\n' % '  0 for error')
            ferr.write('%s\n' % '  1 for exact matching of output error - typically not fatal, but be careful.');
            ferr.write('%s\n' % '  2 for errors in running code block - fatal error.')
            ferr.write('%s\n' % 'Detailed error message below.')
            for m in msg:
                ferr.write('%s' % m)
            if err == 2:
                ferr.close()
                raise Exception('Fatal error encountered in {}. Read errors.log for details.'.format(file))
            elif err == 1:
                warnings.warn('Inconsistent outputs generated in {}. Read errors.log for details'.format(file))
            else:
                ferr.write('\n')
                ferr.write('******************************************************************************\n')
        else:
            continue

    ferr.close()
