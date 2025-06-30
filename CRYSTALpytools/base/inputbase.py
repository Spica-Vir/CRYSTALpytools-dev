#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base object of all the input (d12/d3) blocks.
"""
import numpy as np
from warnings import warn

class BlockBASE():
    """
    The base class of 'block' objects

    Args:
        bg (str): Beginning line. Keyword or title
        ed (str): Ending line.
        sep (str): Separator between keyword and value pairs.
        line_ed (str): Line ending.
        dic (dict): Keyword and value (in text) pairs. Format is listed below.

    Returns:
        self (BlockBASE): New attributes listed below
        self._block_bg (str): Keyword of the block
        self._block_ed (str): End of block indicator
        self._separator (str): Separator between keyword and value pairs.
        self._block_data (str) : Formatted text string for input file.
        self._block_dict (dict): Keyword and value (in text) pairs. Key: Input
            keyword in string. Value: A 3\*1 or 4\*1 list, see below.
            * 1st element: For keywords: String, formatted output; 'None', not including the keyword; '', keyword only
            * 1st element: For subblocks: Name of the 'real' attribute (protected by property decorator)
            * 2nd: bool, whether it is a sub-block or not;
            * 3rd: A list of conflicting keywords.
            * 4th: Subblock only. String that initializes the subblock object.
        self._block_key (list): Allowed keyword list
        self._block_valid (bool): Whether this block is valid and for print.
    """

    def __init__(self, bg, ed, sep, line_ed, dic):
        from CRYSTALpytools import io

        self._block_bg = bg # Title or keyword of the block
        self._block_ed = ed # End of block indicator
        self._separator = sep # Separator between keyword and value pairs.
        self._line_ending = line_ed # Line ending indicator.
        self._block_data = '' # Formatted text string
        self._block_dict = dic # Data
        self._block_valid = True # Whether this block is valid and for print

        key = list(self._block_dict.keys())
        self._block_key = sorted(set(key), key=key.index)
        for k in key:
            if self._block_dict[k][1] == True: # Initialize all the subblock objs
                obj = eval(self._block_dict[k][3])
                obj._block_valid = False
                setattr(self, self._block_dict[k][0], obj)

    def __call__(self, obj=''):
        if type(obj) == str:
            self.__init__()
            if obj != '': self.analyze_text(obj)
        elif obj is None:
            self.__init__()
            self._block_valid = False
        elif type(obj) == type(self):
            self = obj
        else:
            raise Exception('Unknown data type.')

    @property
    def data(self):
        """Settings in all the attributes are summarized here."""
        if self._block_valid == False:
            warn("This block is not visible. Set 'self._block_valid = True' to get data", stacklevel=2)
            return ''

        self.update_block()
        text = ''
        for i in [self._block_bg, self._block_data, self._block_ed]:
            text += i
        return text

    def assign_keyword(self, key, shape, value=''):
        """
        Transform value into string formats.

        Args:
            shape (list[int]): 1D list. Shape of input text. Length: Number of
                lines; Element: Number of values
            value (list | str): List, a 1D list of arguments; '' or a list
                begins with '', return to ''; 'None' or a list begins with
                'None', return 'None'.
        Returns:
            text (str): Formatted input
        """
        # Check the validity of key
        if key not in self._block_key: raise Exception(f"Unknown keyword '{key}'.")

        self.clean_conflict(key)

        if type(value) == np.ndarray: value = value.tolist()
        if type(value) != list and type(value) != tuple:
            value = [value, ]

        # Keyword only or cleanup: Value is '' or None
        if value[0]=='' or value[0] is None:
            self._block_dict[key][0] = value[0]
            return self

        # Wrong input: Number of args defined by shape != input.
        if sum(shape) != len(value):
            raise Exception(f"Inconsistent shapes and values for keyword '{key}'.")

        # Correct number of args
        text = ''
        value_counter = 0
        try:
            for nvalue in shape:
                text += ' '.join([str(v) for v in value[value_counter:value_counter + nvalue]]) + self._line_ending
                value_counter += nvalue
        except IndexError:
            raise Exception("The shape of input data does not satisfy requirements.")
        self._block_dict[key][0] = text
        return self

    @staticmethod
    def set_matrix(mx):
        """Set matrix-like data to get ``assign_keyword()`` inputs. 'Matrix-like'
        means having the same number of rows and cols and having no shape indicator.

        Args:
            mx (list | str): nDimen\*nDimen list, 'None', or ''
        Returns:
            shape (list): 1\*nDimen 1D list. All elements are nDimen.
            value (list): 1\*nDimen^2 1D list. Flattened matrix.
        """
        if type(mx) == np.ndarray: mx = mx.tolist()

        if mx is None:  # Clean data
            return [], None
        elif mx == '':  # Keyword only
            return [], ''

        shape = []; value = []
        for row in mx:
            if len(mx) != len(row): raise ValueError("Input matrix is not a square matrix.")
            shape.append(len(row))
            value.extend(row)
        return shape, value

    @staticmethod
    def set_list(*args):
        """
        Set list-like data to get ``assign_keyword()`` inputs. Used for lists with
        known dimensions.

        Args:
            \*args : '', Clean data; 'None', Return keyword only; 
                int, list, int for length of the list, list for list data
        Returns:
            shape (list): 1 + length 1D list, first element is 1, and the rest
                is the number of elements for each line. Or [].
            args (list): Flattened list, the first element is number of lines. Or [] or ''
        """
        if args[0] is None:  # Clean data
            return [], None
        elif args[0] == '':  # Keyword only
            return [], ''

        if len(args) != 2 or int(args[0]) != len(args[1]):
            return ValueError('Input format error. Arguments should be int + list')
        if type(args[1]) == np.ndarray:
            args[1] = args[1].tolist()

        shape = [1, ]; value = [int(args[0]), ]
        for i in args[1]:
            shape.append(len(i))
            value.extend(i)
        return shape, value

    def clean_conflict(self, key):
        """
        Addressing the conflictions between keywords.

        Args:
            key (str): The keyword specified.
        """
        for cttr in self._block_dict[key][2]:
            cttr = cttr.upper()
            if cttr == key:
                continue
            try:
                if self._block_dict[cttr][1] == False: # keyword
                    if self._block_dict[cttr][0] is not None:
                        warn(f"'{key}' conflicts with the existing '{cttr}'. The old one is deleted.", stacklevel=3)
                        self._block_dict[cttr][0] = None
                else: # subblock
                    obj = getattr(self, self._block_dict[cttr][0])
                    if obj._block_valid == True:
                        warn(f"'{key}' conflicts with the existing '{cttr}'. The old one is deleted.", stacklevel=3)
                        obj(None)
                        setattr(self, self._block_dict[cttr][0], obj)
            except KeyError:
                warn(f"The specified keyword '{cttr}' is not defined. Ignored for conflict check.", stacklevel=3)
                continue
        return self

    def update_block(self):
        """
        Update the ``_block_data`` attribute: Summarizing all the settings to
        ``_block_data`` attribute for inspection and print
        """
        self._block_data = ''
        for key in self._block_key:
            if self._block_dict[key][1] == False: # Keyword-like attributes
                if self._block_dict[key][0] is not None:
                    self._block_data += f"{key}{self._separator}{self._block_dict[key][0]}"
            else: # Block-like attributes, get data from the corresponding attribute
                # It is important to use real attribute here for subblocks
                # To avoid automatically setting objreal._block_valid == True
                objreal = getattr(self, self._block_dict[key][0])
                if objreal._block_valid == True:
                    # It is important to use @property decorated attribute here for sub-sub-blocks
                    # some of them are the same as subblocks but different keywords
                    # The modification method is a decorated attribute in its upper block (self)
                    obj = getattr(self, key.lower())
                    self._block_data += obj.data # update and print subblock
                    setattr(self, self._block_dict[key][0], obj)
        return self

    def analyze_text(self, text, bg_block_label=None, end_block_label=None):
        """
        Analyze the input text and return to corresponding attributes

        Args:
            text (str):
            bg_block_label (str): Marks the begin of the block. 'None' to use 'self._block_bg'.
                '' to use first / last lines.
            end_block_label (str): Marks the end of the block. 'None' to use 'self._block_ed'.
                '' to use first / last lines.
        """
        separator = self._separator
        ending = self._line_ending
        textline = text.strip().split(ending)
        allcapkeys = np.array([k.upper() for k in self._block_key])

        # Range of the block
        if bg_block_label is None: bg_block_label = self._block_bg.strip(ending)
        if end_block_label is None: end_block_label = self._block_ed.strip(ending)

        line_bg = -1;
        if bg_block_label != '':
            for iline, line in enumerate(textline):
                if bg_block_label.upper() in line.upper():
                    line_bg = iline; break

        line_ed = len(textline)+1
        if end_block_label != '':
            for iline, line in enumerate(textline[::-1]):
                if end_block_label.upper() in line.upper():
                    line_ed = len(textline)-iline; break

        # Data in the block
        value = ''
        for iline, t in enumerate(textline[line_bg+1:line_ed-1][::-1]):
            line = t.strip().split(separator)
            guess = line[0].upper()
            iguess = np.where(allcapkeys==guess)[0]
            if len(iguess) > 0: # Keyword line: ending point of saved values
                key = self._block_key[iguess[0]]
                inline = f'{separator}'.join(line[1:]) ## When keyword and value in the same line
                if inline != '': inline += ending
                value =  inline + value
                if self._block_dict[key][1] == False: # Keyword-like attributes
                    if self._block_dict[key][0] is not None:
                        warn(f"Keyword '{key}' exists. The new entry will cover the old one", stacklevel=2)
                    self._block_dict[key][0] = value
                else:  # Block-like attributes
                    obj = getattr(self, self._block_dict[key][0])
                    if obj._block_valid == True:
                        warn(f"Keyword '{key}' exists. The new entry will cover the old one", stacklevel=2)
                    obj(value)
                    setattr(self, self._block_dict[key][0], obj)

                # Clean saved values
                value = ''
            else: # No keyword line
                value = t + ending + value

        # Last lines if unallocated string exists,  saved into beginning lines
        if np.all(value!=''):
            textline = value.strip().split(ending)
            for t in textline:
                self._block_bg = self._block_bg + t + ending
        return self
