import sys
import numpy as np
import pandas as pd
import random
import os
import traceback
import io
import csv

from Conversions import *
from Atoms import Atoms
from Interpolate import Interpolate
from Hamil import Hamil, Wavefunctions
from Spectra import Spectra

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QEvent, QAbstractTableModel, QRect, QPoint, QObject, QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class PlotCurve_111(FigureCanvasQTAgg):
    '''Class used to construct a single matplotlib plot'''

    def __init__(self, Parent=None, dpi=100):
        fig = Figure(dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super(PlotCurve_111, self).__init__(fig)

class PlotCurve_212(FigureCanvasQTAgg):
    '''Class used to construct side-by-side matplotlib plots'''

    def __init__(self, Parent=None, dpi=100):
        fig = Figure(dpi=dpi, tight_layout=True)
        self.axes1 = fig.add_subplot(211)
        self.axes2 = fig.add_subplot(212)
        super(PlotCurve_212, self).__init__(fig)

class PandasModel(QAbstractTableModel):
    '''Class used to open and populate external datatables'''

    def __init__(self, data, edit):
        super().__init__()
        self._data = data
        self._edit = edit

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                value = self._data.iloc[index.row(), index.column()]
                return str(value)

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._data.iloc[index.row(), index.column()] = value
            return True
        return False

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]

    def flags(self, index):
        if self._edit == True:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled


class CoefficientWindow(QWidget):
    '''Class used to open and populate an external window to display power series
       coefficient data'''

    def __init__(self, coef, val):
        super(CoefficientWindow, self).__init__()

        self.resize(400, 400)

        if val == 'energy':
            self.setWindowTitle("Power Series Coefficients")
            self.coef = coef

            cc = np.zeros((coef.size))

            for j in range(cc.size):
                cc[j] = self.coef[j] * ang_m**(j+2) / hart_J


            self.df = pd.DataFrame({"Order (n)" : np.arange(0, self.coef.size)+2,
                                    "Value (j/m^n)" : self.coef,
                                    "Value (Hartree/Ang^n)" : cc
                                    })

        elif val == 'dipole':
            self.setWindowTitle("Dipole Momemnt Coefficients")
            self.coef = coef

            cc = np.zeros((coef.size))

            for j in range(cc.size):
                cc[j] = self.coef[j] * ang_m**j * D_au


            self.df = pd.DataFrame({"Order (n)" : np.arange(0, self.coef.size),
                                    "Value (au/m^n)" : cc,
                                    "Value (D/Ang^n)" : self.coef
                                    })

        self.table = QTableView()
        self.table.installEventFilter(self)
        self.model = PandasModel(self.df, edit=False)
        self.table.setModel(self.model)
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.table)

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def eventFilter(self, source, event):
        '''Function used to copy cells from an external datatable'''
        if (event.type() == QEvent.KeyPress and
            event.matches(QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(CoefficientWindow, self).eventFilter(source, event)

    def copySelection(self):
        '''Function used to copy cells from an external datatable'''
        selection = self.table.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())

class InterpolationErrorWindow(QWidget):
    '''Class used to open and populate an external datatable that displayes
        the error of interpolation for both the energy and dipole curves'''

    def __init__(self, R, error, val):
        super(InterpolationErrorWindow, self).__init__()

        self.R = R
        self.error = error
        self.val = val

        self.resize(800, 400)

        if val == 'energy':
            self.setWindowTitle("Error of Energy Interpolation")
            self.df = pd.DataFrame({"Bond Displacement (Å) " :  self.R,
                                   "Error (Hartrees) " : self.error,
                                   "Error (J) " : self.error * hart_J,
                                   "Error (eV) " : self.error * hart_eV,
                                   "Error (cm^-1) " : self.error * hart_cm,
                                   "Error (kcal/mol) " : self.error * hart_kcal
                                   })

        elif val == 'dipole':
            self.setWindowTitle("Error of Dipole Moment Interpolation")
            self.df = pd.DataFrame({"Bond Displacement (Å) " :  self.R,
                                   "Error (D) " : self.error,
                                   "Error (au) " : self.error * D_au
                                   })

        self.table = QTableView()
        self.table.installEventFilter(self)
        self.model = PandasModel(self.df, edit=False)
        self.table.setModel(self.model)
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.table)

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def eventFilter(self, source, event):
        '''Function used to copy cells from an external datatable'''
        if (event.type() == QEvent.KeyPress and
            event.matches(QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(InterpolationErrorWindow, self).eventFilter(source, event)

    def copySelection(self):
        '''Function used to copy cells from an external datatable'''
        selection = self.table.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())

class StabilityWindow(QWidget):
    '''Class used to open an external window to detail the stability of the
        constructed total Hamiltonian matrix'''

    def __init__(self, trunc, val):
        super(StabilityWindow, self).__init__()

        self.trunc = trunc
        self.val = val

        self.setWindowTitle("Stability of Hamiltonian Matrix")
        self.resize(400, 50)

        self.layout = QVBoxLayout()

        if val == 1:
            self.label = QLabel("Matrix stable ν = " + str(self.trunc))
        else:
            self.label = QLabel("Matrix not stable beyond ν = " + str(self.trunc))

        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

class MatrixWindow(QWidget):
    '''Class used to populate and open an external datatable that displays
        one of the constructed Hamiltonian matrices.

        If the matrix is J-dependent, an initial window is opened asking
        about the J-surface to display.'''

    def __init__(self, *args, **kwargs):
        super(MatrixWindow, self).__init__()

        self.matrix = kwargs['matrix']
        self.val    = kwargs['val']

        self.resize(1000,1000)

        if 'J' in kwargs:
            if kwargs['J'] != 0:
                self.askJ = MatrixWindow_AskJ(J=kwargs['J'],
                                              matrix=self.matrix,
                                              val=self.val)
                self.askJ.show()
            else:
                self.setWindowTitle(self.val.upper())

                self.df = pd.DataFrame()

                for v in range(self.matrix.shape[0]):
                    self.df['ν = ' + str(v)] = self.matrix[v,:]

                self.df = self.df.round(decimals=7)

                self.table = QTableView()
                self.table.installEventFilter(self)
                self.model = PandasModel(self.df, edit=False)
                self.table.setModel(self.model)
                vlayout = QVBoxLayout(self)
                vlayout.addWidget(self.table)
                self.show()
        else:
            self.setWindowTitle(self.val.upper())

            self.df = pd.DataFrame()

            for v in range(self.matrix.shape[0]):
                self.df['ν = ' + str(v)] = self.matrix[v,:]

            self.df = self.df.round(decimals=7)

            self.table = QTableView()
            self.table.installEventFilter(self)
            self.model = PandasModel(self.df, edit=False)
            self.table.setModel(self.model)
            vlayout = QVBoxLayout(self)
            vlayout.addWidget(self.table)
            self.show()

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def eventFilter(self, source, event):
        '''Function used to copy cells from an external datatable'''
        if (event.type() == QEvent.KeyPress and
            event.matches(QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(MatrixWindow, self).eventFilter(source, event)

    def copySelection(self):
        '''Function used to copy cells from an external datatable'''
        selection = self.table.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())

class MatrixWindow_AskJ(QWidget):
    '''Class used to open a window to ask which J-surface to display the centrifugal
        or total Hamiltonian matrices on'''

    def __init__(self, *args, **kwargs):
        super(MatrixWindow_AskJ, self).__init__()

        self.J = kwargs['J']
        self.matrix = kwargs['matrix']
        self.val    = kwargs['val']

        if kwargs['J'] == 0:
            self.J = 0
            self.c = MatrixWindow(matrix=self.matrix[self.J],val=self.val + "(J=" + str(self.J) + ")")
            self.c.show()

        else:
            self.layout = QVBoxLayout()
            self.lab = QLabel("On which J-surface?")
            self.box = QComboBox()
            self.box.addItems([''])
            self.box.addItems([str(e) for e in range(kwargs['J'])])
            self.box.currentIndexChanged.connect(self.__newJval)

            self.layout.addWidget(self.lab)
            self.layout.addWidget(self.box)

            self.setLayout(self.layout)

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def __newJval(self):
        self.J = int(self.box.currentText())
        self.c = MatrixWindow(matrix=self.matrix[self.J],val=self.val + "(J=" + str(self.J) + ")")
        self.c.show()


class TruncationWindowValues(QWidget):
    '''Class to populate and open an external datatable with the error values associated
        with the truncation of the total Hamiltonian matrix'''

    def __init__(self, vals):
        super(TruncationWindowValues, self).__init__()

        self.resize(600,600)

        self.vals = vals

        self.df = pd.DataFrame({'ν' : np.arange(0, self.vals.shape[1])
                               })
        for j in range(self.vals.shape[0]):
            self.df['Errors (J=' + str(j) + ')'] = self.vals[j]

        self.df = self.df.round(decimals=7)

        self.table = QTableView()
        self.model = PandasModel(self.df, edit=False)
        self.table.setModel(self.model)
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.table)

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()


class TruncationWindow(QWidget):
    '''Class to display the vibrational states on differetn J-surfaces which have
        converged according to a truncation of the total Hamiltonian matrix.

        Can open an additional window that shows the truncation errors for all
        vibrational states.'''

    def __init__(self, trunc, vals):
        super(TruncationWindow, self).__init__()

        self.trunc = trunc
        self.vals = vals

        self.setWindowTitle("Convergence of Eigenvalues")
        self.resize(400, 50)

        self.layout = QVBoxLayout()

        self.txt = ''

        for j in range(len(trunc)):
            self.txt += "Eigenvalues converged up to ν = " + str(int(self.trunc[j])+1) + " on the J = " + str(j) + " surface\n"

        self.label = QLabel(self.txt)
        self.layout.addWidget(self.label)

        self.q_label = QLabel("View Truncations Errors?")
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.q_label)

        self.layout.addLayout(self.hbox)

        self.yes_btn = QPushButton("Yes")
        self.no_btn = QPushButton("No")

        self.yes_btn.clicked.connect(self.__view_truncation)
        self.no_btn.clicked.connect(self.__exit_program)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.yes_btn)
        self.hbox.addWidget(self.no_btn)

        self.layout.addLayout(self.hbox)

        self.setLayout(self.layout)

        self.show()

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def __view_truncation(self):
        self.c = TruncationWindowValues(self.vals)
        self.c.show()


class SaveWindow(QWidget):
    '''Class used to open an external dialog box to indicate that the chosen matrix
        has been saved to an external file'''

    def __init__(self, *args, **kwargs):
        super(SaveWindow, self).__init__()

        self.layout = QVBoxLayout()
        self.label = QLabel(str(args[0]))
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.show()

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()


class VibSpecDataTable(QWidget):
    '''Class used to populate and open an external datatable with information
        about the vibrational excitations.

        Can display either populations of vibrational states or intensities
        of excitations.'''

    def __init__(self, *args, **kwargs):
        super(VibSpecDataTable, self).__init__()

        self.resize(600,600)

        v   = kwargs['v']
        j   = kwargs['J']

        self.df = pd.DataFrame()

        if kwargs['method'] == 'rov':
            if kwargs['vib_type'] == 'pop':
                pop = kwargs['pop']
                p_size = int(pop.size/2)

                temp_arr_v = np.zeros((p_size))
                temp_arr_j = np.zeros((p_size))
                temp_arr_e = np.zeros((p_size))
                temp_arr_p = np.zeros((p_size))
                c = 0
                for jj in range(pop.shape[0]):
                    for vv in range(pop.shape[2]):
                        temp_arr_v[c] = vv
                        temp_arr_j[c] = jj
                        temp_arr_e[c] = pop[jj,0,vv]
                        temp_arr_p[c] = pop[jj,1,vv]
                        c += 1

                self.df['v'] = np.asarray(temp_arr_v, dtype=int)
                self.df['J'] = np.asarray(temp_arr_j, dtype=int)
                self.df['Energy (cm^-1)'] = temp_arr_e
                self.df['Population'] = temp_arr_p

            elif kwargs['vib_type'] == 'int':
                inten = kwargs['inten']
                i_size = int(inten.size/4)

                temp_arr_v  = np.zeros((i_size))
                temp_arr_vv = np.zeros((i_size))
                temp_arr_j  = np.zeros((i_size))
                temp_arr_e  = np.zeros((i_size))
                temp_arr_i  = np.zeros((i_size))
                c = 0
                for jj in range(inten.shape[0]):
                    cc = 0
                    for vv_ in range(inten.shape[2]):
                        temp_arr_v[c]  = inten[jj,2,cc]
                        temp_arr_vv[c] = inten[jj,3,cc]
                        temp_arr_j[c]  = jj
                        temp_arr_e[c]  = inten[jj,0,cc]
                        temp_arr_i[c]  = inten[jj,1,cc]
                        c+=1
                        cc+=1

                self.df['v initial'] = np.asarray(temp_arr_v, dtype=int)
                self.df['v final'] = np.asarray(temp_arr_vv, dtype=int)
                self.df['J'] = np.asarray(temp_arr_j, dtype=int)
                self.df['Energy (cm^-1)'] = temp_arr_e
                self.df['Intensity (s^-1)'] = temp_arr_i


        elif kwargs['method'] == 'vib':
            if kwargs['vib_type'] == 'pop':
                self.df['v'] = np.arange(0, v)
                self.df['J'] = np.ones((v), dtype=int)*j
                self.df['Energy (cm^-1)'] = kwargs['pop'][0]
                self.df['Population'] = kwargs['pop'][1]

            elif kwargs['vib_type'] == 'int':
                self.df['v initial'] = np.asarray(kwargs['inten'][2], dtype=int)
                self.df['v final']   = np.asarray(kwargs['inten'][3], dtype=int)
                self.df['J'] = np.ones((kwargs['inten'][0].size), dtype=int)*j
                self.df['Energy (cm^-1)'] = kwargs['inten'][0]
                self.df['Intensity (s^-1)'] = kwargs['inten'][1]

        self.df = self.df[self.df['Energy (cm^-1)'] != 0]

        self.table = QTableView()
        self.table.installEventFilter(self)
        self.model = PandasModel(self.df, edit=False)
        self.table.setModel(self.model)
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.table)
        self.show()

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def eventFilter(self, source, event):
        '''Function used to copy cells from an external datatable'''
        if (event.type() == QEvent.KeyPress and
            event.matches(QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(VibSpecDataTable, self).eventFilter(source, event)

    def copySelection(self):
        '''Function used to copy cells from an external datatable'''
        selection = self.table.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())


class RotSpecDataTable(QWidget):
    '''Class used to populate and open an external datatable with information
        about the rotational excitations.

        Can display either populations of rotational states or intensities
        of excitations.'''

    def __init__(self, *args, **kwargs):
        super(RotSpecDataTable, self).__init__()
        v   = kwargs['v']
        J   = kwargs['J']

        self.resize(600,600)

        self.df = pd.DataFrame()

        if kwargs['method'] == 'rov':
            if kwargs['rot_type'] == 'pop':
                pop = kwargs['pop']
                p_size = int(pop.size/2)

                temp_arr_v = np.zeros((p_size))
                temp_arr_j = np.zeros((p_size))
                temp_arr_e = np.zeros((p_size))
                temp_arr_p = np.zeros((p_size))
                c = 0
                for vv in range(pop.shape[0]):
                    for jj in range(pop.shape[2]):
                        temp_arr_v[c] = vv
                        temp_arr_j[c] = jj
                        temp_arr_e[c] = pop[vv,0,jj]
                        temp_arr_p[c] = pop[vv,1,jj]
                        c += 1

                self.df['v'] = np.asarray(temp_arr_v, dtype=int)
                self.df['J'] = np.asarray(temp_arr_j, dtype=int)
                self.df['Energy (cm^-1)'] = temp_arr_e
                self.df['Population'] = temp_arr_p

            elif kwargs['rot_type'] == 'int':
                inten = kwargs['inten']
                i_size = int(inten.size/4)

                temp_arr_j  = np.zeros((i_size))
                temp_arr_jj = np.zeros((i_size))
                temp_arr_v  = np.zeros((i_size))
                temp_arr_e  = np.zeros((i_size))
                temp_arr_i  = np.zeros((i_size))
                c = 0
                for vv in range(inten.shape[0]):
                    cc = 0
                    for jj_ in range(inten.shape[2]):
                        temp_arr_j[c]  = inten[vv,2,cc]
                        temp_arr_jj[c] = inten[vv,3,cc]
                        temp_arr_v[c]  = vv
                        temp_arr_e[c]  = inten[vv,0,cc]
                        temp_arr_i[c]  = inten[vv,1,cc]
                        c+=1
                        cc+=1

                self.df['J initial'] = np.asarray(temp_arr_j, dtype=int)
                self.df['J final'] = np.asarray(temp_arr_jj, dtype=int)
                self.df['v'] = np.asarray(temp_arr_v, dtype=int)
                self.df['Energy (cm^-1)'] = temp_arr_e
                self.df['Intensity (s^-1)'] = temp_arr_i


        elif kwargs['method'] == 'rot':
            if kwargs['rot_type'] == 'pop':
                pop = kwargs['pop']
                self.df['J'] = np.arange(0, J)
                self.df['v'] = np.ones((J), dtype=int)*v
                self.df['Energy (cm^-1)'] = kwargs['pop'][0]
                self.df['Population'] = kwargs['pop'][1]

            elif kwargs['rot_type'] == 'int':
                self.df['J initial'] = np.asarray(kwargs['inten'][2], dtype=int)
                self.df['J final']   = np.asarray(kwargs['inten'][3], dtype=int)
                self.df['v'] = np.ones((kwargs['inten'][0].size), dtype=int)*v
                self.df['Energy (cm^-1)'] = kwargs['inten'][0]
                self.df['Intensity (s^-1)'] = kwargs['inten'][1]

        self.df = self.df[self.df['Energy (cm^-1)'] != 0]

        self.table = QTableView()
        self.table.installEventFilter(self)
        self.model = PandasModel(self.df, edit=False)
        self.table.setModel(self.model)
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.table)
        self.show()

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def eventFilter(self, source, event):
        '''Function used to copy cells from an external datatable'''
        if (event.type() == QEvent.KeyPress and
            event.matches(QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(RotSpecDataTable, self).eventFilter(source, event)

    def copySelection(self):
        '''Function used to copy cells from an external datatable'''
        selection = self.table.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())

class RovSpecDataTable(QWidget):
    '''Class used to populate and open an external datatable with information
        about the rovibrational excitations.

        Can display either populations of rovibrational states or intensities
        of excitations.'''

    def __init__(self, *args, **kwargs):
        super(RovSpecDataTable, self).__init__()
        v   = kwargs['v']
        J   = kwargs['J']

        self.resize(600,600)

        self.df = pd.DataFrame()

        if kwargs['rov_type'] == 'pop':
            pop = kwargs['pop']
            self.df['v'] = np.asarray(pop[3], dtype=int)
            self.df['J'] = np.asarray(pop[2], dtype=int)
            self.df['Energy (cm^-1)'] = pop[0]
            self.df['Population'] = pop[1]

        elif kwargs['rov_type'] == 'int':
            inten = kwargs['inten']

            self.df['v initial'] = np.asarray(inten[0], dtype=int)
            self.df['J initial'] = np.asarray(inten[1], dtype=int)
            self.df['v final'] = np.asarray(inten[2], dtype=int)
            self.df['J final'] = np.asarray(inten[3], dtype=int)
            self.df['E initial'] = inten[4]
            self.df['E final'] = inten[5]
            self.df['Energy (cm^-1)'] = inten[6]
            self.df['Intensity (s^-1)'] = inten[7]

        self.table = QTableView()
        self.table.installEventFilter(self)
        self.model = PandasModel(self.df, edit=False)
        self.table.setModel(self.model)
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.table)
        self.show()

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

    def eventFilter(self, source, event):
        '''Function used to copy cells from an external datatable'''
        if (event.type() == QEvent.KeyPress and
            event.matches(QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(RovSpecDataTable, self).eventFilter(source, event)

    def copySelection(self):
        '''Function used to copy cells from an external datatable'''
        selection = self.table.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())


class ErrorWindow(QWidget):
    '''Class used to open a dialog box with error information'''

    def __init__(self, errorText):
        super().__init__()

        self.txt = errorText
        self.setWindowTitle("Error")
        self.resize(100,50)
        layout = QVBoxLayout()
        self.label = QLabel(self.txt)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()

class TabBar(QTabBar):
    '''Class used to contruct the tab-based main window'''

    def tabSizeHint(self, index):
        s = QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)
            painter.save()

            s = opt.rect.size()
            s.transpose()
            r = QRect(QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r

            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(90)
            painter.translate(-c)
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt);
            painter.restore()

class ProxyStyle(QProxyStyle):
    '''Class used to modify the base proxy style for the tab-based main window'''

    def drawControl(self, element, opt, painter, widget):
        if element == QStyle.CE_TabBarTabLabel:
            ic = self.pixelMetric(QStyle.PM_TabBarIconSize)
            r = QRect(opt.rect)
            w =  0 if opt.icon.isNull() else opt.rect.width() + self.pixelMetric(QStyle.PM_TabBarIconSize)
            r.setHeight(opt.fontMetrics.width(opt.text) + w)
            r.moveBottom(opt.rect.bottom())
            opt.rect = r
        QProxyStyle.drawControl(self, element, opt, painter, widget)

class ElementWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.atom1 = ''
        self.atom2 = ''


        self.h_btn = QPushButton("H")
        self.he_btn = QPushButton("He")
        self.li_btn = QPushButton("Li")
        self.be_btn = QPushButton("Be")
        self.b_btn = QPushButton("B")
        self.c_btn = QPushButton("C")
        self.n_btn = QPushButton("N")
        self.o_btn = QPushButton("O")
        self.f_btn = QPushButton("F")
        self.ne_btn = QPushButton("Ne")
        self.na_btn = QPushButton("Na")
        self.mg_btn = QPushButton("Mg")
        self.al_btn = QPushButton("Al")
        self.si_btn = QPushButton("Si")
        self.p_btn = QPushButton("P")
        self.s_btn = QPushButton("S")
        self.cl_btn = QPushButton("Cl")
        self.ar_btn = QPushButton("Ar")
        self.k_btn = QPushButton("K")
        self.ca_btn = QPushButton("Ca")
        self.sc_btn = QPushButton("Sc")
        self.ti_btn = QPushButton("Ti")
        self.v_btn = QPushButton("V")
        self.cr_btn = QPushButton("Cr")
        self.mn_btn = QPushButton("Mn")
        self.fe_btn = QPushButton("Fe")
        self.co_btn = QPushButton("Co")
        self.ni_btn = QPushButton("Ni")
        self.cu_btn = QPushButton("Cu")
        self.zn_btn = QPushButton("Zn")
        self.ga_btn = QPushButton("Ga")
        self.ge_btn = QPushButton("Ge")
        self.as_btn = QPushButton("As")
        self.se_btn = QPushButton("Se")
        self.br_btn = QPushButton("Br")
        self.kr_btn = QPushButton("Kr")
        self.rb_btn = QPushButton("Rb")
        self.sr_btn = QPushButton("Sr")
        self.y_btn = QPushButton("Y")
        self.zr_btn = QPushButton("Zr")
        self.nb_btn = QPushButton("Nb")
        self.mo_btn = QPushButton("Mo")
        self.tc_btn = QPushButton("Tc")
        self.ru_btn = QPushButton("Ru")
        self.rh_btn = QPushButton("Rh")
        self.pd_btn = QPushButton("Pd")
        self.ag_btn = QPushButton("Ag")
        self.cd_btn = QPushButton("Cd")
        self.in_btn = QPushButton("In")
        self.sn_btn = QPushButton("Sn")
        self.sb_btn = QPushButton("Sb")
        self.te_btn = QPushButton("Te")
        self.i_btn = QPushButton("I")
        self.xe_btn = QPushButton("Xe")
        self.cs_btn = QPushButton("Cs")
        self.ba_btn = QPushButton("Ba")
        self.la_btn = QPushButton("La")
        self.ce_btn = QPushButton("Ce")
        self.pr_btn = QPushButton("Pr")
        self.nd_btn = QPushButton("Nd")
        self.pm_btn = QPushButton("Pm")
        self.sm_btn = QPushButton("Sm")
        self.eu_btn = QPushButton("Eu")
        self.gd_btn = QPushButton("Gd")
        self.tb_btn = QPushButton("Tb")
        self.dy_btn = QPushButton("Dy")
        self.ho_btn = QPushButton("Ho")
        self.er_btn = QPushButton("Er")
        self.tm_btn = QPushButton("Tm")
        self.yb_btn = QPushButton("Yb")
        self.lu_btn = QPushButton("Lu")
        self.hf_btn = QPushButton("Hf")
        self.ta_btn = QPushButton("Ta")
        self.w_btn = QPushButton("W")
        self.re_btn = QPushButton("Re")
        self.os_btn = QPushButton("Os")
        self.ir_btn = QPushButton("Ir")
        self.pt_btn = QPushButton("Pt")
        self.au_btn = QPushButton("Au")
        self.hg_btn = QPushButton("Hg")
        self.tl_btn = QPushButton("Tl")
        self.pb_btn = QPushButton("Pb")
        self.bi_btn = QPushButton("Bi")
        self.po_btn = QPushButton("Po")
        self.at_btn = QPushButton("At")
        self.rn_btn = QPushButton("Rn")
        self.fr_btn = QPushButton("Fr")
        self.ra_btn = QPushButton("Ra")
        self.ac_btn = QPushButton("Ac")
        self.th_btn = QPushButton("Th")
        self.pa_btn = QPushButton("Pa")
        self.u_btn = QPushButton("U")
        self.np_btn = QPushButton("Np")
        self.pu_btn = QPushButton("Pu")
        self.am_btn = QPushButton("Am")
        self.cm_btn = QPushButton("Cm")
        self.bk_btn = QPushButton("Bk")
        self.cf_btn = QPushButton("Cf")
        self.es_btn = QPushButton("Es")
        self.fm_btn = QPushButton("Fm")
        self.md_btn = QPushButton("Md")
        self.no_btn = QPushButton("No")
        self.lr_btn = QPushButton("Lr")
        self.rf_btn = QPushButton("Rf")
        self.db_btn = QPushButton("Db")
        self.sg_btn = QPushButton("Sg")
        self.bh_btn = QPushButton("Bh")
        self.hs_btn = QPushButton("Hs")
        self.mt_btn = QPushButton("Mt")
        self.ds_btn = QPushButton("Ds")
        self.rg_btn = QPushButton("Rg")
        self.cn_btn = QPushButton("Cn")
        self.nh_btn = QPushButton("Nh")
        self.fl_btn = QPushButton("Fl")
        self.mc_btn = QPushButton("Mc")
        self.lv_btn = QPushButton("Lv")
        self.ts_btn = QPushButton("Ts")
        self.og_btn = QPushButton("Og")

        width=25

        self.h_btn.setFixedWidth(width)
        self.he_btn.setFixedWidth(width)
        self.li_btn.setFixedWidth(width)
        self.be_btn.setFixedWidth(width)
        self.b_btn.setFixedWidth(width)
        self.c_btn.setFixedWidth(width)
        self.n_btn.setFixedWidth(width)
        self.o_btn.setFixedWidth(width)
        self.f_btn.setFixedWidth(width)
        self.ne_btn.setFixedWidth(width)
        self.na_btn.setFixedWidth(width)
        self.mg_btn.setFixedWidth(width)
        self.al_btn.setFixedWidth(width)
        self.si_btn.setFixedWidth(width)
        self.p_btn.setFixedWidth(width)
        self.s_btn.setFixedWidth(width)
        self.cl_btn.setFixedWidth(width)
        self.ar_btn.setFixedWidth(width)
        self.k_btn.setFixedWidth(width)
        self.ca_btn.setFixedWidth(width)
        self.sc_btn.setFixedWidth(width)
        self.ti_btn.setFixedWidth(width)
        self.v_btn.setFixedWidth(width)
        self.cr_btn.setFixedWidth(width)
        self.mn_btn.setFixedWidth(width)
        self.fe_btn.setFixedWidth(width)
        self.co_btn.setFixedWidth(width)
        self.ni_btn.setFixedWidth(width)
        self.cu_btn.setFixedWidth(width)
        self.zn_btn.setFixedWidth(width)
        self.ga_btn.setFixedWidth(width)
        self.ge_btn.setFixedWidth(width)
        self.as_btn.setFixedWidth(width)
        self.se_btn.setFixedWidth(width)
        self.br_btn.setFixedWidth(width)
        self.kr_btn.setFixedWidth(width)
        self.rb_btn.setFixedWidth(width)
        self.sr_btn.setFixedWidth(width)
        self.y_btn.setFixedWidth(width)
        self.zr_btn.setFixedWidth(width)
        self.nb_btn.setFixedWidth(width)
        self.mo_btn.setFixedWidth(width)
        self.tc_btn.setFixedWidth(width)
        self.ru_btn.setFixedWidth(width)
        self.rh_btn.setFixedWidth(width)
        self.pd_btn.setFixedWidth(width)
        self.ag_btn.setFixedWidth(width)
        self.cd_btn.setFixedWidth(width)
        self.in_btn.setFixedWidth(width)
        self.sn_btn.setFixedWidth(width)
        self.sb_btn.setFixedWidth(width)
        self.te_btn.setFixedWidth(width)
        self.i_btn.setFixedWidth(width)
        self.xe_btn.setFixedWidth(width)
        self.cs_btn.setFixedWidth(width)
        self.ba_btn.setFixedWidth(width)
        self.la_btn.setFixedWidth(width)
        self.ce_btn.setFixedWidth(width)
        self.pr_btn.setFixedWidth(width)
        self.nd_btn.setFixedWidth(width)
        self.pm_btn.setFixedWidth(width)
        self.sm_btn.setFixedWidth(width)
        self.eu_btn.setFixedWidth(width)
        self.gd_btn.setFixedWidth(width)
        self.tb_btn.setFixedWidth(width)
        self.dy_btn.setFixedWidth(width)
        self.ho_btn.setFixedWidth(width)
        self.er_btn.setFixedWidth(width)
        self.tm_btn.setFixedWidth(width)
        self.yb_btn.setFixedWidth(width)
        self.lu_btn.setFixedWidth(width)
        self.hf_btn.setFixedWidth(width)
        self.ta_btn.setFixedWidth(width)
        self.w_btn.setFixedWidth(width)
        self.re_btn.setFixedWidth(width)
        self.os_btn.setFixedWidth(width)
        self.ir_btn.setFixedWidth(width)
        self.pt_btn.setFixedWidth(width)
        self.au_btn.setFixedWidth(width)
        self.hg_btn.setFixedWidth(width)
        self.tl_btn.setFixedWidth(width)
        self.pb_btn.setFixedWidth(width)
        self.bi_btn.setFixedWidth(width)
        self.po_btn.setFixedWidth(width)
        self.at_btn.setFixedWidth(width)
        self.rn_btn.setFixedWidth(width)
        self.fr_btn.setFixedWidth(width)
        self.ra_btn.setFixedWidth(width)
        self.ac_btn.setFixedWidth(width)
        self.th_btn.setFixedWidth(width)
        self.pa_btn.setFixedWidth(width)
        self.u_btn.setFixedWidth(width)
        self.np_btn.setFixedWidth(width)
        self.pu_btn.setFixedWidth(width)
        self.am_btn.setFixedWidth(width)
        self.cm_btn.setFixedWidth(width)
        self.bk_btn.setFixedWidth(width)
        self.cf_btn.setFixedWidth(width)
        self.es_btn.setFixedWidth(width)
        self.fm_btn.setFixedWidth(width)
        self.md_btn.setFixedWidth(width)
        self.no_btn.setFixedWidth(width)
        self.lr_btn.setFixedWidth(width)
        self.rf_btn.setFixedWidth(width)
        self.db_btn.setFixedWidth(width)
        self.sg_btn.setFixedWidth(width)
        self.bh_btn.setFixedWidth(width)
        self.hs_btn.setFixedWidth(width)
        self.mt_btn.setFixedWidth(width)
        self.ds_btn.setFixedWidth(width)
        self.rg_btn.setFixedWidth(width)
        self.cn_btn.setFixedWidth(width)
        self.nh_btn.setFixedWidth(width)
        self.fl_btn.setFixedWidth(width)
        self.mc_btn.setFixedWidth(width)
        self.lv_btn.setFixedWidth(width)
        self.ts_btn.setFixedWidth(width)
        self.og_btn.setFixedWidth(width)

        

        self.h_btn.setStyleSheet("background-color : cyan")
        self.he_btn.setStyleSheet("background-color : red")
        self.li_btn.setStyleSheet("background-color : blue")
        self.be_btn.setStyleSheet("background-color : darkgreen")
        self.b_btn.setStyleSheet("background-color : yellow")
        self.c_btn.setStyleSheet("background-color : cyan")
        self.n_btn.setStyleSheet("background-color : cyan")
        self.o_btn.setStyleSheet("background-color : cyan")
        self.f_btn.setStyleSheet("background-color : cyan")
        self.ne_btn.setStyleSheet("background-color : red")
        self.na_btn.setStyleSheet("background-color : blue")
        self.mg_btn.setStyleSheet("background-color : darkgreen")
        self.al_btn.setStyleSheet("background-color : green")
        self.si_btn.setStyleSheet("background-color : yellow")
        self.p_btn.setStyleSheet("background-color : cyan")
        self.s_btn.setStyleSheet("background-color : cyan")
        self.cl_btn.setStyleSheet("background-color : cyan")
        self.ar_btn.setStyleSheet("background-color : red")
        self.k_btn.setStyleSheet("background-color : blue")
        self.ca_btn.setStyleSheet("background-color : darkgreen")
        self.sc_btn.setStyleSheet("background-color : darkred")
        self.ti_btn.setStyleSheet("background-color : darkred")
        self.v_btn.setStyleSheet("background-color : darkred")
        self.cr_btn.setStyleSheet("background-color : darkred")
        self.mn_btn.setStyleSheet("background-color : darkred")
        self.fe_btn.setStyleSheet("background-color : darkred")
        self.co_btn.setStyleSheet("background-color : darkred")
        self.ni_btn.setStyleSheet("background-color : darkred")
        self.cu_btn.setStyleSheet("background-color : darkred")
        self.zn_btn.setStyleSheet("background-color : darkred")
        self.ga_btn.setStyleSheet("background-color : green")
        self.ge_btn.setStyleSheet("background-color : yellow")
        self.as_btn.setStyleSheet("background-color : yellow")
        self.se_btn.setStyleSheet("background-color : cyan")
        self.br_btn.setStyleSheet("background-color : cyan")
        self.kr_btn.setStyleSheet("background-color : red")
        self.rb_btn.setStyleSheet("background-color : blue")
        self.sr_btn.setStyleSheet("background-color : darkgreen")
        self.y_btn.setStyleSheet("background-color : darkred")
        self.zr_btn.setStyleSheet("background-color : darkred")
        self.nb_btn.setStyleSheet("background-color : darkred")
        self.mo_btn.setStyleSheet("background-color : darkred")
        self.tc_btn.setStyleSheet("background-color : darkred")
        self.ru_btn.setStyleSheet("background-color : darkred")
        self.rh_btn.setStyleSheet("background-color : darkred")
        self.pd_btn.setStyleSheet("background-color : darkred")
        self.ag_btn.setStyleSheet("background-color : darkred")
        self.cd_btn.setStyleSheet("background-color : darkred")
        self.in_btn.setStyleSheet("background-color : green")
        self.sn_btn.setStyleSheet("background-color : green")
        self.sb_btn.setStyleSheet("background-color : yellow")
        self.te_btn.setStyleSheet("background-color : yellow")
        self.i_btn.setStyleSheet("background-color : cyan")
        self.xe_btn.setStyleSheet("background-color : red")
        self.cs_btn.setStyleSheet("background-color : blue")
        self.ba_btn.setStyleSheet("background-color : darkgreen")
        self.la_btn.setStyleSheet("background-color : darkcyan")
        self.ce_btn.setStyleSheet("background-color : darkcyan")
        self.pr_btn.setStyleSheet("background-color : darkcyan")
        self.nd_btn.setStyleSheet("background-color : darkcyan")
        self.pm_btn.setStyleSheet("background-color : darkcyan")
        self.sm_btn.setStyleSheet("background-color : darkcyan")
        self.eu_btn.setStyleSheet("background-color : darkcyan")
        self.gd_btn.setStyleSheet("background-color : darkcyan")
        self.tb_btn.setStyleSheet("background-color : darkcyan")
        self.dy_btn.setStyleSheet("background-color : darkcyan")
        self.ho_btn.setStyleSheet("background-color : darkcyan")
        self.er_btn.setStyleSheet("background-color : darkcyan")
        self.tm_btn.setStyleSheet("background-color : darkcyan")
        self.yb_btn.setStyleSheet("background-color : darkcyan")
        self.lu_btn.setStyleSheet("background-color : darkcyan")
        self.hf_btn.setStyleSheet("background-color : darkred")
        self.ta_btn.setStyleSheet("background-color : darkred")
        self.w_btn.setStyleSheet("background-color : darkred")
        self.re_btn.setStyleSheet("background-color : darkred")
        self.os_btn.setStyleSheet("background-color : darkred")
        self.ir_btn.setStyleSheet("background-color : darkred")
        self.pt_btn.setStyleSheet("background-color : darkred")
        self.au_btn.setStyleSheet("background-color : darkred")
        self.hg_btn.setStyleSheet("background-color : darkred")
        self.tl_btn.setStyleSheet("background-color : green")
        self.pb_btn.setStyleSheet("background-color : green")
        self.bi_btn.setStyleSheet("background-color : green")
        self.po_btn.setStyleSheet("background-color : green")
        self.at_btn.setStyleSheet("background-color : green")
        self.rn_btn.setStyleSheet("background-color : red")
        self.fr_btn.setStyleSheet("background-color : blue")
        self.ra_btn.setStyleSheet("background-color : darkgreen")
        self.ac_btn.setStyleSheet("background-color : magenta")
        self.th_btn.setStyleSheet("background-color : magenta")
        self.pa_btn.setStyleSheet("background-color : magenta")
        self.u_btn.setStyleSheet("background-color : magenta")
        self.np_btn.setStyleSheet("background-color : magenta")
        self.pu_btn.setStyleSheet("background-color : magenta")
        self.am_btn.setStyleSheet("background-color : magenta")
        self.cm_btn.setStyleSheet("background-color : magenta")
        self.bk_btn.setStyleSheet("background-color : magenta")
        self.cf_btn.setStyleSheet("background-color : magenta")
        self.es_btn.setStyleSheet("background-color : magenta")
        self.fm_btn.setStyleSheet("background-color : magenta")
        self.md_btn.setStyleSheet("background-color : magenta")
        self.no_btn.setStyleSheet("background-color : magenta")
        self.lr_btn.setStyleSheet("background-color : magenta")
        self.rf_btn.setStyleSheet("background-color : darkred")
        self.db_btn.setStyleSheet("background-color : darkred")
        self.sg_btn.setStyleSheet("background-color : darkred")
        self.bh_btn.setStyleSheet("background-color : darkred")
        self.hs_btn.setStyleSheet("background-color : darkred")
        self.mt_btn.setStyleSheet("background-color : gray")
        self.ds_btn.setStyleSheet("background-color : gray")
        self.rg_btn.setStyleSheet("background-color : gray")
        self.cn_btn.setStyleSheet("background-color : gray")
        self.nh_btn.setStyleSheet("background-color : gray")
        self.fl_btn.setStyleSheet("background-color : gray")
        self.mc_btn.setStyleSheet("background-color : gray")
        self.lv_btn.setStyleSheet("background-color : gray")
        self.ts_btn.setStyleSheet("background-color : gray")
        self.og_btn.setStyleSheet("background-color : gray")
        

        self.grid_layout = QGridLayout()

        row = 0
        self.grid_layout.addWidget(self.h_btn,  row, 0,  1, 1)
        self.grid_layout.addWidget(self.he_btn, row, 17, 1, 1)
       
        row+=1
        self.grid_layout.addWidget(self.li_btn, row, 0,  1, 1)
        self.grid_layout.addWidget(self.be_btn, row, 1,  1, 1)
        self.grid_layout.addWidget(QLabel("Atom 1 - "), row, 5, 1, 2)
        self.grid_layout.addWidget(QLabel(self.atom1), row, 7, 1, 2)
        self.grid_layout.addWidget(self.b_btn,  row, 12, 1, 1)
        self.grid_layout.addWidget(self.c_btn,  row, 13, 1, 1)
        self.grid_layout.addWidget(self.n_btn,  row, 14, 1, 1)
        self.grid_layout.addWidget(self.o_btn,  row, 15, 1, 1)
        self.grid_layout.addWidget(self.f_btn,  row, 16, 1, 1)
        self.grid_layout.addWidget(self.ne_btn, row, 17, 1, 1)

        row+=1
        self.grid_layout.addWidget(self.na_btn, row, 0,  1, 1)
        self.grid_layout.addWidget(self.mg_btn, row, 1,  1, 1)
        self.grid_layout.addWidget(QLabel("Atom 2 - "), row, 5, 1, 2)
        self.grid_layout.addWidget(QLabel(self.atom2), row, 7, 1, 2)
        self.grid_layout.addWidget(self.al_btn, row, 12, 1, 1)
        self.grid_layout.addWidget(self.si_btn, row, 13, 1, 1)
        self.grid_layout.addWidget(self.p_btn,  row, 14, 1, 1)
        self.grid_layout.addWidget(self.s_btn,  row, 15, 1, 1)
        self.grid_layout.addWidget(self.cl_btn, row, 16, 1, 1)
        self.grid_layout.addWidget(self.ar_btn, row, 17, 1, 1)

        row+=1
        self.grid_layout.addWidget(self.k_btn,  row, 0,  1, 1)
        self.grid_layout.addWidget(self.ca_btn, row, 1,  1, 1)
        self.grid_layout.addWidget(self.sc_btn, row, 2,  1, 1)
        self.grid_layout.addWidget(self.ti_btn, row, 3,  1, 1)
        self.grid_layout.addWidget(self.v_btn,  row, 4,  1, 1)
        self.grid_layout.addWidget(self.cr_btn, row, 5,  1, 1)
        self.grid_layout.addWidget(self.mn_btn, row, 6,  1, 1)
        self.grid_layout.addWidget(self.fe_btn, row, 7,  1, 1)
        self.grid_layout.addWidget(self.co_btn, row, 8,  1, 1)
        self.grid_layout.addWidget(self.ni_btn, row, 9,  1, 1)
        self.grid_layout.addWidget(self.cu_btn, row, 10, 1, 1)
        self.grid_layout.addWidget(self.zn_btn, row, 11, 1, 1)
        self.grid_layout.addWidget(self.ga_btn, row, 12, 1, 1)
        self.grid_layout.addWidget(self.ge_btn, row, 13, 1, 1)
        self.grid_layout.addWidget(self.as_btn, row, 14, 1, 1)
        self.grid_layout.addWidget(self.se_btn, row, 15, 1, 1)
        self.grid_layout.addWidget(self.br_btn, row, 16, 1, 1)
        self.grid_layout.addWidget(self.kr_btn, row, 17, 1, 1)

        row+=1
        self.grid_layout.addWidget(self.rb_btn, row, 0,  1, 1)
        self.grid_layout.addWidget(self.sr_btn, row, 1,  1, 1)
        self.grid_layout.addWidget(self.y_btn,  row, 2,  1, 1)
        self.grid_layout.addWidget(self.zr_btn, row, 3,  1, 1)
        self.grid_layout.addWidget(self.nb_btn, row, 4,  1, 1)
        self.grid_layout.addWidget(self.mo_btn, row, 5,  1, 1)
        self.grid_layout.addWidget(self.tc_btn, row, 6,  1, 1)
        self.grid_layout.addWidget(self.ru_btn, row, 7,  1, 1)
        self.grid_layout.addWidget(self.rh_btn, row, 8,  1, 1)
        self.grid_layout.addWidget(self.pd_btn, row, 9,  1, 1)
        self.grid_layout.addWidget(self.ag_btn, row, 10, 1, 1)
        self.grid_layout.addWidget(self.cd_btn, row, 11, 1, 1)
        self.grid_layout.addWidget(self.in_btn, row, 12, 1, 1)
        self.grid_layout.addWidget(self.sn_btn, row, 13, 1, 1)
        self.grid_layout.addWidget(self.sb_btn, row, 14, 1, 1)
        self.grid_layout.addWidget(self.te_btn, row, 15, 1, 1)
        self.grid_layout.addWidget(self.i_btn,  row, 16, 1, 1)
        self.grid_layout.addWidget(self.xe_btn, row, 17, 1, 1)

        row+=1
        self.grid_layout.addWidget(self.cs_btn, row, 0,  1, 1)
        self.grid_layout.addWidget(self.ba_btn, row, 1,  1, 1)
        self.grid_layout.addWidget(self.la_btn, row, 2,  1, 1)
        self.grid_layout.addWidget(self.hf_btn, row, 3,  1, 1)
        self.grid_layout.addWidget(self.ta_btn, row, 4,  1, 1)
        self.grid_layout.addWidget(self.w_btn,  row, 5,  1, 1)
        self.grid_layout.addWidget(self.re_btn, row, 6,  1, 1)
        self.grid_layout.addWidget(self.os_btn, row, 7,  1, 1)
        self.grid_layout.addWidget(self.ir_btn, row, 8,  1, 1)
        self.grid_layout.addWidget(self.pt_btn, row, 9,  1, 1)
        self.grid_layout.addWidget(self.au_btn, row, 10, 1, 1)
        self.grid_layout.addWidget(self.hg_btn, row, 11, 1, 1)
        self.grid_layout.addWidget(self.tl_btn, row, 12, 1, 1)
        self.grid_layout.addWidget(self.pb_btn, row, 13, 1, 1)
        self.grid_layout.addWidget(self.bi_btn, row, 14, 1, 1)
        self.grid_layout.addWidget(self.po_btn, row, 15, 1, 1)
        self.grid_layout.addWidget(self.at_btn, row, 16, 1, 1)
        self.grid_layout.addWidget(self.rn_btn, row, 17, 1, 1)

        row+=1
        self.grid_layout.addWidget(self.fr_btn, row, 0,  1, 1)
        self.grid_layout.addWidget(self.ra_btn, row, 1,  1, 1)
        self.grid_layout.addWidget(self.ac_btn, row, 2,  1, 1)
        self.grid_layout.addWidget(self.rf_btn, row, 3,  1, 1)
        self.grid_layout.addWidget(self.db_btn, row, 4,  1, 1)
        self.grid_layout.addWidget(self.sg_btn, row, 5,  1, 1)
        self.grid_layout.addWidget(self.bh_btn, row, 6,  1, 1)
        self.grid_layout.addWidget(self.hs_btn, row, 7,  1, 1)
        self.grid_layout.addWidget(self.mt_btn, row, 8,  1, 1)
        self.grid_layout.addWidget(self.ds_btn, row, 9,  1, 1)
        self.grid_layout.addWidget(self.rg_btn, row, 10, 1, 1)
        self.grid_layout.addWidget(self.cn_btn, row, 11, 1, 1)
        self.grid_layout.addWidget(self.nh_btn, row, 12, 1, 1)
        self.grid_layout.addWidget(self.fl_btn, row, 13, 1, 1)
        self.grid_layout.addWidget(self.mc_btn, row, 14, 1, 1)
        self.grid_layout.addWidget(self.lv_btn, row, 15, 1, 1)
        self.grid_layout.addWidget(self.ts_btn, row, 16, 1, 1)
        self.grid_layout.addWidget(self.og_btn, row, 17, 1, 1)

        row+=1
        self.grid_layout.addWidget(QLabel(""), row, 0, 1, 16)

        row+=1
        self.grid_layout.addWidget(self.ce_btn, row, 3,   1, 1)
        self.grid_layout.addWidget(self.pr_btn, row, 4,   1, 1)
        self.grid_layout.addWidget(self.nd_btn, row, 5,   1, 1)
        self.grid_layout.addWidget(self.pm_btn, row, 6,   1, 1)
        self.grid_layout.addWidget(self.sm_btn, row, 7,   1, 1)
        self.grid_layout.addWidget(self.eu_btn, row, 8,   1, 1)
        self.grid_layout.addWidget(self.gd_btn, row, 9,   1, 1)
        self.grid_layout.addWidget(self.tb_btn, row, 10,  1, 1)
        self.grid_layout.addWidget(self.dy_btn, row, 11,  1, 1)
        self.grid_layout.addWidget(self.ho_btn, row, 12,  1, 1)
        self.grid_layout.addWidget(self.er_btn, row, 13,  1, 1)
        self.grid_layout.addWidget(self.tm_btn, row, 14,  1, 1)
        self.grid_layout.addWidget(self.yb_btn, row, 15,  1, 1)
        self.grid_layout.addWidget(self.lu_btn, row, 16,  1, 1)


        row+=1
        self.grid_layout.addWidget(self.th_btn, row, 3,   1, 1)
        self.grid_layout.addWidget(self.pa_btn, row, 4,   1, 1)
        self.grid_layout.addWidget(self.u_btn,  row, 5,   1, 1)
        self.grid_layout.addWidget(self.np_btn, row, 6,   1, 1)
        self.grid_layout.addWidget(self.pu_btn, row, 7,   1, 1)
        self.grid_layout.addWidget(self.am_btn, row, 8,   1, 1)
        self.grid_layout.addWidget(self.cm_btn, row, 9,   1, 1)
        self.grid_layout.addWidget(self.bk_btn, row, 10,  1, 1)
        self.grid_layout.addWidget(self.cf_btn, row, 11,  1, 1)
        self.grid_layout.addWidget(self.es_btn, row, 12,  1, 1)
        self.grid_layout.addWidget(self.fm_btn, row, 13,  1, 1)
        self.grid_layout.addWidget(self.md_btn, row, 14,  1, 1)
        self.grid_layout.addWidget(self.no_btn, row, 15,  1, 1)
        self.grid_layout.addWidget(self.lr_btn, row, 16,  1, 1)





        self.setLayout(self.grid_layout)


        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()


