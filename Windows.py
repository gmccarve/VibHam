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
from HamilP import Hamil, Wavefunctions
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
                cc[j] = self.coef[j] / ang_m**j / D_au


            self.df = pd.DataFrame({"Order (n)" : np.arange(0, self.coef.size),
                                    "Value (au/m^n)" : self.coef,
                                    "Value (D/Ang^n)" : cc
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

    signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        self.elements = ['H', 'He', 
                         'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                         'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                         'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                         'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                         'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                         'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
                           
        
        pos1  = ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
        pos2  = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
        pos3  = ['Sc', 'Y', 'La', 'Ac']
        pos4  = ['Ti', 'Zr', 'Hf', 'Rf', 'Ce', 'Th']
        pos5  = ['V', 'Nb', 'Ta', 'Db', 'Pr', 'Pa']
        pos6  = ['Cr', 'Mo', 'W', 'Sg', 'Nd', 'U']
        pos7  = ['Mn', 'Tc', 'Re', 'Bh', 'Pm', 'Np']
        pos8  = ['Fe', 'Ru', 'Os', 'Hs', 'Sm', 'Pu']
        pos9  = ['Co', 'Rh', 'Ir', 'Mt', 'Eu', 'Am']
        pos10 = ['Ni', 'Pd', 'Pt', 'Ds', 'Gd', 'Cm']
        pos11 = ['Cu', 'Ag', 'Au', 'Rg', 'Tb', 'Bk']
        pos12 = ['Zn', 'Cd', 'Hg', 'Cn', 'Dy', 'Cf']
        pos13 = ['B', 'Al', 'Ga', 'In', 'Tl', 'Nh', 'Ho', 'Es']
        pos14 = ['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl', 'Er', 'Fm']
        pos15 = ['N', 'P', 'As', 'Sb', 'Bi', 'Mc', 'Tm', 'Md']
        pos16 = ['O', 'S', 'Se', 'Te', 'Po', 'Lv', 'Yb', 'No']
        pos17 = ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'Lu', 'Lr']
        pos18 = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']

        pos_ = [pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, pos11, pos12, pos13, pos14, pos15, pos16, pos17, pos18]

        row1  = ['H', 'He']
        row2  = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
        row3  = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
        row4  = ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']
        row5  = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe']
        row6  = ['Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
        row7  = ['Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        row9  = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        row10 = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

        row_ = [row1, row2, row3, row4, row5, row6, row7, [''], row9, row10]

        
        alkali     = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
        alkaline   = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
        transition = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                      'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
                      'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                      'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']
        lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        actinides   = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        metalloids = ['B', 'Si', 'Ge', 'As', 'Sb', 'Te']
        nonmetals = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I']
        unknown = ['Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        noble = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
        post = ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At']

        groups_ = [alkali, alkaline, transition, lanthanides, actinides, metalloids, nonmetals, unknown, noble, post]
        
        colors = ['#0099ff', '#ff8080', '#9999ff', '#ff4dd2', '#bf80ff', '#66ff33', '#CCCCFF', '#afafb6', '#668cff', '#33ff99']

        width=30

        self.atom1 = QLabel("H")
        self.atom2 = QLabel("H")

        self.grid_layout = QGridLayout()

        for element in self.elements:
            btn = QPushButton(element, self)
            btn.setFixedWidth(width)
            btn.clicked.connect(lambda ch, element=element: self.__addElements(element))

            c = 0
            for pos in pos_:
                if element in pos:
                    cc = c
                c+=1

            r = 0
            for row in row_:
                if element in row:
                    rr = r
                r+=1

            color = 0
            for group in groups_:
                if element in group:
                    btn.setStyleSheet('background-color : ' + colors[color])
                color += 1

            self.grid_layout.addWidget(btn, rr, cc, 1, 1)

            if element == 'Be':
                self.grid_layout.addWidget(QLabel("Atom 1 - "), 1, 5, 1, 2)
                self.grid_layout.addWidget(self.atom1, 1, 7, 1, 2)

            if element == 'Mg':
                self.grid_layout.addWidget(QLabel("Atom 2 - "), 2, 5, 1, 2)
                self.grid_layout.addWidget(self.atom2, 2, 7, 1, 2)

        
        self.grid_layout.addWidget(QLabel(""), 7, 1, 1, 1)

        self.setLayout(self.grid_layout)

        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

    def __exit_program(self):
        self.close()


    def __addElements(self, element):
        self.atom1.setText(self.atom2.text())
        self.atom2.setText(element)
    
        self.signal.emit(self.atom1.text(), self.atom2.text())
    




