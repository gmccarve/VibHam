#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import random
import os
import traceback
import io
import csv
from pathlib import Path

from Conversions import *
from Atoms import Atoms
from Interpolate import Interpolate
from Hamil import Hamil, Wavefunctions
from Spectra import Spectra
from Windows import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QEvent, QAbstractTableModel, QRect, QPoint, QObject, QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 


class MainWindow(QMainWindow):
    '''Main window of the VibHam application'''
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.showMaximized()

        self._tabwidget = TabWidget(self)
        self.setCentralWidget(self._tabwidget)

class TabWidget(QTabWidget):
    '''Tab widget used to separate the many different functions into individual tabs'''
    def __init__(self, *args, **kwargs):
        QTabWidget.__init__(self, *args, **kwargs)
        self.setTabBar(TabBar(self))
        self.setTabPosition(QTabWidget.West)

        # Exit program
        self.exit_shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        self.exit_shortcut.activated.connect(self.__exit_program)

        # Initialize the Tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        self.tab6 = QWidget()
        self.tab7 = QWidget()
        self.tab8 = QWidget()
        self.tab9 = QWidget()

        # Add tabs
        self.addTab(self.tab1, "&Molecular Properties")
        self.addTab(self.tab2, "Power Series &Expansion")
        self.addTab(self.tab3, "Vibrational &Hamiltonian")
        self.addTab(self.tab4, "Spectroscopic &Constants")
        self.addTab(self.tab5, "E&xcitations")
        self.addTab(self.tab6, "&Dunham Parameters")
        self.addTab(self.tab7, "Wave&functions")
        self.addTab(self.tab8, "&Turning Points")
        self.addTab(self.tab9, "Simulated &Spectra")

        # Populate tabs

        self.tab1.grid_layout = self.__MolecularPropertiesTab()
        self.tab2.grid_layout = self.__PowerSeriesExpansionTab()
        self.tab3.grid_layout = self.__VibrationalHamiltonianTab()
        self.tab4.grid_layout = self.__SpectroscopicConstantsTab()
        self.tab5.grid_layout = self.__ExcitationsTab()
        self.tab6.grid_layout = self.__DunhamTab()
        self.tab7.grid_layout = self.__WavefunctionsTab()
        self.tab8.grid_layout = self.__TurningPointsTab()
        self.tab9.grid_layout = self.__SimulatedSpectraTab()

        self.path = os.path.abspath(__file__)[:-6]


    def __exit_program(selaf):
        '''Function used to exit the program and close all windows'''
        exit()

    def __openErrorMessage(self):
        '''Function used to open all error windows used for the VibHam GUI'''
        self.EW = ErrorWindow(self.errorText)
        self.EW.show()

    def __diagonalize(self, M):
        '''Function used to diagonalize square matrices or tensors composed of square matrices.'''
        try:
            val = np.zeros((M.shape[:2]))
            vec = np.zeros((M.shape))
            for j in range(M.shape[0]):
                val_, vec_ = np.linalg.eig(M[j])
                idx = val_.argsort()
                val[j] = val_[idx]
                vec[j] = vec_[:,idx].T
        except:
            val, vec = np.linalg.eig(M)
            idx = val.argsort()
            val = val[idx]
            vec = vec[:,idx]
        return val*J_cm, vec

    def eventFilter(self, source, event):
        '''Function used to copy data from internal tables.'''
        if (event.type() == QEvent.KeyPress and
            event.matches(QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(TabWidget, self).eventFilter(source, event)

    def copySelection(self):
        '''Function used to copy data from internal tables.'''
        list_of_tables = [self.data_table, 
                          self.eigenvalue_table, 
                          self.vib_spec_table,
                          self.rot_spec_table,
                          self.rovib_spec_table,
                          self.excitations_table,
                          self.dunham_coef_table,
                          self.dunham_param_table,
                          self.tps_table
                          ]
        
        for table in list_of_tables:
            selection = table.selectedIndexes()
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


    def __MolecularPropertiesTab(self):

        '''
        Tab Number 1 - Molecular Properties

        Used to read in a 2- or 3-column data and convert the data therein to appropriate units

        Functions:

            Load data file
            Assign atoms
            Assign isotopes
            Assign masses
            Assign energy units
            Assign length units
            Assign dipole moment units

        '''

        # Initialize some of the variables
        
        self.filename = ''                      # Data file to load
        self.data_ = np.zeros((3, 1))           # Initial numpy matrix of zeros
        self.data = self.data_.copy()
    
        self.Atoms = Atoms()                    # Dictionary of atomic dictionaries

        self.atom1 = self.Atoms.AtomDict['H']   # Dictionary of isotopes and their masses for atom #1
        self.atom2 = self.Atoms.AtomDict['H']   # Dictionary of isotopes and their masses for atom #2

        self.iso1 = 0       # Identity of Isotope #1
        self.iso2 = 0       # Identity of Isotope #2

        self.mass1 = self.atom1[self.iso1]     # Mass of specific isotope for atom #1
        self.mass2 = self.atom2[self.iso2]     # Mass of specific isotope for atom #2

        self.energy_unit = 'Hartrees'       # Energy unit of data file
        self.length_unit = 'Å'              # Distance unit of data file
        self.dipole_unit = 'D'              # Dipole unit of data file
        
        self.charge = 0             # Charge of diatomic

        # Load Example File

        self.load_HF_example_btn = QPushButton("&Load HF Example")
        self.load_HF_example_btn.clicked.connect(self.__load_HF_example_data)
        self.load_HF_example_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.load_HF_example_shortcut.activated.connect(self.__load_HF_example_data)
        self.load_HF_example_btn.setFixedWidth(120)

        self.load_CO_example_btn = QPushButton("Load CO Example")
        self.load_CO_example_btn.clicked.connect(self.__load_CO_example_data)
        self.load_CO_example_btn.setFixedWidth(120)

        # Location of datafile
        
        self.browse_files = QPushButton("&Open File")
        self.browse_files.clicked.connect(self.__browsefiles)
        self.browse_files.clicked.connect(self.__show_datatable)
        self.browse_files.clicked.connect(self.__plot_datatable)
        self.browse_files.setFixedWidth(120)
        self.browse_files.setToolTip("Browse for a datafile")

        self.browse_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self.browse_shortcut.activated.connect(self.__browsefiles)
        self.browse_shortcut.activated.connect(self.__show_datatable)
        self.browse_shortcut.activated.connect(self.__plot_datatable)

        self.loc = QLineEdit(self)
        self.loc.setText(self.filename)
        self.loc.setReadOnly(True)

        # Show the loaded data file

        self.data_table = QTableWidget()
        self.data_table.setFixedWidth(440)
        self.data_table.installEventFilter(self)

        # Refresh the data arrays

        self.clear_data_btn = QPushButton("Clear Data")
        self.clear_data_btn.clicked.connect(self.__clear_data_values)
        self.clear_data_btn.setFixedWidth(120)        

        self.refresh_data_btn = QPushButton("&Refresh Data")
        self.refresh_data_btn.clicked.connect(self.__refresh_data_values)
        self.refresh_data_btn.setFixedWidth(120)

        self.refresh_data_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self.refresh_data_shortcut.activated.connect(self.__refresh_data_values)
        self.refresh_data_shortcut.activated.connect(self.__show_eigenvalue_table)

        # Plot the loaded data file

        self.data_plot = PlotCurve_212(self)
        self.toolbar1 = NavigationToolbar2QT(self.data_plot, self)

        # Define the atomic information
        self.atom1_lab = QLabel("Atom #1")
        self.atom2_lab = QLabel("Atom #2")
        self.atom1_lab.setFixedWidth(100)
        self.atom2_lab.setFixedWidth(100)

        # Define the atoms

        self.element_lab = QPushButton("Elements")
        self.element_lab.clicked.connect(self.__elementsWindow)
        self.element_lab.setToolTip("Atoms in periodic order")

        self.atom1_id = QComboBox()
        self.atom1_id.addItems(self.Atoms.AtomDict.keys())
        self.atom1_id.setStyleSheet("QComboBox { combobox-popup: 0}")
        self.atom1_id.setMaxVisibleItems(10)
        self.atom1_id.currentIndexChanged.connect(self.__atom1_id_changed)
        self.atom1_id.currentIndexChanged.connect(self.__iso1_combo_selected)
        self.atom1_id.currentIndexChanged.connect(self.__mass1_str_changed)
        self.atom1_id.setFixedWidth(100)

        self.atom2_id = QComboBox()
        self.atom2_id.addItems(self.Atoms.AtomDict.keys())
        self.atom2_id.setStyleSheet("QComboBox { combobox-popup: 0}")
        self.atom2_id.setMaxVisibleItems(10)
        self.atom2_id.currentIndexChanged.connect(self.__atom2_id_changed)
        self.atom2_id.currentIndexChanged.connect(self.__iso2_combo_selected)
        self.atom2_id.currentIndexChanged.connect(self.__mass2_str_changed)
        self.atom2_id.setFixedWidth(100)

        # Define the Isotopes

        self.iso_lab  = QLabel("Isotopes")
        self.iso_lab.setToolTip("List of Isotopes \n\n'0' corresponds to the most \nabundant isotope")

        self.iso1_box = QComboBox()
        self.iso1_box.addItems([str(e) for e in self.atom1.keys()])
        self.iso1_box.currentIndexChanged.connect(self.__iso1_combo_selected)
        self.iso1_box.setFixedWidth(100)
    
        self.iso2_box = QComboBox()
        self.iso2_box.addItems([str(e) for e in self.atom2.keys()])
        self.iso2_box.currentIndexChanged.connect(self.__iso2_combo_selected)
        self.iso2_box.setFixedWidth(100)

        # Define the Masses

        self.mass_lab = QLabel("Mass")
        self.mass_lab.setToolTip("Mass of atom (amu)\n\nCan be based on given isotope\n or manually inputed")

        self.mass1_str = QLineEdit(str(round(self.mass1, 8)))
        self.mass1_str.editingFinished.connect(self.__mass1_str_changed)
        self.mass1_str.setFixedWidth(100)

        self.mass2_str = QLineEdit(str(round(self.mass2, 8)))
        self.mass2_str.editingFinished.connect(self.__mass2_str_changed)
        self.mass2_str.setFixedWidth(100)

        # Define the molecular charge

        self.charge_box = QSpinBox()
        self.charge_box.setRange(-10, 10)
        self.charge_box.setFixedWidth(100)
        self.charge_box.valueChanged.connect(self.__charge_box_changed)

        # Define the choices for the input energy

        self.energy_box = QComboBox()
        self.energy_box.addItems(['Hartrees', 'kcal/mol', 'kj/mol', 'eV', 'j', 'wavenumbers'])
        self.energy_box.currentIndexChanged.connect(self.__energy_box_selected)
        self.energy_box.currentIndexChanged.connect(self.__plot_datatable)
        self.energy_box.setFixedWidth(100)

        self.length_box = QComboBox()
        self.length_box.addItems(['Å', 'm', 'bohr'])
        self.length_box.currentIndexChanged.connect(self.__length_box_selected)
        self.length_box.currentIndexChanged.connect(self.__plot_datatable)
        self.length_box.setFixedWidth(100)

        self.dipole_box = QComboBox()
        self.dipole_box.addItems(["D", "au"])
        self.dipole_box.currentIndexChanged.connect(self.__dipole_box_selected)
        self.dipole_box.currentIndexChanged.connect(self.__plot_datatable)
        self.dipole_box.setFixedWidth(100)

        # Define the layout of the tab using a grid

        self.spacer_right = QSpacerItem(1, 10, QSizePolicy.Expanding)
        self.dotted_line1 = QLabel("-"*70)
        self.dotted_line2 = QLabel("-"*70)
        self.dotted_line0 = QLabel("-"*70)

        self.tab1.grid_layout = QGridLayout()

        row=0
        self.tab1.grid_layout.addWidget(self.load_HF_example_btn, row, 0, 1, 1, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.load_CO_example_btn, row, 1, 1, 1, alignment=Qt.AlignCenter)

        self.tab1.grid_layout.addWidget(self.data_table, row, 5, 41, 1)
        self.tab1.grid_layout.addWidget(self.data_plot, row, 7, 41, 1)

        row+=1
        self.tab1.grid_layout.addWidget(self.clear_data_btn, row, 0, 1, 1, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.refresh_data_btn, row, 1, 1, 1, alignment=Qt.AlignCenter)
        
        row+=1
        self.tab1.grid_layout.addWidget(self.browse_files, row, 0, 1, 1)
        self.tab1.grid_layout.addWidget(self.loc, row, 1, 1, 4)

        self.spacer_right1 = QSpacerItem(1, 10, QSizePolicy.Expanding)
        self.tab1.grid_layout.addItem(self.spacer_right1, row, 7)

        row+=1
        self.tab1.grid_layout.addWidget(QLabel(""), row, 0, 1, 5, alignment=Qt.AlignCenter)
    
        row+=1
        self.tab1.grid_layout.addWidget(self.dotted_line1, row, 0, 1, 5, alignment=Qt.AlignCenter)
        
        row+=1
        self.tab1.grid_layout.addWidget(QLabel(""), row, 0, 1, 5, alignment=Qt.AlignCenter)

        row+=2
        self.tab1.grid_layout.addWidget(self.atom1_lab, row, 1, 1, 2)
        self.tab1.grid_layout.addWidget(self.atom2_lab, row, 3, 1, 2)
    
        row+=1
        self.tab1.grid_layout.addWidget(self.element_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.atom1_id, row, 1, 1, 2)
        self.tab1.grid_layout.addWidget(self.atom2_id, row, 3, 1, 2)
    
        row+=1
        self.tab1.grid_layout.addWidget(self.iso_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.iso1_box, row, 1, 1, 2)
        self.tab1.grid_layout.addWidget(self.iso2_box, row, 3, 1, 2)
    
        row+=1
        self.tab1.grid_layout.addWidget(self.mass_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.mass1_str, row, 1, 1, 2)
        self.tab1.grid_layout.addWidget(self.mass2_str, row, 3, 1, 2)

        row+=1
        self.tab1.grid_layout.addWidget(QLabel(""), row, 0, 1, 5, alignment=Qt.AlignCenter)

        row+=1
        self.tab1.grid_layout.addWidget(self.dotted_line2, row, 0, 1, 5, alignment=Qt.AlignCenter)

        row+=1
        self.tab1.grid_layout.addWidget(QLabel(""), row, 0, 1, 5, alignment=Qt.AlignCenter)

        row+=1
        self.tab1.grid_layout.addWidget(QLabel("Energy Units"), row, 0, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.energy_box, row, 1, 1, 2)
    
        row+=1
        self.tab1.grid_layout.addWidget(QLabel("Length Units"), row, 0, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.length_box, row, 1, 1, 2)

        row+=1
        self.tab1.grid_layout.addWidget(QLabel("Dipole Units"), row, 0, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.dipole_box, row, 1, 1, 2)

        row+=1
        self.tab1.grid_layout.addWidget(QLabel("Charge"), row, 0, alignment=Qt.AlignCenter)
        self.tab1.grid_layout.addWidget(self.charge_box, row, 1, 1, 2)
        
        for j in range(25):
            row+=1
            self.tab1.grid_layout.addWidget(QLabel(""), row, 0, 1, 3, alignment=Qt.AlignCenter)

        row+=1
        self.tab1.grid_layout.addWidget(self.toolbar1, row, 7, 1, 3, alignment=Qt.AlignCenter)


        self.tab1.setLayout(self.tab1.grid_layout)

    def __load_CO_example_data(self):
        '''Used to load example data for carbon monoxide (CO)'''
        try:
            self.filename = self.path + "/Examples/CO/CO.txt"
            self.data_ = np.loadtxt(self.filename).T
            self.data = self.data_.copy()
            self.loc.setText(self.filename)
        except:
            self.filename = self.path + "\Examples\CO\CO.txt"
            self.data_ = np.loadtxt(self.filename).T
            self.data = self.data_.copy()
            self.loc.setText(self.filename)

        self.atom1 = self.Atoms.AtomDict['C']
        self.atom2 = self.Atoms.AtomDict['O']

        self.iso1 = 0
        self.iso2 = 0

        self.mass1 = self.atom1[self.iso1]
        self.mass2 = self.atom2[self.iso2]

        self.atom1_id.setCurrentText("C")
        self.atom2_id.setCurrentText("O")

        self.__iso1_combo_selected()
        self.__iso2_combo_selected()
        self.__mass1_str_changed()
        self.__mass2_str_changed()

        self.data_plot.axes1.cla()
        self.data_plot.axes2.cla()

        self.__show_datatable()
        self.__plot_datatable()

    def __load_HF_example_data(self):
        '''Used to load example data for hydrogen fluoride (HF)'''
        try:
            self.filename = self.path + "/Examples/HF/HF.txt"
            self.data_ = np.loadtxt(self.filename).T
            self.data = self.data_.copy()
            self.loc.setText(self.filename)
        except:
            self.filename = self.path + "\Examples\HF\HF.txt"
            self.data_ = np.loadtxt(self.filename).T
            self.data = self.data_.copy()
            self.loc.setText(self.filename)

        self.atom1 = self.Atoms.AtomDict['H']
        self.atom2 = self.Atoms.AtomDict['F']

        self.iso1 = 0
        self.iso2 = 0

        self.mass1 = self.atom1[self.iso1]
        self.mass2 = self.atom2[self.iso2]

        self.atom1_id.setCurrentText("H")
        self.atom2_id.setCurrentText("F")
        
        self.__iso1_combo_selected()
        self.__iso2_combo_selected()
        self.__mass1_str_changed()
        self.__mass2_str_changed()


        self.data_plot.axes1.cla()
        self.data_plot.axes2.cla()

        self.__show_datatable()
        self.__plot_datatable()

    def __browsefiles(self):
        '''Open a file system to load a specific data file'''
        try:
            self.__clear_data_values()
            fname = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd())
            if fname != ('', ''):
                self.data_ = np.loadtxt(fname[0]).T
                self.data = self.data_.copy()
                self.loc.setText(fname[0])
                
                self.__plot_datatable()
                self.__show_datatable()
        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __show_datatable(self):
        '''Display the loaded data file to an internal table'''
        try:
            if self.data_.shape[0] == 2:
                self.data_labels = ["R", "E", "Include?"]

            elif self.data_.shape[0] == 3:
                self.data_labels = ["R", "E", "D", "Include?"]

            else:
                self.data_labels = []
        
            self.data_table.setColumnCount(len(self.data_labels))
            self.data_table.rowCount()
            self.data_table.setHorizontalHeaderLabels(self.data_labels)

            while self.data_table.rowCount() > 0:
                self.data_table.removeRow(0)

            self.list_checkbox = []
            for i in range(self.data_.shape[1]):
                self.list_checkbox.append('')
                self.__add_data_table_row([*self.data_[:,i], self.list_checkbox[i]])

        except AttributeError:
            self.errorText = "File Not Yet Loaded\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()

    def __add_data_table_row(self, row_data):
        '''Used to add a new row to the internal data table'''
        row = self.data_table.rowCount()
        self.data_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            if item == row_data[-1]:
                cell = QTableWidgetItem(item)
                cell.setText(item)
                cell.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                cell.setCheckState(Qt.Checked)
            else:
                cell = QTableWidgetItem(str(item))
            self.data_table.setItem(row, col, cell)
            col += 1

    def __refresh_data_values(self):
        '''Refresh the data table. Useful when loading an external file'''
        try:
            self.data_include_vals = []
            for row in range(self.data_.shape[1]):
                if self.data_table.item(row, len(self.data_labels)-1).checkState() == 2:
                    self.data_include_vals.append(row)

            self.data = self.data_.T[self.data_include_vals].T
            self.__plot_datatable()
        except:
            self.errorText = "File Not Yet Loaded\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()     

    def __clear_data_values(self):
        '''Clear the data table of all values. Useful for loading data'''
        self.data_ = np.zeros((3, 1))
        self.data = np.zeros((3, 1))

        self.loc.setText("")
 
        self.__show_datatable()
        self.__plot_datatable()

    def __plot_datatable(self):
        '''Used to plot the information in the data table in 1-2 plots'''
        try:
            self.data[0]
        except:
            return

        if self.data.shape[0] == 3:
            try:
                self.data_plot.axes1.cla()
                self.data_plot.axes1.scatter(self.data[0], self.data[1], color='b')
                self.data_plot.axes1.set_xlabel("Bond Length (" + self.length_unit + ")")
                self.data_plot.axes1.set_ylabel("Energy (" + self.energy_unit + ")")
                self.data_plot.axes1.grid()

                self.data_plot.axes2.cla()
                self.data_plot.axes2.scatter(self.data[0], self.data[2], color='r')
                self.data_plot.axes2.set_xlabel("Bond Length (" + self.length_unit + ")")
                self.data_plot.axes2.set_ylabel("Dipole (" + self.dipole_unit + ")")
                self.data_plot.axes2.grid()

                self.data_plot.draw()
            except:
                self.errorText = "Plot could not be constructed\n\n" + str(traceback.format_exc())
                self.__openErrorMessage()

        elif self.data.shape[0] == 2:
            try:
                self.data_plot.axes1.cla()
                self.data_plot.axes1.scatter(self.data[0], self.data[1], color='b')
                self.data_plot.axes1.set_xlabel("Bond Length (" + self.length_unit + ")")
                self.data_plot.axes1.set_ylabel("Energy (" + self.energy_unit + ")")
                self.data_plot.axes1.grid()

                self.data_plot.axes2.text(0.5, 0.5, "No Dipole Data To Display", horizontalalignment='center')

                self.data_plot.draw()
            except:
                self.errorText = "Plot could not be constructed\n\n" + str(traceback.format_exc())
                self.__openErrorMessage()


    def __elementsWindow(self):
        self.c = ElementWindow()
        self.c.signal.connect(self.__change_atoms_)
        self.c.show()

    def __change_atoms_(self, atom1, atom2):
        self.atom1_id.setText(atom1)
        self.atom2_id.setText(atom2)

    def __atom1_id_changed(self):
        '''Activated following a change for atom 1'''
        try:
            self.atom1 = self.Atoms.AtomDict[self.atom1_id.currentText()]

            self.iso1_box.clear()
            self.iso1_box.addItems([str(e) for e in self.atom1])
            self.iso1 = self.iso1_box.currentText()

            self.mass1_str.clear()
            self.mass1 = self.atom1[int(self.iso1)]
            self.mass1_str.setText(str(round(self.mass1, 8)))

        except:
            self.errorText = "Atom not found\n\n"  + str(traceback.format_exc()) 
            self.__openErrorMessage()


    def __atom2_id_changed(self):
        '''Activated following a change of atom 2'''
        try:
            self.atom2 = self.Atoms.AtomDict[self.atom2_id.currentText()]

            self.iso2_box.clear()
            self.iso2_box.addItems([str(e) for e in self.atom2])
            self.iso2 = self.iso2_box.currentText()

            self.mass2_str.clear()
            self.mass2 = self.atom1[int(self.iso2)]
            self.mass2_str.setText(str(round(self.mass2, 8)))

        except:
            self.errorText = "Atom not found"
            self.__openErrorMessage()


    def __iso1_combo_selected(self):
        '''Activated following a change in the combo box for the identify of isotope 1'''
        self.iso1 = self.iso1_box.currentText()

        try:
            self.mass1_str.clear()
            self.mass1 = self.atom1[int(self.iso1)]
            self.mass1_str.setText(str(round(self.mass1, 8)))
        except ValueError:
            pass

    def __iso2_combo_selected(self):
        '''Activated following a change in the combo box for the identify of isotope 2'''
        self.iso2 = self.iso2_box.currentText()

        try:
            self.mass2_str.clear()
            self.mass2 = self.atom2[int(self.iso2)]
            self.mass2_str.setText(str(round(self.mass2, 8)))
        except ValueError:
            pass

    def __mass1_str_changed(self):
        '''Activated following a change in the value for the mass of atom 1'''
        try:
            self.mass1 = float(self.mass1_str.text())
        except:
            self.errorText = "Mass must be of type float" + str(traceback.format_exc())
            self.__openErrorMessage()
            self.mass1_str.clear()
            self.mass1 = self.atom1[int(self.iso1)]
            self.mass1_str.setText(str(self.mass1))

    def __mass2_str_changed(self):
        '''Activated following a change in the value for the mass of atom 2'''
        try:
            self.mass2 = float(self.mass2_str.text())
        except:
            self.errorText = "Mass must be of type float\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()
            self.mass2_str.clear()
            self.mass2 = self.atom1[int(self.iso2)]
            self.mass2_str.setText(str(self.mass2))

    def __energy_box_selected(self):
        '''Activated following a change in the combo box for the energy unit'''
        self.energy_unit = self.energy_box.currentText()

    def __length_box_selected(self):
        '''Activated following a change in the combo box for the length unit'''
        self.length_unit = self.length_box.currentText()

    def __dipole_box_selected(self):
        '''Activated following a change in the combo box for the dipole moment unit'''
        self.dipole_unit = self.dipole_box.currentText()

    def __charge_box_changed(self):
        self.charge = self.charge_box.value()


    def __elementsWindow(self):
        self.c = ElementWindow()
        self.c.signal.connect(self.__change_atoms_)
        self.c.show()

    def __change_atoms_(self, atom1, atom2):
        self.atom1_id.setCurrentText(atom1)
        self.atom2_id.setCurrentText(atom2)



    def __PowerSeriesExpansionTab(self):

        '''
        Tab Number 2 - Power Series Expansion

        Used to interpolate the given data using power series expansions for the energy and dipole curves

        Functions:

            Calculate the equilibrium bond length
            Calculate the minimem energy 
            Calcualte the equilibrium rotational constant
            Interpolate the energy curve 
            Calculate the equilibrium vibrational constant
            Calculate errors for the fit
            Plot the energy curve and the interpolated energy curve
            Plot the error for the fit
            Show the error for the fit in a table
            Interpolate the dipole moment curve (if given)
            Calculate the equilibrium dipole moment (if given)
            Calculate the errors for the fit (if given)
            Plot the dipole curve and the interpoalted dipole curve (if given)
            Plot the error for the fit (if given)
            Show the error for the fit in a table (if given)
        '''


        # Initialize some of the variables

        self.pec_order = 8         # Order of power series expansion
        self.dip_order = 5          # Order of dipole moment function
        self.pec_inter_pts = 10000  # Number of interpolation points for potential energy curve
        self.dip_inter_pts = 10000  # Number of interpolation points for dipole moment function
        
        self.energy_coef = ['']     # Power series coefficients for energy
        self.dipole_coef = ['']     # Power series coefficients for dipole


        self.rEq = ''           # Equilibrium bond length
        self.eEq = ''           # Minimum energy
        self.bEq = ''           # Equilibrium rotational constant
        self.wEq = ''           # Equilibrium vibrational constant
        self.dEq = ''           # Equilibrium dipole moment
        self.poly_err = ''      # Error of standard polynomial
        self.inter_err = ''     # Error of power series expansion
        self.dip_err = ''       # Error of dipole moment function

        self.dipole_bool = False

        #
        self.energy_lab = QLabel("Energy")
        self.energy_lab.setFixedWidth(50)

        self.dipole_lab = QLabel("Dipole")
        self.dipole_lab.setFixedWidth(50)
        
        # Select Power Series order
        
        self.pec_order_box = QComboBox()
        self.pec_order_box.addItems(['2', '4', '6', '8', '10', '12', '14', '16'])
        self.pec_order_box.setCurrentText(str(self.pec_order))
        self.pec_order_box.currentIndexChanged.connect(self.__pec_order_box_selected)
        self.pec_order_box.currentIndexChanged.connect(self.__interpolate_data)
        self.pec_order_box.currentIndexChanged.connect(self.__plot_inter_energy)
        self.pec_order_box.currentIndexChanged.connect(self.__plot_inter_energy_err)
        self.pec_order_box.setToolTip("Order of the power series expansion for the energy curve")
        self.pec_order_box.setFixedWidth(100)

        self.dip_order_box = QComboBox()
        self.dip_order_box.addItems(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        self.dip_order_box.setCurrentText(str(self.dip_order))
        self.dip_order_box.currentIndexChanged.connect(self.__dip_order_box_selected)
        self.dip_order_box.currentIndexChanged.connect(self.__interpolate_data)
        self.dip_order_box.currentIndexChanged.connect(self.__plot_inter_dipole)
        self.dip_order_box.currentIndexChanged.connect(self.__plot_inter_dipole_err)
        self.dip_order_box.setToolTip("Order of power series expansion for the dipole moment curve")
        self.dip_order_box.setFixedWidth(100)

        # Number of interpolatino points
        
        self.inter_lab = QLabel("Interpolation Points")
        self.inter_lab.setToolTip("Number of points to use\nto interpolate the\nrespective curves")

        self.pec_inter_val = QLineEdit(str(self.pec_inter_pts))
        self.pec_inter_val.editingFinished.connect(self.__pec_inter_pts_changed)
        self.pec_inter_val.editingFinished.connect(self.__interpolate_data)
        self.pec_inter_val.setFixedWidth(100)

        self.dip_inter_val = QLineEdit(str(self.dip_inter_pts))
        self.dip_inter_val.editingFinished.connect(self.__dip_inter_pts_changed)
        self.dip_inter_val.editingFinished.connect(self.__interpolate_data)
        self.dip_inter_val.setFixedWidth(100)

        # Interpolate Data
        
        self.inter_push = QPushButton("&Interpolate the Energy && Dipole Curves")
        self.inter_push.clicked.connect(self.__interpolate_data)
        self.inter_push.clicked.connect(self.__plot_inter_energy_err)
        self.inter_push.clicked.connect(self.__plot_inter_energy)

        self.inter_shortcut = QShortcut(QKeySequence("Ctrl+I"), self)
        self.inter_shortcut.activated.connect(self.__interpolate_data)
        self.inter_shortcut.activated.connect(self.__plot_inter_energy_err)
        self.inter_shortcut.activated.connect(self.__plot_inter_energy)

        # Power Series Coefficients

        self.coef_lab = QLabel("Power Series Coefficients")
        self.coef_lab.setToolTip("View the power series coefficients")

        self.energy_coef_btn = QPushButton("Energy")
        self.energy_coef_btn.clicked.connect(self.__energy_coefficients)
        self.energy_coef_btn.setFixedWidth(100)

        self.dipole_coef_btn = QPushButton("Dipole")
        self.dipole_coef_btn.clicked.connect(self.__dipole_coefficients)
        self.dipole_coef_btn.setFixedWidth(100)

        # Error of Interpolation

        self.inter_mad_lab = QLabel("Mean Abolute Error")
        self.inter_mad_lab.setToolTip("Mean absolute error for the\nrespective interpolations")

        self.inter_mad_val = QLineEdit('')
        self.inter_mad_val.setReadOnly(True)
        self.inter_mad_val.setFixedWidth(100)

        self.dip_mad_val = QLineEdit('')
        self.dip_mad_val.setReadOnly(True)
        self.dip_mad_val.setFixedWidth(100)

        self.inter_rmse_lab = QLabel("Root Mean Squared Error")
        self.inter_rmse_lab.setToolTip("Root mean squared error for the\nrespective interpolations")

        self.inter_rmse_val = QLineEdit('')
        self.inter_rmse_val.setReadOnly(True)
        self.inter_rmse_val.setFixedWidth(100)

        self.dip_rmse_val = QLineEdit('')
        self.dip_rmse_val.setReadOnly(True)
        self.dip_rmse_val.setFixedWidth(100)

        self.inter_cod_lab = QLabel("Coefficient of Determination")
        self.inter_cod_lab.setToolTip("Coefficient of determination for the\nrespective interpolations")

        self.inter_cod_val = QLineEdit('')
        self.inter_cod_val.setReadOnly(True)
        self.inter_cod_val.setFixedWidth(100)

        self.dip_cod_val = QLineEdit('')
        self.dip_cod_val.setReadOnly(True)
        self.dip_cod_val.setFixedWidth(100)

        # Display Equilibrium Information
        self.rEq_lab = QLabel("Bond Length")
        self.rEq_lab.setToolTip("Equilibrium bond length calculated\nnumerically using the interpolated\nsurface")
        self.rEq_val = QLineEdit(str(self.rEq))
        self.rEq_val.setReadOnly(True)
        self.rEq_val.setFixedWidth(100)

        self.eEq_lab = QLabel("Minimum Energy")
        self.eEq_lab.setToolTip("Energy at the equilibrium bond length")
        self.eEq_val = QLineEdit(str(self.eEq))
        self.eEq_val.setReadOnly(True)
        self.eEq_val.setFixedWidth(100)

        self.bEq_lab = QLabel("Rotational Constant")
        self.bEq_lab.setToolTip("Calculated using the equilibrium bond length\nand the reduced mass")
        self.bEq_val = QLineEdit(str(self.bEq))
        self.bEq_val.setReadOnly(True)
        self.bEq_val.setFixedWidth(100)
    
        self.wEq_lab = QLabel("Vibrational Constant")
        self.wEq_lab.setToolTip("Calculated using the force constant")
        self.wEq_val = QLineEdit(str(self.wEq))
        self.wEq_val.setReadOnly(True)
        self.wEq_val.setFixedWidth(100)

        self.dEq_lab = QLabel("Dipole Moment")
        self.dEq_lab.setToolTip("Calculated using the interpolated\nsurface and the equilibrium\nbond length")
        self.dEq_val = QLineEdit(str(self.dEq))
        self.dEq_val.setReadOnly(True)
        self.dEq_val.setFixedWidth(100)

        # Plot the potential energy curve and errors and show errors
        
        self.plot_inter_data = PlotCurve_111(self)
        self.plot_inter_data_tb = NavigationToolbar2QT(self.plot_inter_data, self)

        self.plot_inter_data_err = PlotCurve_111(self)
        self.plot_inter_data_err_tb = NavigationToolbar2QT(self.plot_inter_data_err, self)

        # Open tables to view error values
        
        self.show_data_lab = QLabel("Display Error Values")
        self.show_data_err = QPushButton("Energy")
        self.show_data_err.clicked.connect(self.__show_energy_err)
        self.show_data_err.setFixedWidth(100)

        self.show_dip_err = QPushButton("Dipole")
        self.show_dip_err.clicked.connect(self.__show_dipole_err)
        self.show_dip_err.setFixedWidth(100)

        # Choose which plot to show

        self.plot_data_lab = QLabel("Plot Interpolated Data")
        self.plot_energy_btn = QPushButton("Energy")
        self.plot_energy_btn.clicked.connect(self.__plot_inter_energy)
        self.plot_energy_btn.clicked.connect(self.__plot_inter_energy_err)

        self.plot_energy_btn.setFixedWidth(100)

        self.plot_dipole_btn = QPushButton("Dipole")
        self.plot_dipole_btn.clicked.connect(self.__plot_inter_dipole)
        self.plot_dipole_btn.clicked.connect(self.__plot_inter_dipole_err)
        self.plot_dipole_btn.setFixedWidth(100)


        # Define the layout of the tab using a grid

        self.dotted_line3 = QLabel("-"*90)

        self.tab2.grid_layout = QGridLayout()

        row=0
        self.tab2.grid_layout.addWidget(self.energy_lab, row, 1, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.dipole_lab, row, 2, alignment=Qt.AlignCenter)

        self.tab2.grid_layout.addWidget(QLabel(""), row, 4, 1, 2, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.plot_inter_data_tb, row, 6, 1, 3, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.plot_inter_data_err_tb, row, 9, 1, 3, alignment=Qt.AlignCenter)

        row+=1
        self.tab2.grid_layout.addWidget(QLabel("Power Series Order"), row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.pec_order_box, row, 1)
        self.tab2.grid_layout.addWidget(self.dip_order_box, row, 2)

        self.tab2.grid_layout.addWidget(QLabel(""), row, 4, 1, 2, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.plot_inter_data, row, 6, 16, 3, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.plot_inter_data_err, row, 9, 16, 3, alignment=Qt.AlignCenter)

        row+=1
        self.tab2.grid_layout.addWidget(self.inter_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.pec_inter_val, row, 1)
        self.tab2.grid_layout.addWidget(self.dip_inter_val, row, 2)

        row+=1 
        self.tab2.grid_layout.addWidget(self.inter_push, row, 0, 1, 3)

        row+=1
        self.tab2.grid_layout.addWidget(QLabel("Power Series Coefficients"), row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.energy_coef_btn, row, 1)
        self.tab2.grid_layout.addWidget(self.dipole_coef_btn, row, 2)

        row+=1
        self.tab2.grid_layout.addWidget(self.inter_mad_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.inter_mad_val, row, 1, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.dip_mad_val, row, 2, alignment=Qt.AlignCenter)
    
        row+=1
        self.tab2.grid_layout.addWidget(self.inter_rmse_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.inter_rmse_val, row, 1, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.dip_rmse_val, row, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab2.grid_layout.addWidget(self.inter_cod_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.inter_cod_val, row, 1, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.dip_cod_val, row, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab2.grid_layout.addWidget(self.show_data_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.show_data_err, row, 1, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.show_dip_err, row, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab2.grid_layout.addWidget(self.plot_data_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.plot_energy_btn, row, 1, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.plot_dipole_btn, row, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab2.grid_layout.addWidget(self.dotted_line3, row, 0, 1, 3)

        row+=1
        self.tab2.grid_layout.addWidget(QLabel("Equilibrium Constants"), row, 0, 1, 3, alignment=Qt.AlignCenter)

        row+=1
        self.tab2.grid_layout.addWidget(self.rEq_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.rEq_val, row, 1)
        self.tab2.grid_layout.addWidget(QLabel("Å"), row, 2)

        row+=1
        self.tab2.grid_layout.addWidget(self.eEq_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.eEq_val, row, 1)
        self.tab2.grid_layout.addWidget(QLabel("Hartree"), row, 2)

        row+=1
        self.tab2.grid_layout.addWidget(self.wEq_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.wEq_val, row, 1)
        self.tab2.grid_layout.addWidget(QLabel("cm^-1"), row, 2)

        row+=1
        self.tab2.grid_layout.addWidget(self.bEq_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.bEq_val, row, 1)
        self.tab2.grid_layout.addWidget(QLabel("cm^-1"), row, 2)

        row+=1
        self.tab2.grid_layout.addWidget(self.dEq_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab2.grid_layout.addWidget(self.dEq_val, row, 1)
        self.tab2.grid_layout.addWidget(QLabel("Debye"), row, 2)

        row+=1
        self.spacer_bottom2 = QSpacerItem(10, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.tab2.grid_layout.addItem(self.spacer_bottom2, row, 0)

        
        self.tab2.setLayout(self.tab2.grid_layout)

    def __pec_order_box_selected(self):
        '''Used to assign the order of power series expansion for the energy curve'''
        self.pec_order = self.pec_order_box.currentText()

    def __dip_order_box_selected(self):
        '''Used to assign the order of power series expansion for the dipole curve'''
        self.dip_order = self.dip_order_box.currentText()

    def __pec_inter_pts_changed(self):
        '''Used to assign the number of points used for the interpolation of the energy curve'''
        try:
            self.pec_inter_pts = int(self.pec_inter_val.text())
        except:
            self.errorText = "Number of interpolation points must be an interger\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()
            self.pec_inter_val.setText(str(self.pec_inter_pts))

    def __dip_inter_pts_changed(self):
        '''Used to assign the number of points used for the interpolation of the dipole curve'''
        try:
            self.dip_inter_pts = int(self.dip_inter_val.text())
        except:
            self.errorText = "Number of interpolation points must be an interger\n\n" +  str(traceback.format_exc())
            self.__openErrorMessage()
            self.dip_inter_val.setText(str(self.dip_inter_pts))

    def __interpolate_data(self):
        '''Interpolate the energy curve and the dipole curve (if given)'''

        self.dEq_val.setText('')
        self.dip_mad_val.setText('')
        self.dip_rmse_val.setText('')
        self.dip_cod_val.setText('')
        self.rEq_val.setText('')
        self.eEq_val.setText('')
        self.bEq_val.setText('')
        self.wEq_val.setText('')
        self.inter_mad_val.setText('')
        self.inter_cod_val.setText('')
        self.inter_rmse_val.setText('')

        
        def convert_units():
            '''Convert the energy and dipole curve to hatrees, Angstrom, and Debye'''
            if self.energy_unit.lower() == 'hartrees':
                self.temp_data[1] /= 1
            elif self.energy_unit.lower() == 'kcal/mol':
                self.temp_data[1] /= hart_kcal
            elif self.energy_unit.lower() == 'kj/mol':
                self.temp_data[1] /= (hart_kcal * kcal_kj)
            elif self.energy_unit.lower() == 'ev':
                self.temp_data[1] /= hart_eV
            elif self.energy_unit.lower() == 'j':
                self.temp_data[1] /= hart_J
            elif self.energy_unit.lower() == 'wavenumbers':
                self.temp_data[1] /= hart_cm

            if self.length_unit.lower() == 'Å':
                self.temp_data[0] /= 1
            elif self.length_unit.lower() == 'm':
                self.temp_data[0] /= ang_m
            elif self.length_unit.lower() == 'bohr':
                self.temp_data[0] /= ang_bohr

            if self.temp_data.shape[0] == 3:
                if self.dipole_unit.lower() == 'd':
                    self.temp_data[2] /= 1
                elif self.dipole_unit.lower() == 'au':
                    self.temp_data[2] /= D_au

        try:
            self.temp_data = self.data.copy()

            convert_units()

            idx = self.temp_data[0].argsort()
            self.temp_data = self.temp_data[:,idx]
            

            if self.temp_data.shape[1] < int(self.pec_order) or self.temp_data.shape[1] < int(self.dip_order):
                self.errorText = 'Order of power series must be less than the number of data points '
                self.__openErrorMessage()
                return
            

            inter = Interpolate(temp_data = self.temp_data,
                                atoms     = [self.atom1, self.atom2],
                                isotopes  = [self.iso1, self.iso2],
                                masses    = [self.mass1, self.mass2],
                                charge    = self.charge,
                                numpoints = [self.pec_inter_pts, self.dip_inter_pts],
                                order_e   = int(self.pec_order),
                                order_d   = int(self.dip_order)
                                )

            if self.data.shape[0] == 3:
                '''Dipole data provided'''
                self.dEq         = inter.dEq        # Equilibrium dipole moment value
                self.PEC_d       = inter.PEC_d      # Actual dipole moment values
                self.PEC_d_      = inter.PEC_d_     # Interpolated dipole moment values
                self.dip_mad     = inter.D_mad      # Mean Absolute Error
                self.dip_cod     = inter.D_cod      # Coefficient of Determination
                self.dip_rmse    = inter.D_rmse     # Root Mean Squared Error
                self.dip_err_arr = inter.polyDerr_arr
                self.dipole_coef = inter.polyDfit   # Coefficients for dipole moment function
                self.dipole_bool = True
                
                self.dEq_val.setText(str(round(self.dEq/D_au, 8)))
                self.dip_mad_val.setText(str(round(self.dip_mad/D_au, 8)))
                self.dip_rmse_val.setText(str(round(self.dip_rmse/D_au, 8)))
                self.dip_cod_val.setText(str(round(self.dip_cod, 8)))
                
                self.plot_dipole_btn.setEnabled(True)
                self.show_dip_err.setEnabled(True)
                self.dipole_coef_btn.setEnabled(True)

            else:
                '''No dipole data provided'''
                self.dEq = 0
                self.PEC_d = np.zeros((1))
                self.PEC_d_ = np.zeros((1))
                self.dip_mad = ''
                self.dip_cod = ''
                self.dip_rmse = ''
                self.dip_err_arr = ['']
                self.dipole_coef = ['']
                self.dipole_bool = False

                self.dEq_val.setText("")

                self.plot_dipole_btn.setEnabled(False)
                self.show_dip_err.setEnabled(False)
                self.dipole_coef_btn.setEnabled(False)

            self.poly_err = inter.polyerr   # Error of standard polynomial fit

            self.rEq = inter.rEq            # Equilibrium Bond Length
            self.eEq = inter.eEq            # Minimum Energy
            self.bEq = inter.bEq            # Equilibrium Rotational Constant
            self.wEq = inter.omega          # Equilbrium Vibrational Constant

            self.energy_coef = inter.Coef         # Power Series Expansion Coefficients
            self.PEC_r       = inter.R_           # Actual R-values for power series
            self.PEC_e       = inter.E_           # Actual Energy values for power series
            self.PEC_r_      = inter.PEC_r_       # Interpolated R-values for power series 
            self.PEC_e_      = inter.PEC_e_       # Interpolated Energy values for power series

            self.inter_err_arr = inter.error     # Error of power series interpolation
            self.inter_err     = sum(abs(self.inter_err_arr)) / self.inter_err_arr.size

            self.reduced_mass = inter.reducedMass
            self.beta         = (((self.reduced_mass * inter.k) ** 0.25) / (h_bar**0.5)) * ang_m
            self.nu           = inter.nu

            self.rEq_val.setText(str(round(self.rEq, 8)))
            self.eEq_val.setText(str(round(self.eEq, 8)))
            self.bEq_val.setText(str(round(self.bEq, 8)))
            self.wEq_val.setText(str(round(self.wEq, 8)))
            self.inter_mad_val.setText(str(round(inter.inter_mad, 8)))
            self.inter_cod_val.setText(str(round(inter.inter_cod, 8)))
            self.inter_rmse_val.setText(str(round(inter.inter_rmse, 8)))


        except:
            self.errorText = 'No data file found\n\n' + str(traceback.format_exc())
            self.__openErrorMessage()

    def __energy_coefficients(self):
        '''Open an external table to view the power series expansion coefficients for the energy curve'''
        try:
            self.coef_win.close()
        except:
            pass
        if self.energy_coef[0] != '':
            self.coef_win = CoefficientWindow(self.energy_coef, 'energy')
            self.coef_win.show()
        else:
            self.errorText = "No coeffecients found"
            self.__openErrorMessage()

    def __dipole_coefficients(self):
        '''Open an external table to view the power series expansion coefficients for the dipole curve'''
        try:
            self.coef_win.close()
        except:
            pass
        if self.dipole_coef[0] != '':
            self.coef_win = CoefficientWindow(self.dipole_coef[::-1], 'dipole')
            self.coef_win.show()
        else:
            self.errorText = "No coefficients found"
            self.__openErrorMessage()

    def __plot_inter_energy(self):
        '''Plot the interpolated and actual energy curves'''
        try:
            self.plot_inter_data.axes.cla()
            self.plot_inter_data.axes.scatter(self.PEC_r, self.PEC_e, marker='x', c='b', label='EST Data')
            self.plot_inter_data.axes.plot(self.PEC_r_, self.PEC_e_, marker=',', markersize=1, c='r', label='Fit Data')
            self.plot_inter_data.axes.set_xlabel("Bond Displacement (Å)")
            self.plot_inter_data.axes.set_ylabel("Energy (E$_{H}$)")
            self.plot_inter_data.axes.grid()
            self.plot_inter_data.axes.legend()
            self.plot_inter_data.draw()
        except AttributeError:
            self.errorText = "File not yet loaded\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()

    def __plot_inter_dipole(self):
        '''Plot the interpolated and actual dipole curves'''
        if self.dipole_bool == True:
            try:
                self.plot_inter_data.axes.cla()
                self.plot_inter_data.axes.scatter(self.PEC_r, self.PEC_d, marker='x', c='b', label='EST Data')
                self.plot_inter_data.axes.plot(self.PEC_r_, self.PEC_d_, marker=',', markersize=1, c='r', label='Fit Data')
                self.plot_inter_data.axes.set_xlabel("Bond Displacement (Å)")
                self.plot_inter_data.axes.set_ylabel("Energy (D)")
                self.plot_inter_data.axes.grid()
                self.plot_inter_data.axes.legend()
                self.plot_inter_data.draw()
            except AttributeError:
                self.errorText = "File not yet loaded\n\n" + str(traceback.format_exc())
                self.__openErrorMessage()


    def __plot_inter_energy_err(self):
        '''Plot the error between the interpolated and actual energy curves'''
        try:
            self.plot_inter_data_err.axes.cla()
            self.plot_inter_data_err.axes.scatter(self.PEC_r, self.inter_err_arr, marker='x', c='r')
            self.plot_inter_data_err.axes.set_xlabel("Bond Displacement (Å)")
            self.plot_inter_data_err.axes.set_ylabel("Erorr (E$_{H}$)")
            self.plot_inter_data_err.axes.grid()
            self.plot_inter_data_err.axes.set_ylim(min(self.inter_err_arr)*1.1, max(self.inter_err_arr)*1.1)
            self.plot_inter_data_err.draw()

        except AttributeError:
            self.errorText = "Attempte to plot data failed"
            self.__openErrorMessage()

    def __plot_inter_dipole_err(self):
        '''Plot the error between the interpolated and actual dipole curves'''
        if self.dipole_bool == True:
            try:
                self.plot_inter_data_err.axes.cla()
                self.plot_inter_data_err.axes.scatter(self.PEC_r, self.dip_err_arr, marker='x', c='r')
                self.plot_inter_data_err.axes.set_xlabel("Bond Displacement (Å)")
                self.plot_inter_data_err.axes.set_ylabel("Erorr (D)")
                self.plot_inter_data_err.axes.grid()
                self.plot_inter_data_err.axes.set_ylim(min(self.dip_err_arr)*1.1, max(self.dip_err_arr)*1.1)
                self.plot_inter_data_err.draw()

            except AttributeError:
                self.errorText = "Attempte to plot data failed"
                self.__openErrorMessage()

    def __show_energy_err(self):
        '''Open an external table to show the error between the interpolated and acutal energy curves'''
        try:
            self.errval_win.close()
        except:
            pass
        try:
            self.errval_win = InterpolationErrorWindow(self.PEC_r, self.inter_err_arr, 'energy')
            self.errval_win.show()
        except:
            self.errorText = "No energy error values found"
            self.__openErrorMessage()

    def __show_dipole_err(self):
        '''Open an external table to show the error between the interpolated and acutal dipole curves'''
        if self.dipole_bool == False:
            self.errorText = "No dipole data given"
            self.__openErrorMessage()
        else:
            try:
                self.errval_win.close()
            except:
                pass
            try:
                self.errval_win = InterpolationErrorWindow(self.PEC_r, self.dip_err_arr, 'dipole')
                self.errval_win.show()
            except Exception as e:
                self.errorText = str(traceback.format_exc())
                self.__openErrorMessage()




    def __VibrationalHamiltonianTab(self):

        '''
            Tab Number 3 - Vibrational Hamiltonian

        Used to generate the vibrational hamilatonians

        Functions:

            Define the maximum vibrational quantum number
            Define the maximum rotational quantum number
            Generate the Harmonic matrix
            Generate the Anharmonic matrix
            Generate the Centrifugal matrix
            Generate the Transition Dipole Moment matrix
            Generate all matrices
            View each of the five matrices
            Save each of the five matrcies
            Check the stability of the total matrix
            Calculate the highest converged state of the total matrix
            View the eigenvectors
            View the eigenvector contributions
            Refresh the eigenvalues

        '''

        # Initialize some of the variables

        self.maxV = 25          # Maximum vibrational quantum number
        self.maxJ = 10          # Maximum rotational quantum number

        self.harmonic    = np.zeros((self.maxV+1, self.maxV+1))                 # Initial harmonic matrix
        self.anharmonic  = np.zeros((self.maxV+1, self.maxV+1))                 # Initial anharmonic matrix
        self.centrifugal = np.zeros((self.maxJ+1, self.maxV+1, self.maxV+1))    # Initial centriffugal matrix
        self.tdm         = np.zeros((self.maxV+1, self.maxV+1))                 # Initial transition dipole moment (tdm) matrix
        
        self.total = self.harmonic + self.anharmonic + self.centrifugal         # Initial total matrix

        self.trunc_err_val = 0.01                                       # Maximum error for truncation
        self.max_trunc_val = self.maxV                                  # Maximum converged qauntum state
        self.trunc_err_arr = np.ones((self.max_trunc_val))*self.maxV    # Array of converged quantum states


        # Change values for quantum numbers

        self.maxV_lab = QLabel("Maximum ν value")
        self.maxV_str = QLineEdit(str(self.maxV))
        self.maxV_str.editingFinished.connect(self.__maxV_str_changed)
        self.maxV_str.editingFinished.connect(self.__vib_spec_changed)
        self.maxV_str.setFixedWidth(100)

        self.maxJ_lab = QLabel("Maximum J value")
        self.maxJ_str = QLineEdit(str(self.maxJ))
        self.maxJ_str.editingFinished.connect(self.__maxJ_str_changed)
        self.maxJ_str.editingFinished.connect(self.__rot_spec_changed)
        self.maxJ_str.setFixedWidth(100)        
    
        # Generate Hamiltonian Matrices

        self.harm_lab   = QLabel("Harmonic Matrix")
        self.anharm_lab = QLabel("Anarhmonic Matrix")
        self.cent_lab   = QLabel("Centrifugal Matrix")
        self.tdm_lab    = QLabel("Transition Dipole Moment Matrix")
        self.all_lab    = QLabel("All Matrices")

        self.harm_btn   = QPushButton("Generate")
        self.anharm_btn = QPushButton("Generate")
        self.cent_btn   = QPushButton("Generate")
        self.tdm_btn    = QPushButton("Generate")
        self.all_btn    = QPushButton("&Generate")

        self.harm_btn.setFixedWidth(100)
        self.anharm_btn.setFixedWidth(100)
        self.cent_btn.setFixedWidth(100)
        self.tdm_btn.setFixedWidth(100)
        self.all_btn.setFixedWidth(100)

        self.harm_btn.clicked.connect(self.__generate_harm_matrix)
        self.anharm_btn.clicked.connect(self.__generate_anharm_matrix)
        self.cent_btn.clicked.connect(self.__generate_cent_matrix)
        self.tdm_btn.clicked.connect(self.__generate_tdm_matrix)
        self.all_btn.clicked.connect(self.__generate_all_matrix)
        self.all_btn.clicked.connect(self.__show_eigenvalue_table)

        self.all_btn_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        self.all_btn_shortcut.activated.connect(self.__generate_all_matrix)
        self.all_btn_shortcut.activated.connect(self.__show_eigenvalue_table)

        self.harm_view_btn   = QPushButton("View")
        self.anharm_view_btn = QPushButton("View")
        self.cent_view_btn   = QPushButton("View")
        self.tdm_view_btn    = QPushButton("View")
        self.all_view_btn    = QPushButton("View")

        self.harm_view_btn.setFixedWidth(100)
        self.anharm_view_btn.setFixedWidth(100)
        self.cent_view_btn.setFixedWidth(100)
        self.tdm_view_btn.setFixedWidth(100)
        self.all_view_btn.setFixedWidth(100)

        self.harm_view_btn.clicked.connect(self.__view_harm_matrix)
        self.anharm_view_btn.clicked.connect(self.__view_anharm_matrix)
        self.cent_view_btn.clicked.connect(self.__view_cent_matrix)
        self.tdm_view_btn.clicked.connect(self.__view_tdm_matrix)
        self.all_view_btn.clicked.connect(self.__view_all_matrix)

        self.harm_save_btn   = QPushButton("Save")
        self.anharm_save_btn = QPushButton("Save")
        self.cent_save_btn   = QPushButton("Save")
        self.tdm_save_btn    = QPushButton("Save")
        self.all_save_btn    = QPushButton("Save")
    
        self.harm_save_btn.setFixedWidth(100)
        self.anharm_save_btn.setFixedWidth(100)
        self.cent_save_btn.setFixedWidth(100)
        self.tdm_save_btn.setFixedWidth(100)
        self.all_save_btn.setFixedWidth(100)


        self.harm_save_btn.clicked.connect(self.__save_harm_matrix)
        self.anharm_save_btn.clicked.connect(self.__save_anharm_matrix)
        self.cent_save_btn.clicked.connect(self.__save_cent_matrix)
        self.tdm_save_btn.clicked.connect(self.__save_tdm_matrix)
        self.all_save_btn.clicked.connect(self.__save_all_matrix)

        self.harm_btn.setFixedWidth(100)
        self.anharm_btn.setFixedWidth(100)
        self.cent_btn.setFixedWidth(100)
        self.tdm_btn.setFixedWidth(100)
        self.all_btn.setFixedWidth(100)

        # Check stability of matrix

        self.check_stabil_btn = QPushButton("Check Stability of Total Matrix")
        self.check_stabil_btn.clicked.connect(self.__check_matrix_stability)
        
        # Truncation of Matrix 

        self.max_trunc_error_lab = QLabel("Maximum Truncation Error")
        self.max_trunc_error_lab.setToolTip("This error is based on changes\nin the eigenvalues when\na smaller matrix is used")

        self.max_trunc_error_val = QLineEdit(str(self.trunc_err_val))
        self.max_trunc_error_val.setFixedWidth(100)
        self.max_trunc_error_val.editingFinished.connect(self.__max_trunc_error_changed)

        self.max_trunc_lab = QLabel("Calculate Highest Converged State?")
        self.max_trunc_yes_btn = QPushButton("Yes")
        self.max_trunc_yes_btn.clicked.connect(self.__truncate_matrix)

        # View EigenVectors

        self.view_eigen_btn = QPushButton("View Eigenvectors")
        self.view_eigen_btn.clicked.connect(self.__show_eigenvectors)

        # View Contributtions

        self.view_contribution_btn = QPushButton("View Contributions")
        self.view_contribution_btn.clicked.connect(self.__show_contributions)
        self.view_contribution_btn.setToolTip("Square of the eigenvectors")

        # Eigenvalue table

        self.eigenvalue_table = QTableWidget()
        self.eigenvalue_table.installEventFilter(self)
        self.refresh_eigenvalue_table = QPushButton("Refresh Eigenvalues")
        self.refresh_eigenvalue_table.clicked.connect(self.__show_eigenvalue_table)

        # Define the layout of the tab using a grid
    
        self.dotted_line4 = QLabel("-"*120)
        self.dotted_line5 = QLabel("-"*120)
        self.dotted_line6 = QLabel("-"*120)
        self.dotted_line7 = QLabel("-"*120)

        self.tab3.grid_layout = QGridLayout()

        row=0
        self.tab3.grid_layout.addWidget(self.maxV_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.maxV_str, row, 1)

        self.tab3.grid_layout.addWidget(QLabel("Eigenvalues"), row, 4, 1, 4, alignment=Qt.AlignCenter)

        row+=1
        self.tab3.grid_layout.addWidget(self.maxJ_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.maxJ_str, row, 1)
        self.tab3.grid_layout.addWidget(self.eigenvalue_table, row, 4, 16, 4)

        row+=1
        self.tab3.grid_layout.addWidget(self.dotted_line4, row, 0, 1, 4)
        
        row+=1
        self.tab3.grid_layout.addWidget(self.harm_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.harm_btn, row, 1)
        self.tab3.grid_layout.addWidget(self.harm_view_btn, row, 2)
        self.tab3.grid_layout.addWidget(self.harm_save_btn, row, 3)


        row+=1
        self.tab3.grid_layout.addWidget(self.anharm_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.anharm_btn, row, 1)
        self.tab3.grid_layout.addWidget(self.anharm_view_btn, row, 2)
        self.tab3.grid_layout.addWidget(self.anharm_save_btn, row, 3)

        row+=1
        self.tab3.grid_layout.addWidget(self.cent_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.cent_btn, row, 1)
        self.tab3.grid_layout.addWidget(self.cent_view_btn, row, 2)
        self.tab3.grid_layout.addWidget(self.cent_save_btn, row, 3)

        row+=1
        self.tab3.grid_layout.addWidget(self.tdm_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.tdm_btn, row, 1)
        self.tab3.grid_layout.addWidget(self.tdm_view_btn, row, 2)
        self.tab3.grid_layout.addWidget(self.tdm_save_btn, row, 3)

        row+=1
        self.tab3.grid_layout.addWidget(self.all_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.all_btn, row, 1)
        self.tab3.grid_layout.addWidget(self.all_view_btn, row, 2)
        self.tab3.grid_layout.addWidget(self.all_save_btn, row, 3)

        row+=1
        self.tab3.grid_layout.addWidget(self.dotted_line5, row, 0, 1, 4)
        
        row+=1
        self.tab3.grid_layout.addWidget(self.check_stabil_btn, row, 0, 1, 4)

        row+=1
        self.tab3.grid_layout.addWidget(self.dotted_line6, row, 0, 1, 4)
        
        row+=1
        self.tab3.grid_layout.addWidget(self.max_trunc_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.max_trunc_yes_btn, row, 1)

        row+=1
        self.tab3.grid_layout.addWidget(self.max_trunc_error_lab, row, 0, alignment=Qt.AlignCenter)
        self.tab3.grid_layout.addWidget(self.max_trunc_error_val, row, 1)

        row+=1
        self.tab3.grid_layout.addWidget(self.dotted_line7, row, 0, 1, 4)

        row+=1
        self.tab3.grid_layout.addWidget(self.view_eigen_btn, row, 0, 1, 4)

        row+=1
        self.tab3.grid_layout.addWidget(self.view_contribution_btn, row, 0, 1, 4)

        row+=1
        self.tab3.grid_layout.addWidget(self.refresh_eigenvalue_table, row, 0, 1, 4)

        row+=1
        self.spacer_bottom3 = QSpacerItem(10, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.tab3.grid_layout.addItem(self.spacer_bottom3, row, 4)


        self.tab3.setLayout(self.tab3.grid_layout)


    def __maxV_str_changed(self):
        '''
        Used to change the maximum vibrational quantum number. 
        Cascades to change values in other tabs:
            Limits for plotting the simulated spectra
        '''
        try:
            self.maxV = int(self.maxV_str.text())
            self.harmonic    = np.zeros((self.maxV+1, self.maxV+1))
            self.anharmonic  = np.zeros((self.maxV+1, self.maxV+1))
            self.centrifugal = np.zeros((self.maxJ+1, self.maxV+1, self.maxV+1))
            self.tdm         = np.zeros((self.maxV+1, self.maxV+1))
            
            self.max_trunc_val = self.maxV
            self.vib_plot.axes.cla()

            self.__change_vib_limits()
            self.__change_rot_limits()
            self.__change_rov_limits()

        except:
            self.errorText = "Maximum ν value must be an integer"
            self.__openErrorMessage()
            self.maxV_str.setText(str(self.maxV))

    def __maxJ_str_changed(self):
        '''
        Used to change the maximum vibrational quantum number. 
        Cascades to change values in other tabs:
            Limits for plotting the simulated spectra
            Limits for the order of the spectroscopic constants
            Limits for the order of the plotted wavefunctions
        '''
        try:
            if int(self.maxJ_str.text()) > -1:
                self.maxJ = int(self.maxJ_str.text())

                if self.maxJ != 0:

                    gen_hamil = Hamil(ID = 'cent',
                                  cent = self.centrifugal,
                                  maxJ = self.maxJ,
                                  maxV = self.maxV+1,
                                  rEq  = self.rEq,
                                  beta = self.beta,
                                  reduced_mass = self.reduced_mass,
                                  Trap = 2000
                                  )
                    self.centrifugal = gen_hamil.centrifugal
                    self.total = self.harmonic + self.anharmonic + self.centrifugal

                    self.rot_spec_order_box.setRange(0, min(int(self.maxJ)-1, 8))
                    self.__vib_spec_changed()
            
                    self.view_wfs_j_box.setRange(0, self.maxJ)
                    self.tps_box.setRange(0, self.maxJ)
                
                    self.__change_vib_limits()
                    self.__change_rot_limits()
                    self.__change_rov_limits()

                else:
                    self.rot_spec_order_box.setRange(0, min(int(self.maxJ), 8))
                    self.__vib_spec_changed()
                    self.__rot_spec_changed()
                    self.__rovib_spec_changed()

                    self.rot_plot.axes.cla()

                    self.view_wfs_j_box.setRange(0, self.maxJ)
                    self.tps_box.setRange(0, self.maxJ)
                
            else:
                self.errorText = "Maximum J value must be a positive integer\n\n" + str(traceback.format_exc())
                self.__openErrorMessage()
                self.maxJ_str.setText(str(self.maxJ))
        except:
            self.errorText = "Maximum J value must be an positive integer\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()
            self.maxJ_str.setText(str(self.maxJ))
    
    def __generate_harm_matrix(self):
        '''Generate the harmonic matrix'''
        try:
            gen_hamil = Hamil(ID   = 'harm',
                              maxv = self.maxV+1,
                              nu   = self.nu                              
                              )
            self.harmonic = gen_hamil.harmonic
            self.total = self.harmonic + self.anharmonic + self.centrifugal
            
            self.__show_eigenvalue_table()

        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __generate_anharm_matrix(self):
        '''Generate the anharmonic matrix'''
        try:
            gen_hamil = Hamil(ID = 'anharm',
                              maxV = self.maxV+1,
                              coef = self.energy_coef, 
                              beta = self.beta
                              )
            self.anharmonic = gen_hamil.anharmonic
            self.total = self.harmonic + self.anharmonic + self.centrifugal
            
            self.__show_eigenvalue_table()

        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __generate_cent_matrix(self):
        '''Generate the centrifugal matrix'''
        try:
            gen_hamil = Hamil(ID = 'cent',
                              cent = self.centrifugal,
                              maxJ = self.maxJ,
                              maxV = self.maxV+1, 
                              rEq  = self.rEq, 
                              beta = self.beta, 
                              reduced_mass = self.reduced_mass,
                              Trap = 2000
                              )
            self.centrifugal = gen_hamil.centrifugal
            self.total = self.harmonic + self.anharmonic + self.centrifugal
            
            self.__show_eigenvalue_table()

        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __generate_tdm_matrix(self):
        '''Generate the transition dipole moment matrix'''
        try:
            gen_hamil = Hamil(ID = 'tdm',
                              maxV = self.maxV+1,
                              coef = self.dipole_coef,
                              beta = self.beta
                              )
            self.tdm = gen_hamil.tdm
        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __generate_all_matrix(self):
        '''Generate the harmonic, anharmonic, centrifugal, and transition dipole moment matrices'''
        try:
            self.__interpolate_data()

            gen_harm = Hamil(ID   = 'harm',
                             maxv = self.maxV+1,
                             nu   = self.nu
                             )
            self.harmonic = gen_harm.harmonic
            
            gen_anharm = Hamil(ID = 'anharm',
                               maxV = self.maxV+1,
                               coef = self.energy_coef, 
                               beta = self.beta
                               )
            self.anharmonic = gen_anharm.anharmonic

            gen_cent = Hamil(ID = 'cent',
                             cent = self.centrifugal,
                             maxJ = self.maxJ,
                             maxV = self.maxV+1, 
                             rEq  = self.rEq, 
                             beta = self.beta, 
                             reduced_mass = self.reduced_mass,
                             Trap = 2000
                             )
            self.centrifugal = gen_cent.centrifugal
            self.total = self.harmonic + self.anharmonic + self.centrifugal

            if self.dipole_bool == True:
                gen_tdm = Hamil(ID = 'tdm',
                                maxV = self.maxV+1,
                                coef = self.dipole_coef,
                                beta = self.beta
                                )
                self.tdm = gen_tdm.tdm

        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __view_harm_matrix(self):
        '''Open an external table to view the constructed harmonic matrix'''
        try:
            self.view.close()
        except:
            pass
        self.view = MatrixWindow(matrix=self.harmonic*J_cm, 
                                 val='Harmonic Matrix')
        self.view.show()
    def __view_anharm_matrix(self):
        '''Open an external table to view the constructed anharmonic matrix'''
        try:
            self.view.close()
        except:
            pass
        self.view = MatrixWindow(matrix=self.anharmonic*J_cm, 
                                 val='ANharmonic Matrix')
        self.view.show()

    def __view_cent_matrix(self):
        '''Open an external table to view the constructed centrifugal matrix'''
        try:
            self.view.close()
        except:
            pass
        if self.maxJ != 0:
            self.view = MatrixWindow(matrix=self.centrifugal*J_cm/(2.), 
                                     val='Centrifugal Distortion Matrix', 
                                     J=self.maxJ+1)
        else:
            self.view = MatrixWindow(matrix=np.zeros((self.maxV, self.maxV)),
                                     val='Centrifugal Distortion Matrix',
                                     J=self.maxJ)

    def __view_tdm_matrix(self):
        '''Open an external table to view the constructed transition dipole moment matrix'''
        try:
            self.view.close()
        except:
            pass

        self.view = MatrixWindow(matrix=self.tdm*au_D, 
                                 val='Transition Dipole Moment Matrix')
        self.view.show()

    def __view_all_matrix(self):
        '''Open an external table to view the constructed total matrix'''
        try:
            self.view.close()
        except:
            pass
        if self.maxJ != 0:
            self.view = MatrixWindow(matrix=self.total*J_cm, 
                                     val='Total RoVibrational Matrix', 
                                     J=self.maxJ+1)
        else:
            self.view = MatrixWindow(matrix=self.total[0]*J_cm,
                                     val='Total RoVibrational Matrix')

    def __save_harm_matrix(self):
        '''Save the constructed harmonic matrix to a file'''
        try:
            self.save_file_window.close()
        except:
            pass
        np.save("Harmonic_Matrix", self.harmonic)
        message="Matrix saved as 'Harmonic_Matrix.npy'\nTo unpack, use 'np.load('Harmonic_Matrix.npy')'"
        self.save_file_window = SaveWindow(message)

    def __save_anharm_matrix(self):
        '''Save the constructed anharmonic matrix to a file'''
        try:
            self.save_file_window.close()
        except:
            pass
        np.save("Anharmonic_Matrix", self.anharmonic)
        message="Matrix saved as 'Anharmonic_Matrix.npy'\nTo unpack, use 'np.load('Anharmonic_Matrix.npy')'"
        self.save_file_window = SaveWindow(message)

    def __save_cent_matrix(self):
        '''Save the constructed centrifugal matrix to a file'''
        try:
            self.save_file_window.close()
        except:
            pass
        np.save("Centrifugal_Matrix", self.centrifugal)
        message="Tensor saved as 'Centrifugal_Matrix.npy'\nTo unpack, use 'np.load('Centrifugal_Matrix.npy')'"
        self.save_file_window = SaveWindow(message)

    def __save_tdm_matrix(self):
        '''Save the constructed transition dipole moment matrix to a file'''
        try:
            self.save_file_window.close()
        except:
            pass
        np.save("TransitionDipole_Matrix", self.tdm)
        message="Matrix saved as 'TransitionDipole_Matrix.npy'\nTo unpack, use 'np.load('TransitionDipole_Matrix.npy')'"
        self.save_file_window = SaveWindow(message)

    def __save_all_matrix(self):
        '''Save the constructed total matrix to a file'''
        try:
            self.save_file_window.close()
        except:
            pass
        np.save("Total_Matrix", self.total)
        message="Tensor saved as 'Total_Matrix.npy'\nTo unpack, use 'np.load('Total_Matrix.npy')'"
        self.save_file_window = SaveWindow(message)

    def __max_trunc_error_changed(self):
        '''Activated to change the value for the maximum truncation error'''
        self.trunc_err_val = float(self.max_trunc_error_val.text())
    
    def __truncate_matrix(self):
        '''Used to calculate the maximum vibrational states given a trunction error'''
        try:
            total_val,  total_vec  = self.__diagonalize(self.total)
            total_val_, total_vec_ = self.__diagonalize(self.total[:,:-1,:-1])
            total_val_diff         = abs(total_val[:,:-1] - total_val_)

            diff = np.zeros((total_val_diff.shape[0]), dtype=int) + self.maxV
            for j in range(total_val_diff.shape[0]):
                if np.sum(abs(total_val_diff[j])) == 0.0:
                    diff[j] = len(total_val_diff[j])
                else:
                    diff_id = np.where(total_val_diff[j] < self.trunc_err_val)[0]
                    if len(diff_id) > 0:
                        diff[j] = diff_id.copy()[-1]
            
            self.trunc_err_arr = diff
            self.max_trunc_val = diff[0]
            self.__rot_spec_changed()

            self.vib_sim_spec_v_val.setRange(0, self.max_trunc_val)
            self.vib_sim_spec_v_val.setValue(self.max_trunc_val)
            self.rot_sim_spec_v_val.setRange(-1, self.max_trunc_val)
            self.rot_sim_spec_v_val.setValue(0)
            self.rov_sim_spec_v_val.setRange(0, self.max_trunc_val)
            self.rov_sim_spec_v_val.setValue(self.max_trunc_val)

            self.view_wfs_v_box.setRange(0, self.max_trunc_val-1)

            self.c = TruncationWindow(self.trunc_err_arr-1, total_val_diff)
            self.c.show()

        except AttributeError:
            self.errorText = "No eigenvalues found\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()

    def __check_matrix_stability(self):
        '''Check the stability of the total matrix for negative eigenvalues'''
        try:
            self.vals, self.vects = self.__diagonalize(self.total)
            n = 0
            while self.vals[0,0] < 0:
                n += 1
                self.vals, self.vects = self.__diagonalize(self.total[:,:-n,:-n])
            
            self.trunc = n

            if self.trunc > 0:
                self.trunc_win = StabilityWindow(self.total.shape[1] - self.trunc-1, 0)
                self.trunc_win.show()

                self.trunc = n
                self.harmonic = self.harmonic[:-n, :-n]
                self.total = self.total[:,:-n,:-n]

            else:
                self.trunc_win = StabilityWindow(self.total.shape[1] - self.trunc-1, 1)
                self.trunc_win.show()

        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __show_eigenvalue_table(self):
        '''Update the internal table to display the eigenvalues'''
        try:
            self.harm_val, self.harm_vec = self.__diagonalize(self.harmonic)
            self.total_val, self.total_vec = self.__diagonalize(self.total)

            self.df = pd.DataFrame({'ν' : np.arange(self.harm_val.shape[0]),
                                    'Harmonic' : self.harm_val
                                    })

            for j in range(self.total_val.shape[0]):
                self.df['Anharmonic (J=' + str(j) + ')'] = self.total_val[j]

            temp_arr = self.df.to_numpy()

            self.df = self.df.round(decimals=7)

            self.eigenvalue_table.setColumnCount(temp_arr.shape[1])
            self.eigenvalue_table.rowCount()
            self.eigenvalue_table.setHorizontalHeaderLabels(self.df.columns)

            while self.eigenvalue_table.rowCount() > 0:
                self.eigenvalue_table.removeRow(0)

            for i in range(self.harm_val.shape[0]):
                self.__add_eigenvalue_table_row([*temp_arr[i]])

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __add_eigenvalue_table_row(self, row_data):
        '''Used to add a new row to the eigenvalue table'''
        row = self.eigenvalue_table.rowCount()
        self.eigenvalue_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            if item == row_data[0]:
                cell = QTableWidgetItem(str(int(item)))
            else:
                cell = QTableWidgetItem(str(item))
            self.eigenvalue_table.setItem(row, col, cell)
            col += 1

    def __show_eigenvectors(self):
        '''Used to open an external table to view the eigenvectors'''
        self.total_val, self.total_vec = self.__diagonalize(self.total)
        try:
            self.view.close()
        except:
            pass
        if self.maxJ != 0:
            self.view = MatrixWindow(matrix=self.total_vec,
                                     val='Eigenvectors Matrix',
                                     J=self.maxJ+1)
        else:
            self.view = MatrixWindow(matrix=self.total_vec[0],
                                     val='Eigenvectors Matrix')
    
    def __show_contributions(self):
        '''Used to open an external table to view the eigenvector contributions'''
        self.total_val, self.total_vec = self.__diagonalize(self.total)
        try:
            self.view.close()
        except:
            pass
        if self.maxJ != 0:
            self.view = MatrixWindow(matrix=self.total_vec**2,
                                     val='Contributions Matrix',
                                     J=self.maxJ+1)
        else:
            self.view = MatrixWindow(matrix=self.total_vec[0]**2,
                                     val='Contributions Matrix')



    def __SpectroscopicConstantsTab(self):
        
        '''
            Tab Number 4 - Spectroscopic Constants
        
        Used to calculate spectroscopic constants of arbitrary order

        Functions:

        Calculate vibrational constants
        Calculate rotational constants
        Calculate rovibrational constants
        Plot vibrational constants as a function of the J-state
        Plot rotational constants as a functino of the v-state


        '''

        # Initialize some of the starting variables

        self.vib_spec_order = 0
        self.rot_spec_order = 0

        # Change Order of Vibrational Spectroscopic Constants

        self.vib_spec_lab = QLabel("Vibrational Constants")
        self.vib_spec_order_box = QSpinBox(self)
        self.vib_spec_order_box.setRange(0, int(self.max_trunc_val))
        self.vib_spec_order_box.valueChanged.connect(self.__vib_spec_changed)
        self.vib_spec_order_box.valueChanged.connect(self.__rovib_spec_changed)
        self.vib_spec_order_box.setFixedWidth(100)

        self.vib_spec_order = self.vib_spec_order_box.value()

        # Change Order of Rotational Spectroscopic Constants
        
        self.rot_spec_lab = QLabel("Rotational Constants")
        self.rot_spec_order_box = QSpinBox()
        self.rot_spec_order_box.setRange(0, int(self.maxJ-1))
        self.rot_spec_order_box.valueChanged.connect(self.__rot_spec_changed)
        self.rot_spec_order_box.valueChanged.connect(self.__rovib_spec_changed)
        self.rot_spec_order_box.setFixedWidth(100)

        # Rovibrational Label
        
        self.rovib_spec_lab = QLabel("Rovibrational Constants")
    
        # Display Spectroscopic Constants
        
        self.vib_spec_table = QTableWidget()
        self.vib_spec_table.installEventFilter(self)

        self.rot_spec_table = QTableWidget()
        self.rot_spec_table.installEventFilter(self)

        self.rovib_spec_table = QTableWidget()
        self.rovib_spec_table.installEventFilter(self)

        # Plot Spectroscopic Constants

        self.vib_plot = PlotCurve_111(self)
        self.vib_plot_toolbar = NavigationToolbar2QT(self.vib_plot, self)

        self.rot_plot = PlotCurve_111(self)
        self.rot_plot_toolbar = NavigationToolbar2QT(self.rot_plot, self)

        # Change which Spectroscopic Constant is plotted

        self.vib_spec_plot_box = QSpinBox(self)
        self.vib_spec_plot_box.setRange(0, int(self.vib_spec_order))
        self.vib_spec_plot_box.valueChanged.connect(self.__vib_spec_plot_changed)
        self.vib_spec_plot_box.setFixedWidth(100)

        self.rot_spec_plot_box = QSpinBox(self)
        self.rot_spec_plot_box.setRange(0, int(self.rot_spec_order))
        self.rot_spec_plot_box.valueChanged.connect(self.__rot_spec_plot_changed)
        self.rot_spec_plot_box.setFixedWidth(100)

        # Refresh plots + tables

        self.refresh_spec_values = QPushButton("Refresh")
        self.refresh_spec_values.clicked.connect(self.__refresh_spec_values)

        # Define the layout of the tab using a grid

        self.tab4.grid_layout = QGridLayout()

        row=0
        self.tab4.grid_layout.addWidget(self.refresh_spec_values, row, 0, 1, 1, alignment=Qt.AlignLeft)
        self.tab4.grid_layout.addWidget(self.vib_spec_lab, row, 1, 1, 2, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(self.rot_spec_lab, row, 4, 1, 4, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(self.rovib_spec_lab, row, 8, 1, 4, alignment=Qt.AlignCenter)

        row+=1
        self.tab4.grid_layout.addWidget(QLabel("Order:"), row, 0, 1, 2, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(self.vib_spec_order_box, row, 2, 1, 2, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(QLabel("Order:"), row, 4, 1, 2, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(self.rot_spec_order_box, row, 6, 1, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab4.grid_layout.addWidget(QLabel("Plot:"), row, 0, 1, 2, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(self.vib_spec_plot_box, row, 2, 1, 2, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(QLabel("Plot:"), row, 4, 1, 2, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(self.rot_spec_plot_box, row, 6, 1, 2, alignment=Qt.AlignCenter)


        row+=1
        self.tab4.grid_layout.addWidget(self.vib_spec_table, row, 0, 1, 4)
        self.tab4.grid_layout.addWidget(self.rot_spec_table, row, 4, 1, 4)
        self.tab4.grid_layout.addWidget(self.rovib_spec_table, row, 8, 1, 4)

        row+=1
        self.tab4.grid_layout.addWidget(self.vib_plot, row, 0, 1, 6)
        self.tab4.grid_layout.addWidget(self.rot_plot, row, 6, 1, 6)
    
        row+=1
        self.tab4.grid_layout.addWidget(self.vib_plot_toolbar, row, 0, 1, 6, alignment=Qt.AlignCenter)
        self.tab4.grid_layout.addWidget(self.rot_plot_toolbar, row, 6, 1, 6, alignment=Qt.AlignCenter)


        self.tab4.setLayout(self.tab4.grid_layout)

    def __refresh_spec_values(self):
        '''Used to refresh the spectroscopic constants'''

        self.rot_plot.axes.cla()
        self.vib_plot.axes.cla()

        self.__vib_spec_changed()
        self.__rot_spec_changed()
        self.__rovib_spec_changed()
        self.__vib_spec_plot_changed()
        self.__rot_spec_plot_changed()

        
    def __vib_spec_changed(self):
        '''Activated when the order of the vibrational constant changes'''
        try:
            self.vib_column = ["J=" + str(e) for e in range(self.maxJ+2, -1, -1)]
            self.vib_column.append('Constant')
            self.vib_column = self.vib_column[::-1]
            self.vib_row    = self.__get_vib_spec_id()

            self.vib_spec_table.setColumnCount(self.maxJ+2)
            self.vib_spec_table.rowCount()
            self.vib_spec_table.setHorizontalHeaderLabels(self.vib_column)

            while (self.vib_spec_table.rowCount()) > 0:
                self.vib_spec_table.removeRow(0)

            self.vib_spec_order = self.vib_spec_order_box.value()
            self.vib_row = self.__get_vib_spec_id()
            self.__vib_spec_table()

            for i in range(self.vib_spec_order+1):
                self.__add_vib_table_row([self.vib_row[i], *self.vib_spec_values[i]])

            self.vib_spec_plot_box.setRange(0, self.vib_spec_order)

            self.__vib_spec_plot_changed()

        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __rot_spec_changed(self):
        '''Activated when the order of the rotational constant changes'''
        if self.maxJ == 0:
            while (self.rot_spec_table.rowCount()) > 0:
                self.rot_spec_table.removeRow(0)
        else:
            try:
                self.rot_column = ["ν=" + str(e) for e in range(self.max_trunc_val+1, -1, -1)]
                self.rot_column.append("Constant")
                self.rot_column = self.rot_column[::-1]
                self.rot_row    = self.__get_rot_spec_id()

                self.rot_spec_table.setColumnCount(self.max_trunc_val+1)
                self.rot_spec_table.rowCount()
                self.rot_spec_table.setHorizontalHeaderLabels(self.rot_column)

                while (self.rot_spec_table.rowCount()) > 0:
                    self.rot_spec_table.removeRow(0)

                self.rot_spec_order = self.rot_spec_order_box.value()
                self.rot_row = self.__get_rot_spec_id()
                self.__rot_spec_table()

                for i in range(self.rot_spec_order+1):
                    self.__add_rot_table_row([self.rot_row[i], *self.rot_spec_values[i]])

                self.rot_spec_plot_box.setRange(0, self.rot_spec_order)
                
                self.__rot_spec_plot_changed()
            
            except:
                self.errorText = str(traceback.format_exc())
                self.__openErrorMessage()

    def __rovib_spec_changed(self):
        '''Activated when the order of the rovibrational constants change'''
        if self.maxJ == 0:
            while (self.rovib_spec_table.rowCount()) > 0:
                self.rovib_spec_table.removeRow(0)
        else:
            try:
                self.__rovib_spec_table()

                self.rovib_column = ['ν', 'J', 'Value']
                self.rovib_spec_table.setColumnCount(3)
                self.rovib_spec_table.rowCount()
                self.rovib_spec_table.setHorizontalHeaderLabels(self.rovib_column)

                while (self.rovib_spec_table.rowCount()) > 0:
                    self.rovib_spec_table.removeRow(0)

                for i in range(self.vjmat.shape[0]):
                    self.__add_rovib_table_row([*self.vjmat[i], self.rovib_spec_values[i]])

            except:
                self.errorText = str(traceback.format_exc())
                self.__openErrorMessage()

    def __get_vib_spec_id(self):
        '''Used to assign labels given the order of the vibrational constants'''
        vib_spec_id_list = []
        for v in range(self.vib_spec_order+1):
            if v == 0:
                vib_spec_id_list.append('we')
            elif v == 1:
                vib_spec_id_list.append('wexe')
            elif v == 2:
                vib_spec_id_list.append('weye')
            elif v == 3:
                vib_spec_id_list.append('weze')
            else:
                vib_spec_id_list.append('we' + str(v) + 'e')
        return vib_spec_id_list

    def __get_rot_spec_id(self):
        '''Used to assign labels given the order of the vibrational constants'''
        rot_spec_id_list = []
        for j in range(self.rot_spec_order+1):
            if j == 0:
                rot_spec_id_list.append('Be')
            elif j == 1:
                rot_spec_id_list.append('De')
            elif j == 2:
                rot_spec_id_list.append('He')
            elif j == 3:
                rot_spec_id_list.append('Fe')
            else:
                rot_spec_id_list.append(str(j) + 'e')
        return rot_spec_id_list

    def __vib_spec_table(self):
        '''Used to generate the table of vibrational constants'''
        self.total_val, self.total_vec = self.__diagonalize(self.total)
        try:
            spectra = Spectra()
            self.vib_spec_values = spectra.Vibrational(self.total_val,
                                                       int(self.vib_spec_order),
                                                       self.maxJ
                                                       )
        except Exception as e:

            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __rot_spec_table(self):
        '''Used to generate the table of rotational constants'''
        self.total_val, self.total_vec = self.__diagonalize(self.total)
        try:
            if self.maxJ == 0:
                self.rot_spec_values = np.zeros((int(self.rot_spec_order)+1, self.max_trunc_val))
            else:
                spectra = Spectra()

                self.rot_spec_values = spectra.Rotational(self.total_val,
                                                          self.max_trunc_val,
                                                          int(self.rot_spec_order)
                                                          )
        except Exception as e:
            self.errorText = "HERE\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()

    def __rovib_spec_table(self):
        '''Used to generate the table of rovibrational constants'''
        try:
            spectra = Spectra()
            self.rovib_spec_values, self.vjmat = spectra.Rovibrational(self.total_val,
                                                                       int(self.vib_spec_order)+1,
                                                                       int(self.rot_spec_order)+1
                                                                       )
        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __add_vib_table_row(self, row_data):
        '''Used to add a new row of data to the vibrational constants table'''
        row = self.vib_spec_table.rowCount()
        self.vib_spec_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            try:
                if abs(item) < 1e-5:
                    item_str = "{:.5e}".format(item)
                else:
                    item_str = round(item,5)
                cell = QTableWidgetItem(str(item_str))
            except:
                cell = QTableWidgetItem(str(item))
            self.vib_spec_table.setItem(row, col, cell)
            col += 1

    def __add_rot_table_row(self, row_data):
        '''Used to add a new row of data to the vibrational constants table'''
        row = self.rot_spec_table.rowCount()
        self.rot_spec_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            try:
                if abs(item) < 1e-4:
                    item_str = "{:.4e}".format(item)
                else:
                    item_str = round(item,5)
                cell = QTableWidgetItem(str(item_str))
            except:
                cell = QTableWidgetItem(str(item))
            self.rot_spec_table.setItem(row, col, cell)
            col += 1

    def __add_rovib_table_row(self, row_data):
        '''Used to add a new row of data to the vibrational constants table'''
        row = self.rovib_spec_table.rowCount()
        self.rovib_spec_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            if float(item).is_integer():
                cell = QTableWidgetItem(str(int(item)))
            else:
                if abs(item) < 1e-9:
                    item_str = "{:.9e}".format(item)
                else:
                    item_str = round(item,9)
                cell = QTableWidgetItem(str(item_str))
            self.rovib_spec_table.setItem(row, col, cell)
            col += 1

    def __vib_spec_plot_changed(self):
        '''Used to plot the vibrational constant as a function of the j-states'''
        try:
            j_val = int(self.vib_spec_plot_box.value())

            self.vib_plot.axes.cla()
            self.vib_plot.axes.scatter(np.arange(0, self.maxJ+1), self.vib_spec_values[j_val])
            self.vib_plot.axes.set_xlabel("J value")
            self.vib_plot.axes.set_ylabel("Value cm$^{-1}$")
            self.vib_plot.axes.grid()
            self.vib_plot.axes.set_xticks(np.arange(0, self.maxJ+1))
            self.vib_plot.draw()

        except:
            self.vib_plot.axes.cla()
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __rot_spec_plot_changed(self):
        '''Used to plot the rotational constant as a function of the v-states'''
        try:
            v_val = int(self.rot_spec_plot_box.value())

            if self.maxJ == 0:
                vals = np.zeros((self.max_trunc_val))
                self.rot_plot.axes.cla()
                self.rot_plot.axes.text(0.5, 0.5, "No Data To Plot", horizontalalignment='center')
                self.rot_plot.draw()
            else:
                vals = self.rot_spec_values[v_val,:-1]

                if sum(np.sign(vals)) == vals.size:
                    bottom = np.amin(vals)*0.95
                    top = np.amax(vals)*1.05

                elif sum(np.sign(vals)) == vals.size * -1:
                    bottom = np.amax(vals)*0.95
                    top = np.amin(vals)*1.05
                else:
                    bottom = np.amin(vals)*0.95
                    top = np.amax(vals)*1.05

                self.rot_plot.axes.cla()
                self.rot_plot.axes.scatter(np.arange(0, self.max_trunc_val), vals[:self.max_trunc_val])
                self.rot_plot.axes.set_xlabel("ν value")
                self.rot_plot.axes.set_ylabel("Value cm$^{-1}$")
                self.rot_plot.axes.grid()
                self.rot_plot.axes.set_xticks(np.arange(0, self.max_trunc_val))
                self.rot_plot.axes.set_ylim(bottom, top)
                self.rot_plot.axes.ticklabel_format(style='sci')
                self.rot_plot.draw()

        except:
            self.rot_plot.axes.cla()
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()


    
    def __ExcitationsTab(self):

        '''
            Tab Number 5 - Rovibrational Excitations

        Used to calculate all possible rovibrational excitations

        Functions:

        Calculate all rovibrational excitations with intensities (if given)
        Sort by J-value for vibrational excitations
        Sort by v-value for rotational excitattions
        Sort by J/v-values for rovibrational excitations


        '''

        # Initialize some of the variables

        self.exc_labels = ['Vi', 'Ji', 'Vj', 'Jj', 'Ei', 'Ej', 'dE', 'TDM', 'f', 'A']

        # Refresh Button

        self.excitations_refresh_btn = QPushButton("Refresh")
        self.excitations_refresh_btn.clicked.connect(self.__refresh_excitations)

        # View excitation data

        self.excitation_vib_btn = QPushButton("Sort by Vibrational Excitations")
        self.excitation_rot_btn = QPushButton("Sort by Rotational Excitations")
        self.excitation_all_btn = QPushButton("Sort by Rovibrational Excitations")

        self.excitation_vib_btn.clicked.connect(self.__sort_vib_excitations)
        self.excitation_rot_btn.clicked.connect(self.__sort_rot_excitations)
        self.excitation_all_btn.clicked.connect(self.__sort_all_excitations)

        # Excitation Table

        self.excitations_table = QTableWidget()
        self.excitations_table.installEventFilter(self)
        
        self.excitations_table.setColumnCount(len(self.exc_labels))
        self.excitations_table.rowCount()
        self.excitations_table.setHorizontalHeaderLabels(self.exc_labels)

        self.excitations_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)


        # Define the layout of the tab using a grid

        self.tab5.grid_layout = QGridLayout()

        row=0
        self.tab5.grid_layout.addWidget(self.excitations_refresh_btn, row, 1, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab5.grid_layout.addWidget(self.excitation_vib_btn, row, 0, 1, 1, alignment=Qt.AlignCenter)
        self.tab5.grid_layout.addWidget(self.excitation_rot_btn, row, 1, 1, 1, alignment=Qt.AlignCenter)
        self.tab5.grid_layout.addWidget(self.excitation_all_btn, row, 2, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab5.grid_layout.addWidget(self.excitations_table, row, 0, 1, 3)

        self.tab5.setLayout(self.tab5.grid_layout)

    def __refresh_excitations(self):
        '''Used to refresh the calculated rovibrational excitations'''
        try:
            self.total_val, self.total_vec = self.__diagonalize(self.total)
            excite = Spectra()
            self.excitations = excite.Excitations(self.total_val,
                                                  self.total_vec,
                                                  int(self.max_trunc_val),
                                                  int(self.maxJ),
                                                  self.tdm
                                                  )
            self.__sort_all_excitations()
        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()


    def __sort_vib_excitations(self):
        '''Used to sort the excitation data by J-values for vibrational excitations'''
        try:
            self.df = pd.DataFrame()

            for j in range(len(self.exc_labels)):
                self.df[self.exc_labels[j]] = self.excitations[:,j]
                if j < 4:
                    self.df = self.df.astype({self.exc_labels[j] : int})

            self.df = self.df.sort_values(by=['Ji', 'Jj'])
            self.df = self.df[self.df['Ji'] == self.df['Jj']]

            self.excitations_ = self.df.to_numpy()

            while (self.excitations_table.rowCount()) > 0:
                self.excitations_table.removeRow(0)

            for i in range(self.excitations_.shape[0]):
                self.__add_excitations_table_row([*self.excitations_[i]])

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __sort_rot_excitations(self):
        '''Used to sort the excitation data by v-values for rotational excitations'''
        try:
            self.df = pd.DataFrame()

            for j in range(len(self.exc_labels)):
                self.df[self.exc_labels[j]] = self.excitations[:,j]
                if j < 4:
                    self.df = self.df.astype({self.exc_labels[j] : int})

            self.df = self.df.sort_values(by=['Vi', 'Vj'])
            self.df = self.df[self.df['Vi'] == self.df['Vj']]

            self.excitations_ = self.df.to_numpy()

            while (self.excitations_table.rowCount()) > 0:
                self.excitations_table.removeRow(0)

            for i in range(self.excitations_.shape[0]):
                self.__add_excitations_table_row([*self.excitations_[i]])

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __sort_all_excitations(self):
        '''Used to sort the excitation data by v/J-values for rovibrational excitations'''
        try:
            self.df = pd.DataFrame()

            for j in range(len(self.exc_labels)):
                self.df[self.exc_labels[j]] = self.excitations[:,j]
                if j < 4:
                    self.df = self.df.astype({self.exc_labels[j] : int})

            while (self.excitations_table.rowCount()) > 0:
                self.excitations_table.removeRow(0)

            for i in range(self.excitations.shape[0]):
                self.__add_excitations_table_row([*self.excitations[i]])

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __add_excitations_table_row(self, row_data):
        '''Used to add a new row of data to the data table'''
        row = self.excitations_table.rowCount()
        self.excitations_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            if col < 4:
                cell = QTableWidgetItem(str(int(item)))
            else:
                cell = QTableWidgetItem(str(item))
            self.excitations_table.setItem(row, col, cell)
            col += 1



    def __DunhamTab(self):

        '''
            Tab Number 6 - Dunham Fit

        Used to fit the energy curve to a Dunham polynomial and calculate Dunham parameters

        Functions:

            Fit the energy curve to a Dunham-type polynomial
            Use the fit coefficients to calculate Dunham Y-parameters (spectrocscopic constants)

        '''

        # Dunham Coefficients

        self.dunham_coef_table = QTableWidget()

        self.dunham_coef_table.setColumnCount(2)
        self.dunham_coef_table.rowCount()
        self.dunham_coef_table.setHorizontalHeaderLabels(['Order', 'Value'])

        self.dunham_coef_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.dunham_coef_table.installEventFilter(self)

        # Dunham Parameters

        self.dunham_param_table = QTableWidget()

        self.dunham_param_table.setColumnCount(4)
        self.dunham_param_table.rowCount()
        self.dunham_param_table.setHorizontalHeaderLabels(['J', 'ν', 'ID', 'Value'])

        self.dunham_param_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.dunham_param_table.installEventFilter(self)

        # refresh button

        self.dunham_refresh_btn = QPushButton("Refresh")
        self.dunham_refresh_btn.clicked.connect(self.__dunham_vals)
        self.dunham_refresh_btn.setFixedWidth(100)

        # labels

        self.dunham_coef_lab = QLabel("Dunham Coefficients")
        self.dunham_param_lab = QLabel("Dunham Parameters")

        self.dunham_param_lab.setFixedHeight(10)
        self.dunham_coef_lab.setFixedHeight(10)


        # Define the layout of the tab using a grid

        self.tab6.grid_layout = QGridLayout()

        row=0
        self.tab6.grid_layout.addWidget(self.dunham_refresh_btn, row, 0, 1, 5, alignment=Qt.AlignCenter)
        
        row+=1
        self.tab6.grid_layout.addWidget(self.dunham_coef_lab, row, 0, 1, 2, alignment=Qt.AlignCenter)
        self.tab6.grid_layout.addWidget(self.dunham_param_lab, row, 2, 1, 3, alignment=Qt.AlignCenter)

        
        row+=1
        self.tab6.grid_layout.addWidget(self.dunham_coef_table, row, 0, 10, 2, alignment=Qt.AlignCenter)
        self.tab6.grid_layout.addWidget(self.dunham_param_table, row, 2, 10, 3, alignment=Qt.AlignCenter)

        self.tab6.setLayout(self.tab6.grid_layout)



    def __dunham_vals(self):
        '''Used to calculate and sort the Dunham Y-parameters and coefficients'''
        try:
            dunham = Spectra()
            self.dunham_Y, self.dunham_coef = dunham.Dunham(self.temp_data[0], 
                                                            self.temp_data[1], 
                                                            self.rEq, 
                                                            self.eEq, 
                                                            self.reduced_mass,
                                                            self.wEq,
                                                            self.bEq
                                                            )

            self.Y_id_one = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0 ,1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
            self.Y_id_two = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
            self.Y_id_thr = ['-',    'Be',    'De',   'He', 'Fe',
                            'we',   'alpha', 'beta', '-',   '-',
                            'wexe', 'gamma',  '-',   '-',   '-',
                            'weye', '-',      '-',   '-',   '-',
                            'weze', '-',      '-',   '-',   '-']

            while self.dunham_param_table.rowCount() > 0:
                self.dunham_param_table.removeRow(0)

            for j in range(len(self.dunham_Y)):
                self.__add_dunham_param_table_row([self.Y_id_one[j],
                                                   self.Y_id_two[j], 
                                                   self.Y_id_thr[j],
                                                   self.dunham_Y[j]])


            while self.dunham_coef_table.rowCount() > 0:
                self.dunham_coef_table.removeRow(0)

            for j in range(len(self.dunham_coef)):
                self.__add_dunham_coef_table_row([str(j+1), self.dunham_coef[j]])

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __add_dunham_param_table_row(self, row_data):
        '''Used to add a new row to the Dunham parameter table'''
        row = self.dunham_param_table.rowCount()
        self.dunham_param_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            if col < 2:
                cell = QTableWidgetItem(str(int(item)))
            else:
                cell = QTableWidgetItem(str(item))
            self.dunham_param_table.setItem(row, col, cell)
            col += 1

    def __add_dunham_coef_table_row(self, row_data):
        '''Used to add a new row to the Dunham coefficient table'''
        row = self.dunham_coef_table.rowCount()
        self.dunham_coef_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            if col < 1:
                cell = QTableWidgetItem(str(int(item)))
            else:
                cell = QTableWidgetItem(str(item))
            self.dunham_coef_table.setItem(row, col, cell)
            col += 1

    

    def __WavefunctionsTab(self):

        '''
            Tab Number 7 - Wavefunctions

        Used to calculate the vibrational wavefunctions on different J-surfaces

        '''

        # Initialize some of the variables

        self._wfs_j_val = 0

        # View Wavefunctions

        self.view_wfs_lab1 = QLabel("J Surface :")

        self.view_wfs_j_box = QSpinBox()
        self.view_wfs_j_box.setRange(0, self.maxJ)
        self.view_wfs_j_box.valueChanged.connect(self.__wavefunction_view)

        self.view_wfs_lab2 = QLabel("Number of wavefunctions to plot:")

        self.view_wfs_v_box = QSpinBox()
        self.view_wfs_v_box.setRange(0, self.max_trunc_val-1)
        self.view_wfs_v_box.valueChanged.connect(self.__wavefunction_view)

        self.wfs_plot = PlotCurve_111(self)

        self.wf_toolbar = NavigationToolbar2QT(self.wfs_plot, self)

        self.wf_refresh_btn = QPushButton("Refresh")
        self.wf_refresh_btn.clicked.connect(self.__wavefunction_view)

        # Define the layout of the tab using a grid

        self.tab7.grid_layout = QGridLayout()

        row=0
        self.tab7.grid_layout.addWidget(self.view_wfs_lab1, row, 1, 1, 1, alignment=Qt.AlignCenter)
        self.tab7.grid_layout.addWidget(self.view_wfs_j_box, row, 2, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab7.grid_layout.addWidget(self.view_wfs_lab2, row, 1, 1, 1, alignment=Qt.AlignCenter)
        self.tab7.grid_layout.addWidget(self.view_wfs_v_box, row, 2, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab7.grid_layout.addWidget(self.wf_refresh_btn, row, 1, 1, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab7.grid_layout.addWidget(self.wfs_plot, row, 0, 10, 4, alignment=Qt.AlignCenter)

        row+=10
        self.tab7.grid_layout.addWidget(self.wf_toolbar, row, 0, 1, 4, alignment=Qt.AlignCenter)

        self.tab7.setLayout(self.tab7.grid_layout)
        


    def __wavefunction_view(self):
        '''Used to calculate and plot the wave functions'''
        J = self.view_wfs_j_box.value()
        v = self.view_wfs_v_box.value()

        try:

            self.total_val, self.total_vec = self.__diagonalize(self.total)

            wfs = Wavefunctions(vals=self.total_val[J],
                                vecs=self.total_vec[J,:],
                                trap=2000,
                                beta=self.beta,
                                L=np.amin(self.PEC_r_),
                                R=np.amax(self.PEC_r_),
                                maxV=int(v),
                                )

            wfs_x, wfs_y = wfs.wfs

            self.wfs_plot.axes.cla()

            self.wfs_plot.axes.plot(self.PEC_r,
                                 self.PEC_e*hart_cm,
                                 linewidth=1,
                                 color='k')

            self.wfs_plot.axes.scatter(self.PEC_r,
                                    self.PEC_e*hart_cm,
                                    s=10,
                                    color='k',
                                    marker='o')

            for vv in range(int(v)+1):
                if self.total_val[J,vv] > 0:
                    self.wfs_plot.axes.plot(wfs_x/(wfs.beta),
                                         (wfs_y[vv] * (self.total_val[J,vv+1] - self.total_val[J,vv])*0.75) + self.total_val[J,vv],
                                         label='Ψ$_{' + str(vv) + "}$"
                                         )
                    self.wfs_plot.axes.hlines(self.total_val[J,vv],
                                           np.amin(self.PEC_r_),
                                           np.amax(self.PEC_r_)
                                           )

            self.wfs_plot.axes.set_title("Wavefunctions on J = " + str(J))
            self.wfs_plot.axes.set_xlabel("Displacement from R$_{eq}$ (\u212B)")
            self.wfs_plot.axes.set_ylabel("Energy (cm$^{-1}$)")

            self.wfs_plot.axes.grid()
            self.wfs_plot.axes.legend()
            self.wfs_plot.draw()

        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()




    def __TurningPointsTab(self):

        '''
            Tab Number 7 - Turning Points

        Used to calculate the classical turning points for a diatomic molecule on different J-surfaces
        '''

        # Initialize some of the variables

        self.tps_j_val = 0

        # Calculate Turning Points

        self.tps_lab = QLabel("Turning Points on J = ") 
        
        self.tps_box = QSpinBox()
        self.tps_box.setRange(0, self.maxJ)
        self.tps_box.valueChanged.connect(self.__plot_tps)
        
        self.tps_refresh_btn = QPushButton("Refresh")
        self.tps_refresh_btn.clicked.connect(self.__plot_tps)

        # Turning Point table

        self.tps_table = QTableWidget()
        self.tps_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tps_table.installEventFilter(self)

        # Turning Point plots

        self.tps_plot = PlotCurve_111(self)
        self.tps_plot_tb = NavigationToolbar2QT(self.tps_plot, self)

        # Define the layout of the tab using a grid

        self.tab8.grid_layout = QGridLayout()

        row=0
        self.tab8.grid_layout.addWidget(self.tps_lab, row, 3, 1, 1, alignment=Qt.AlignCenter)
        self.tab8.grid_layout.addWidget(self.tps_box, row, 4, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab8.grid_layout.addWidget(self.tps_refresh_btn, row, 3, 1, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab8.grid_layout.addWidget(self.tps_table, row, 0, 5, 4, alignment=Qt.AlignCenter)
        self.tab8.grid_layout.addWidget(self.tps_plot, row, 4, 5, 4, alignment=Qt.AlignCenter)

        row+=5
        self.tab8.grid_layout.addWidget(self.tps_plot_tb, row, 4, 1, 2, alignment=Qt.AlignCenter)
    

        self.tab8.setLayout(self.tab8.grid_layout)



    def __plot_tps(self):
        '''Calculate and plot the classical turning points on different J-surfaces'''
        self.tps_j_val = self.tps_box.value()

        try:
            tp = Spectra()
            self.tps = tp.TurningPoints(self.PEC_r_+self.rEq, 
                                        self.PEC_e_*hart_cm, 
                                        self.total_val[self.tps_j_val,:int(self.max_trunc_val)+1],
                                        self.rEq,
                                        )

            self.df = pd.DataFrame({"Number" : np.arange(self.tps.shape[1]),
                                    "Left"   : self.tps[0],
                                    "Center" : (self.tps[0] + self.tps[1]) / 2.,
                                    "Right"  : self.tps[1],
                                    })

            self.tps_plot.axes.cla()
            self.tps_plot.axes.set_ylabel("Vibrational State")
            self.tps_plot.axes.set_xlabel("Turning Point (Å)")

            left_tps = self.tps[0][np.where(self.tps[0] > 0)]
            right_tps = self.tps[1][np.where(self.tps[1] > 0)]
            center = (self.tps[0] + self.tps[1]) / 2.
            center = center[:min(left_tps.size, right_tps.size)]

            self.tps_plot.axes.scatter(left_tps, np.arange(0, left_tps.size), label='Left', color='r')
            self.tps_plot.axes.scatter(center, np.arange(0, center.size), label='Center', color='purple')
            self.tps_plot.axes.scatter(right_tps, np.arange(0, right_tps.size), label='Right', color='b')

            self.tps_plot.axes.legend()
            self.tps_plot.axes.grid()

            self.tps_plot.draw()

            self.tps_table.setColumnCount(3)
            self.tps_table.rowCount()
            self.tps_table.setHorizontalHeaderLabels(['Left', 'Center', 'Right'])

            while self.tps_table.rowCount() > 0:
                self.tps_table.removeRow(0)

            for j in range(self.tps[0].size):
                self.__add_tps_table([self.tps[0,j], 
                                     (self.tps[0,j] + self.tps[1,j]) / 2.,
                                     self.tps[1,j]]
                                     )

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __add_tps_table(self, row_data):
        '''Used to add a row to the turning points data table'''
        row = self.tps_table.rowCount()
        self.tps_table.setRowCount(row+1)
        col = 0
        for item in row_data:
            if item == 0:
                cell = QTableWidgetItem('-')
            else:
                cell = QTableWidgetItem(str(round(item, 8)))
            self.tps_table.setItem(row, col, cell)
            col += 1

    
    def __SimulatedSpectraTab(self):

        '''
            Tab Number 9 - Simulated Spectra Tab

        Used to calculate the population of each quantum state and the intensity of each transition (if dipole given)

        Functions:
            Calculate the population of vibrational states on different J-surfaces
            Calculate the population of rotational states on different v-surfaces
            Calculate the population of rovibrational states
            Calculate the intensity of vibrational excitations on one or all J-surfaces
            Calculate the intensity of rotational excitations on one or all v-surfaces
            Calculate the intensity of rovibrational excitations
            Plot all population/intensity data
            Display all population/intensity data in an external table

        '''

        # Initialize some variables

        self.temp = 298
        self.vib_cutoff = 0.001
        self.rot_cutoff = 0.001
        self.rov_cutoff = 0.001

        self.vib_method = 'pop'
        self.rot_method = 'pop'
        self.rov_method = 'pop'

        # Chosen cutoff ranges

        self.vib_cutoff_lab = QLabel("Plotting Cutoff")
        self.rot_cutoff_lab = QLabel("Plotting Cutoff")
        self.rov_cutoff_lab = QLabel("Plotting Cutoff")

        self.vib_cutoff_val = QLineEdit(str(self.vib_cutoff))
        self.rot_cutoff_val = QLineEdit(str(self.rot_cutoff))
        self.rov_cutoff_val = QLineEdit(str(self.rov_cutoff))

        self.vib_cutoff_val.editingFinished.connect(self.__change_vib_cutoff)
        self.rot_cutoff_val.editingFinished.connect(self.__change_rot_cutoff)
        self.rov_cutoff_val.editingFinished.connect(self.__change_rov_cutoff)

        # Chosen Temperature

        self.temp_lab = QLabel("Temperature (K)")

        self.temp_str = QLineEdit(str(self.temp))
        self.temp_str.editingFinished.connect(self.__change_temp)
        self.temp_str.setFixedWidth(100)

        # Refresh Button

        self.sim_spec_refresh_btn = QPushButton("Refresh")
        self.sim_spec_refresh_btn.setFixedWidth(250)
        self.sim_spec_refresh_btn.clicked.connect(self.__change_temp)

        # Vibrational spectra

        self.vib_sim_spec_j_lab = QLabel("Rotational Surface")
        self.vib_sim_spec_v_lab = QLabel("Maximum ν States")

        self.vib_sim_spec_j_val = QSpinBox()
        self.vib_sim_spec_j_val.setRange(-1, self.maxJ)
        self.vib_sim_spec_j_val.setValue(0)
        self.vib_sim_spec_j_val.valueChanged.connect(self.__change_vib_sim_spec)
        self.vib_sim_spec_j_val.setFixedWidth(50)

        self.vib_sim_spec_v_val = QSpinBox()
        self.vib_sim_spec_v_val.setRange(0, self.max_trunc_val)
        self.vib_sim_spec_v_val.setValue(0)
        self.vib_sim_spec_v_val.setFixedWidth(50)
        self.vib_sim_spec_v_val.valueChanged.connect(self.__change_vib_sim_spec)

        self.vib_sim_spec_cutoff = QLineEdit(str(self.vib_cutoff))
        self.vib_sim_spec_cutoff.editingFinished.connect(self.__change_vib_sim_spec)

        self.vib_sim_spec_plot = PlotCurve_111(self)
        self.vib_sim_spec_plot_tb = NavigationToolbar2QT(self.vib_sim_spec_plot, self)
    
        self.vib_sim_spec_method_group = QButtonGroup(self)
    
        self.vib_sim_spec_method_pop = QRadioButton("Population")
        self.vib_sim_spec_method_int = QRadioButton("Intensity")
        
        self.vib_sim_spec_method_pop.setChecked(True)

        self.vib_sim_spec_method_pop.toggled.connect(lambda:self.__change_vib_method(self.vib_sim_spec_method_pop))
        self.vib_sim_spec_method_int.toggled.connect(lambda:self.__change_vib_method(self.vib_sim_spec_method_int))

        self.vib_sim_spec_method_group.addButton(self.vib_sim_spec_method_pop)
        self.vib_sim_spec_method_group.addButton(self.vib_sim_spec_method_int)

        self.vib_sim_spec_table = QPushButton("Vew Datatable")
        self.vib_sim_spec_table.clicked.connect(self.__vib_spec_datatable)

        # Rotational spectra

        self.rot_sim_spec_v_lab = QLabel("Vibrational Surface")
        self.rot_sim_spec_j_lab = QLabel("Maximum J States")

        self.rot_sim_spec_j_val = QSpinBox()
        self.rot_sim_spec_j_val.setRange(0, self.maxJ)
        self.rot_sim_spec_j_val.setValue(0)
        self.rot_sim_spec_j_val.valueChanged.connect(self.__change_rot_sim_spec)
        self.rot_sim_spec_j_val.setFixedWidth(50)

        self.rot_sim_spec_v_val = QSpinBox()
        self.rot_sim_spec_v_val.setRange(-1, self.max_trunc_val)
        self.rot_sim_spec_v_val.setValue(0)
        self.rot_sim_spec_v_val.valueChanged.connect(self.__change_rot_sim_spec)
        self.rot_sim_spec_v_val.setFixedWidth(50)

        self.rot_sim_spec_plot = PlotCurve_111(self)
        self.rot_sim_spec_plot_tb = NavigationToolbar2QT(self.rot_sim_spec_plot, self)

        self.rot_sim_spec_method_group = QButtonGroup(self)
        
        self.rot_sim_spec_method_pop = QRadioButton("Population")
        self.rot_sim_spec_method_int = QRadioButton("Intensity")

        self.rot_sim_spec_method_pop.setChecked(True)

        self.rot_sim_spec_method_pop.toggled.connect(lambda:self.__change_rot_method(self.rot_sim_spec_method_pop))
        self.rot_sim_spec_method_int.toggled.connect(lambda:self.__change_rot_method(self.rot_sim_spec_method_int))

        self.rot_sim_spec_method_group.addButton(self.rot_sim_spec_method_pop)
        self.rot_sim_spec_method_group.addButton(self.rot_sim_spec_method_int)

        self.rot_sim_spec_table = QPushButton("Vew Datatable")
        self.rot_sim_spec_table.clicked.connect(self.__rot_spec_datatable)

        # Rovibrational spectra

        self.rov_sim_spec_j_lab = QLabel("Maximum J Value")
        self.rov_sim_spec_v_lab = QLabel("Maximum ν value")

        self.rov_sim_spec_j_val = QSpinBox()
        self.rov_sim_spec_j_val.setRange(0, self.maxJ)
        self.rov_sim_spec_j_val.valueChanged.connect(self.__change_rov_sim_spec)
        self.rov_sim_spec_j_val.setFixedWidth(50)

        self.rov_sim_spec_v_val = QSpinBox()
        self.rov_sim_spec_v_val.setRange(0, self.max_trunc_val)
        self.rov_sim_spec_v_val.valueChanged.connect(self.__change_rov_sim_spec)
        self.rov_sim_spec_v_val.setFixedWidth(50)

        self.rov_sim_spec_plot = PlotCurve_111(self)
        self.rov_sim_spec_plot_tb = NavigationToolbar2QT(self.rov_sim_spec_plot, self)

        self.rov_sim_spec_method_pop = QRadioButton("Population")
        self.rov_sim_spec_method_int = QRadioButton("Intensity")

        self.rov_sim_spec_method_pop.setChecked(True)

        self.rov_sim_spec_method_pop.toggled.connect(lambda:self.__change_rov_method(self.rov_sim_spec_method_pop))
        self.rov_sim_spec_method_int.toggled.connect(lambda:self.__change_rov_method(self.rov_sim_spec_method_int))

        self.rov_sim_spec_table = QPushButton("Vew Datatable")
        self.rov_sim_spec_table.clicked.connect(self.__rov_spec_datatable)

        
        # Define the layout of the tab using a grid

        self.tab9.grid_layout = QGridLayout()

        row=0
        self.tab9.grid_layout.addWidget(QLabel(""), row, 0, 3, 12)

        row+=3
        self.tab9.grid_layout.addWidget(self.temp_lab, row, 5, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.temp_str, row, 6, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab9.grid_layout.addWidget(self.sim_spec_refresh_btn, row, 5, 1, 4)

        row+=1 
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_j_lab, row, 1, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_j_val, row, 2, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_v_lab, row, 5, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_v_val, row, 6, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_j_lab, row, 9, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_j_val, row, 10, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_v_lab, row, 1, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_v_val, row, 2, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_j_lab, row, 5, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_j_val, row, 6, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_v_lab, row, 9, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_v_val, row, 10, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab9.grid_layout.addWidget(self.vib_cutoff_lab, row, 1, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.vib_cutoff_val, row, 2, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_cutoff_lab, row, 5, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_cutoff_val, row, 6, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_cutoff_lab, row, 9, 1, 1, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_cutoff_val, row, 10, 1, 1, alignment=Qt.AlignCenter)

        row+=1
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_plot, row, 0, 6, 4, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_plot, row, 4, 6, 4, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_plot, row, 8, 6, 4, alignment=Qt.AlignCenter)

        row+=6
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_plot_tb, row, 1, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_plot_tb, row, 5, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_plot_tb, row, 9, 1, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_method_pop, row, 1, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_method_pop, row, 5, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_method_pop, row, 9, 1, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_method_int, row, 1, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_method_int, row, 5, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_method_int, row, 9, 1, 2, alignment=Qt.AlignCenter)

        row+=1
        self.tab9.grid_layout.addWidget(self.vib_sim_spec_table, row, 1, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rot_sim_spec_table, row, 5, 1, 2, alignment=Qt.AlignCenter)
        self.tab9.grid_layout.addWidget(self.rov_sim_spec_table, row, 9, 1, 2, alignment=Qt.AlignCenter)


        self.tab9.setLayout(self.tab9.grid_layout)


    def __change_temp(self):
        '''Used to modify the plots and tables given a change in temperature'''
        try:
            float(self.temp_str.text())

            if float(self.temp_str.text()) < 0:
                self.temp_str.setText(str(self.temp))
            else:
                self.temp = float(self.temp_str.text())

                self.__change_vib_sim_spec()
                self.__change_rot_sim_spec()
                self.__change_rov_sim_spec()
        except:
            self.errorText = "Temperature must be a positive value\n\n" + str(traceback.format_exc())
            self.__openErrorMessage()
            self.temp_str.setText(str(self.temp))

    def __change_vib_limits(self):
        '''Used to change the vibrational plots and tables given new v/J limits'''
        self.vib_sim_spec_v_val.setRange(1, self.max_trunc_val)
        self.vib_sim_spec_v_val.setValue(self.max_trunc_val)
        self.vib_sim_spec_j_val.setRange(-1, self.maxJ)
        self.vib_sim_spec_j_val.setValue(0)

        self.__change_vib_sim_spec()

    def __change_rot_limits(self):
        '''Used to change the rotational plots and tables given new v/J limits'''
        self.rot_sim_spec_j_val.setRange(0, self.maxJ)
        self.rot_sim_spec_j_val.setValue(self.maxJ)
        self.rot_sim_spec_v_val.setRange(-1, self.max_trunc_val)
        self.rot_sim_spec_v_val.setValue(0)

        self.__change_rot_sim_spec()

    def __change_rov_limits(self):
        '''Used to change the rovibrational plots and tables given new v/J limits'''
        self.rov_sim_spec_j_val.setRange(0, self.maxJ)
        self.rov_sim_spec_j_val.setValue(0)
        self.rov_sim_spec_v_val.setRange(0, self.max_trunc_val)
        self.rov_sim_spec_v_val.setValue(0)
        
        self.__change_rov_sim_spec()

    
    def __change_vib_method(self, b):
        '''Used to change the method displayed for the vibrational plots and tables. 
            Population refers to the population of individual vibrational states
            Intensity refers to the intensity of each pure vibrational transition'''
        if b.text() == "Population":
            if b.isChecked() == True:
                self.vib_method = 'pop'
            else:
                self.vib_method = 'int'
        elif b.text() == "Intensity":
            if b.isChecked() == True:
                self.vib_method = 'int'
            else:
                self.vib_method = 'pop'

        self.__change_vib_sim_spec()

    def __change_rot_method(self, b):
        '''Used to change the method displayed for the rotational plots and tables. 
            Population refers to the population of individual rotational states
            Intensity refers to the intensity of each pure rotational transition'''
        if b.text() == "Population":
            if b.isChecked() == True:
                self.rot_method = 'pop'
            else:
                self.rot_method = 'int'
        elif b.text() == "Intensity":
            if b.isChecked() == True:
                self.rot_method = 'int'
            else:
                self.rot_method = 'pop'

        self.__change_rot_sim_spec()

    def __change_rov_method(self, b):
        '''Used to change the method displayed for the rovibrational plots and tables. 
            Population refers to the population of individual rovibrational states
            Intensity refers to the intensity of each pure rovibrational transition'''
        if b.text() == "Population":
            if b.isChecked() == True:
                self.rov_method = 'pop'
            else:
                self.rov_method = 'int'
        elif b.text() == "Intensity":
            if b.isChecked() == True:
                self.rov_method = 'int'
            else:
                self.rov_method = 'pop'

        self.__change_rov_sim_spec()


    def __change_vib_cutoff(self):
        '''Used to change the cutoff value for plotting the vibrational population/intensity data'''
        try:
            self.vib_cutoff = float(self.vib_cutoff_val.text())
        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __change_rot_cutoff(self):
        '''Used to change the cutoff value for plotting the rotational population/intensity data'''
        try:
            self.rot_cutoff = float(self.rot_cutoff_val.text())
        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __change_rov_cutoff(self):
        '''Used to change the cutoff value for plotting the rovibrational population/intensity data'''
        try:
            self.rov_cutoff = float(self.rov_cutoff_val.text())
        except:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()


    def __change_vib_sim_spec(self):
        '''Used to update the plot and data table with the vibrational states.
           
           Calculates the population of each state using a Boltzmann distribution and 
           uses those values with the calculated Einstein-A coefficients to determine the 
           intensity of a given vibrational transition'''
        try:
            val, vec = self.__diagonalize(self.total)
            
            sim = Spectra()

            J  = int(self.vib_sim_spec_j_val.value())
            V = int(self.vib_sim_spec_v_val.value())

            if J == -1:
                method = 'rov'
            else:
                method = 'vib'

            pop = sim.SimulatedVibrationalPop(temp=self.temp, 
                                           J=J,
                                           v=V,
                                           method=method,
                                           vals=val,
                                           )

            if self.vib_method == 'pop':
                max_print_v = V

                self.vib_sim_spec_plot.axes.cla()

                if J != -1:
                    for v_ in range(pop.shape[1]):
                        if pop[1,v_] > self.vib_cutoff:
                            self.vib_sim_spec_plot.axes.vlines(v_, 0, pop[1,v_])
                        else:
                            max_print_v = v_
                    self.vib_sim_spec_plot.axes.set_xlabel("ν")
                    self.vib_sim_spec_plot.axes.set_ylabel("Population")
                    self.vib_sim_spec_plot.axes.set_xticks(np.arange(0, max_print_v, 2))
                    self.vib_sim_spec_plot.draw()
                
                else:
                    for v_ in range(pop.shape[2]):
                        for j_ in range(pop.shape[0]):
                            self.vib_sim_spec_plot.axes.text(v_, 
                                                             j_, 
                                                             round(pop[j_, 1, v_], 3),
                                                             ha='center',   
                                                             va='center',
                                                             color=plt.cm.binary(pop[j_, 1, v_]/np.amax(pop[:,1,:]))
                                                             )

                    self.vib_sim_spec_plot.axes.set_title("Population for all States")
                    self.vib_sim_spec_plot.axes.set_xlim(-0.5, pop.shape[2] - 0.5)
                    self.vib_sim_spec_plot.axes.set_ylim(-0.5, pop.shape[0] - 0.5)
                    self.vib_sim_spec_plot.axes.set_xlabel("V-state")
                    self.vib_sim_spec_plot.axes.set_ylabel("J-state")
                    self.vib_sim_spec_plot.draw()

            elif self.vib_method == 'int':

                self.vib_sim_spec_plot.axes.cla()

                inten = sim.SimulatedVibrationalInt(J=J,
                                                v=V,
                                                val=val,
                                                vec=vec,
                                                tdm=self.tdm,
                                                pop=pop, 
                                                method=method)

                if J != -1:
                    for v_ in range(inten.shape[1]):
                        if abs(inten[1,v_]) > self.vib_cutoff:
                            self.vib_sim_spec_plot.axes.vlines(inten[0,v_], 0, abs(inten[1,v_]))
                else:
                    CHARS = "0123456789ABCDEF"

                    for j in range(inten.shape[0]):
                        leg = False
                        color = "#"
                        for k in range(6):
                            color += CHARS[np.random.randint(0, 15)]
                        for v_ in range(inten.shape[2]):
                            if abs(inten[j,1,v_]) > self.vib_cutoff:
                                if leg == False:
                                    self.vib_sim_spec_plot.axes.vlines(inten[j,0,v_], 0, abs(inten[j,1,v_]), color=color, label='J=' + str(j))
                                    leg = True
                                else:
                                    self.vib_sim_spec_plot.axes.vlines(inten[j,0,v_], 0, abs(inten[j,1,v_]), color=color)

                    self.vib_sim_spec_plot.axes.legend()

                self.vib_sim_spec_plot.axes.set_xlabel("Energy (cm$^{-1}$)")
                self.vib_sim_spec_plot.axes.set_ylabel("Intensity (s$^{-1}$)")
                self.vib_sim_spec_plot.draw()
 
        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()


    def __change_rot_sim_spec(self):
        '''Used to update the plot and data table with the rotational states.

           Calculates the population of each state using a Boltzmann distribution and
           uses those values with the calculated Einstein-A coefficients to determine the
           intensity of a given rotational transition'''

        try:
            self.rot_cutoff = float(self.rot_cutoff_val.text())
            val, vec = self.__diagonalize(self.total)

            sim = Spectra()

            J = int(self.rot_sim_spec_j_val.value())
            v = int(self.rot_sim_spec_v_val.value())

            if v == -1:
                method = 'rov'
            else:
                method = 'rot'

            pop = sim.SimulatedRotationalPop(temp=self.temp,
                                          J=J,
                                          v=v,
                                          method=method,
                                          vals=val,
                                          )

            if self.rot_method == 'pop':
                max_print_j = 0

                self.rot_sim_spec_plot.axes.cla()

                if v != -1:
                    max_print_j = 0
                    for j_ in range(pop.shape[1]):
                        if pop[1,j_] > self.rot_cutoff:
                            self.rot_sim_spec_plot.axes.vlines(j_, 0, pop[1,j_])
                        else:
                            max_print_j = j_
                    self.rot_sim_spec_plot.axes.set_xlabel("J")
                    self.rot_sim_spec_plot.axes.set_ylabel("Population")
                    self.rot_sim_spec_plot.axes.set_xticks(np.arange(0, max(J, max_print_j), 2))
                    self.rot_sim_spec_plot.draw()


                else:
                    print (pop.shape)
                    for v_ in range(pop.shape[2]):
                        for j_ in range(pop.shape[0]):
                            self.rot_sim_spec_plot.axes.text(j_,
                                                             v_,
                                                             round(pop[j_, 1, v_], 3),
                                                             ha='center',
                                                             va='center',
                                                             color=plt.cm.binary(pop[j_, 1, v_]/np.amax(pop[:,1:]))
                                                             )

                    self.rot_sim_spec_plot.axes.set_title("Population for all States")
                    self.rot_sim_spec_plot.axes.set_xlim(-0.5, pop.shape[0] - 0.5)
                    self.rot_sim_spec_plot.axes.set_ylim(-0.5, pop.shape[2] - 0.5)
                    self.rot_sim_spec_plot.axes.set_xlabel("J-state")
                    self.rot_sim_spec_plot.axes.set_ylabel("V-state")
                    self.rot_sim_spec_plot.draw()
                

            elif self.rot_method == 'int':

                val, vec = self.__diagonalize(self.total)

                self.rot_sim_spec_plot.axes.cla()

                inten = sim.SimulatedRotationalInt(J=J,
                                                   v=v,
                                                   val=val,
                                                   vec=vec,
                                                   tdm=self.tdm,
                                                   pop=pop,
                                                   method=method)

                if v != -1:
                    for j_ in range(inten.shape[1]):
                        if abs(inten[1,j_]) > self.rot_cutoff:
                            self.rot_sim_spec_plot.axes.vlines(inten[0,j_], 0, abs(inten[1,j_]))
                else:
                    CHARS = "0123456789ABCDEF"

                    for v in range(inten.shape[2]):
                        leg = False
                        color = "#"
                        for k in range(6):
                            color += CHARS[np.random.randint(0, 15)]
                        for j_ in range(inten.shape[0]):
                            if abs(inten[j_,1,v]) > self.rot_cutoff:
                                if leg == False:
                                    self.rot_sim_spec_plot.axes.vlines(inten[j_,0,v], 0, abs(inten[j_,1,v]), color=color, label='ν=' + str(v))
                                    leg = True
                                else:
                                    self.rot_sim_spec_plot.axes.vlines(inten[j_,0,v], 0, abs(inten[j_,1,v]), color=color)

                    self.rot_sim_spec_plot.axes.legend()

                self.rot_sim_spec_plot.axes.set_xlabel("Energy (cm$^{-1}$)")
                self.rot_sim_spec_plot.axes.set_ylabel("Intensity (s$^{-1}$)")
                self.rot_sim_spec_plot.draw()

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __change_rov_sim_spec(self):
        '''Used to update the plot and data table with the rovibrational states.

           Calculates the population of each state using a Boltzmann distribution and
           uses those values with the Calculated Einstein-A coefficients to determine the
           intensity of a given rovibrational transition'''

        try:
            self.rov_cutoff = float(self.rov_cutoff_val.text())
            val, vec = self.__diagonalize(self.total)

            sim = Spectra()

            J = int(self.rov_sim_spec_j_val.value())
            v = int(self.rov_sim_spec_v_val.value())
            
            pop = sim.SimulatedRovibrationalPop(temp=self.temp,
                                                J=J,
                                                v=v,
                                                vals=val,
                                                )

            if self.rov_method == 'pop':

                self.rov_sim_spec_plot.axes.cla()

                for en in range(pop.shape[1]):
                    if pop[1,en] > self.rov_cutoff:
                        self.rov_sim_spec_plot.axes.vlines(pop[0,en], 0, pop[1,en])

                    self.rov_sim_spec_plot.axes.set_xlabel("Energy (cm$^{-1}$)")
                    self.rov_sim_spec_plot.axes.set_ylabel("Population")
                    self.rov_sim_spec_plot.draw()

            elif self.rov_method == 'int':
                val, vec = self.__diagonalize(self.total)

                self.rov_sim_spec_plot.axes.cla()

                inten = sim.SimulatedRovibrationalInt(vec=vec,
                                                      pop=pop,
                                                      tdm=self.tdm
                                                      )
                
                for en in range(inten.shape[1]):
                    if abs(inten[7,en]) > self.rov_cutoff:
                        self.rov_sim_spec_plot.axes.vlines(inten[6,en], 0, abs(inten[7,en]))

                self.rov_sim_spec_plot.axes.set_xlabel("Energy (cm$^{-1}$)")
                self.rov_sim_spec_plot.axes.set_ylabel("Intensity (s$^{-1}$)")
                self.rov_sim_spec_plot.draw()

                

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()


    def __vib_spec_datatable(self):
        '''Used to update and open an external window for the vibrational data table'''
        try:
            val, vec = self.__diagonalize(self.total)

            sim = Spectra()

            J = int(self.vib_sim_spec_j_val.value())
            V = int(self.vib_sim_spec_v_val.value())

            if J == -1:
                method = 'rov'
            else:
                method = 'vib'

            pop = sim.SimulatedVibrationalPop(temp=self.temp,
                                           J=J,
                                           v=V,
                                           method=method,
                                           vals=val,
                                           )

            inten = sim.SimulatedVibrationalInt(J=J,
                                            v=V,
                                            val=val,
                                            vec=vec,
                                            tdm=self.tdm,
                                            pop=pop,
                                            method=method)

            self.c = VibSpecDataTable(pop=pop,
                                      inten=inten,
                                      v=V,
                                      J=J, 
                                      vib_type=self.vib_method,
                                      method=method)
            self.c.show()

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()


    def __rot_spec_datatable(self):
        '''Used to update and open an external window for the rotational data table'''
        try:
            val, vec = self.__diagonalize(self.total)

            sim = Spectra()

            J = int(self.rot_sim_spec_j_val.value())
            V = int(self.rot_sim_spec_v_val.value())

            if V == -1:
                method = 'rov'
            else:
                method = 'rot'

            pop = sim.SimulatedRotationalPop(temp=self.temp,
                                             J=J,
                                             v=V,
                                             method=method,
                                             vals=val,
                                             )

            inten = sim.SimulatedRotationalInt(J=J,
                                               v=V,
                                               val=val,
                                               vec=vec,
                                               tdm=self.tdm,
                                               pop=pop,
                                               method=method)

            self.c = RotSpecDataTable(pop=pop,
                                      inten=inten,
                                      v=V,
                                      J=J,
                                      rot_type=self.rot_method,
                                      method=method)
            self.c.show()

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()

    def __rov_spec_datatable(self):
        '''Used to update and open an external window for the rovibrational data table'''
        try:
            val, vec = self.__diagonalize(self.total)

            sim = Spectra()

            J = int(self.rov_sim_spec_j_val.value())
            V = int(self.rov_sim_spec_v_val.value())

            pop = sim.SimulatedRovibrationalPop(temp=self.temp,
                                                J=J,
                                                v=V,
                                                vals=val,
                                                )

            inten = sim.SimulatedRovibrationalInt(vec=vec,
                                                  pop=pop,
                                                  tdm=self.tdm
                                                  )

            self.c = RovSpecDataTable(pop=pop,
                                      inten=inten,
                                      v=V,
                                      J=J,
                                      rov_type=self.rov_method)
            self.c.show()

        except Exception as e:
            self.errorText = str(traceback.format_exc())
            self.__openErrorMessage()



def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()
    win.destroy()

if __name__ == '__main__':
    sys.exit(main())
