#!/usr/bin/env python

import re
import os
import sys
import datetime

from itertools import groupby
from functools import partial
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore, Qt

ROUND_DECIMALS = 4
ICONS = {
    'Search': 'search_symbolic.png',
    'Analysis': 'analysis_symbolic.png',
    'Excel_': 'excel.png',
    'Excel_dim': 'excel_dimmed.png',
    'Copies': 'emblem-documents.png',
    'Refresh': 'view-refresh.png',
}

ICONS = {name: os.path.join('ICONS', icon) for name, icon in ICONS.items()}

messages_levels = ['Message', 'Comment', 'Warning', 'Problem', 'Error', 'Bug']


def button(btn_txt, btn_func, btn_tooltip, btn_status=False, pass_txt=False,
           icon=None, icon_size=64):
    '''
    return a QPushButton with all the passed options
    '''
    btn = QtWidgets.QPushButton(btn_txt)
    if pass_txt:
        btn.clicked.connect(partial(btn_func, btn_text=btn_txt))
    else:
        btn.clicked.connect(btn_func)
    if icon:
        btn.setIcon(QtGui.QIcon(icon))
        btn.setIconSize(QtCore.QSize(icon_size, icon_size))
    btn.setToolTip(btn_tooltip)
    btn.setDisabled(btn_status)
    btn.setMinimumSize(btn.sizeHint())
    return btn


class toolButtonHover(QtWidgets.QToolButton):
    hovered = QtCore.pyqtSignal()
    hoveredout = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()

    def enterEvent(self, event):
        self.hovered.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hoveredout.emit()
        super().enterEvent(event)


def read_input_file():
    with open(sys.argv[1], 'r') as f_obj:
        input_file = f_obj.read()

    results = []
    input_file = '\n'.join(l.strip() for l in input_file.splitlines()
                  if l.strip().startswith('@'))


    # sep = '('+'|'.join(f'@--{m.upper()}' for m in messages_levels)+')'
    # input_file = re.split(sep, input_file)
    input_file = input_file.split('@--')

    input_file = [m for m in input_file if m != '']

    # types = [input_file[i][3:] for i in range(0, len(input_file), 2)]
    # messages = [input_file[x] for x in range(1, len(input_file), 2)]
    types = [m.split()[0] for m in input_file]
    dates = [re.search(r'DAYS\s*\((.*)\)', m.splitlines()[0]).group(1) for m in input_file]
    dates = [datetime.datetime.strptime(
        date.replace('JLY', 'JUL'), '%d-%b-%Y').date() for date in dates]
    messages = ['\n'.join(l.lstrip('@').strip('*').strip() for l in m.splitlines()[1:]) for m in input_file]
    messages = ['\n'.join(l for l in m.splitlines() if l) for m in messages]
    df = pd.DataFrame({'Date': dates, 'Type': types, 'Message': messages})
    return df


class AdvanceSearchDialog(QtWidgets.QDialog):
    """
    Filter tableview by column
    """

    hints_str = '''
<h1> HINTS: </h1>
<p>
<big><b># Column</b></big><br>
Pick the column to apply rule on, you may pick the same column in more than one rule
<br>
<big><b># Type</b></big>
<br>
Select the type of check you want to perform
<br>
<b> + Equals</b>
The value in the table must match exactly
<br>
<b> + Contains</b> 
The table column contains this values
<br>
<b> + Larger/Less than</b> 
For comparing against numbers, works with dates too, date has to be in the format: dd-mm-YYYY (i.e. 01-01-2022)
<br>
<b> + Starts/Ends with</b> 
The table column field starts or ends with the specified text, works with numbers and dates if you need that
<br>
<big><b># Value</b></big>
<br>
The value comparison specified in the type column will occur against
<br>
        
<big><b># AND/OR</b></big>
<br>
Relation of this rule with the next, AND means both need to be true, OR means only one need to be true to show item
<br>

</p>
    '''

    def __init__(self, df, fil_rules=None):
        super().__init__()
        self.df = df
        self.fil_rules = fil_rules
        self.initUI()
        self.setWindowTitle('Advance Search')

    def initUI(self):
        vboxs = [QtWidgets.QVBoxLayout() for _ in range(2)]
        hboxs = [QtWidgets.QHBoxLayout() for _ in range(3)]
        splitter = QtWidgets.QSplitter()

        hints_lbl = QtWidgets.QLabel(self.hints_str)
        hints_lbl.setWordWrap(True)
        hints_lbl.setProperty('hintsLabel', True)
        hint_scrollarea = QtWidgets.QScrollArea()
        hint_scrollarea.setWidgetResizable(True)
        hint_scrollarea.setWidget(hints_lbl)

        save_search_btn = button('Save Search', self.save_filter, 'Save current search to external file')
        load_search_btn = button('Load Search', self.load_filter, 'Load a saved search')
        apply_btn = button('Apply', self.apply_filter, 'Apply selected filter and return to table')
        clear_btn = button('Clear', self.clear_filters, 'Clear previous filters and return to table')
        plus_btn = button(' + ', self.new_filter, 'Add a new filter')
        minus_btn = button(' - ', self.del_filter, 'Remove selected filter')

        self.filter_table = QtWidgets.QTableWidget(1, 4)
        self.fil_types = ['Equals', 'Contains', 'Less Than', 'Larger Than', 'Starts with', 'Ends with']
        self.fil_dict = {'Column': self.df.columns.tolist(), 'Type': self.fil_types, 'Value': [], 'OR/AND': ['AND', 'OR']}
        self.filter_table.setHorizontalHeaderLabels(list(self.fil_dict.keys()))
        self.setup_row(0)
        if self.fil_rules is not None:
            self.fil_df = self.fil_rules
            self.write_table()

        hboxs[0].addWidget(load_search_btn)
        hboxs[0].addWidget(save_search_btn)
        hboxs[0].addStretch()
        hboxs[0].addWidget(apply_btn)
        hboxs[0].addWidget(clear_btn)

        vboxs[0].addStretch()
        vboxs[0].addWidget(plus_btn)
        vboxs[0].addWidget(minus_btn)

        hboxs[1].addWidget(self.filter_table)
        hboxs[1].addLayout(vboxs[0])
        splitter_wdgt = QtWidgets.QWidget()
        splitter_wdgt.setLayout(hboxs[1])
        splitter.addWidget(splitter_wdgt)
        splitter.addWidget(hint_scrollarea)
        splitter.setCollapsible(1, True)
        splitter_width = splitter.size().width()
        ratio = 0.7
        splitter.setSizes([splitter_width*ratio, splitter_width*(1-ratio)])

        vboxs[1].addWidget(splitter)
        vboxs[1].addLayout(hboxs[0])

        self.setLayout(vboxs[1])
        self.setGeometry(self.x(), self.y(), splitter.sizeHint().width()*1.2, splitter.sizeHint().height()*1.2)

    def save_filter(self):
        if not os.path.exists('SavedSearches'):
            os.makedirs('SavedSearches')
        fp = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Search', 'SavedSearches', 'Saved Search (*.savedsearch)')
        if fp is None:
            return
        if fp[0].strip() == '' or fp[1].strip() == '':
            return
        self.read_table()
        with open(fp[0], 'wb+') as f_obj:
            pickle.dump(self.fil_df, f_obj)

    def load_filter(self):
        if not os.path.exists('SavedSearches'):
            os.makedirs('SavedSearches')
        fp = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Saved Search', 'SavedSearches', 'Saved Search (*.savedsearch)')
        if fp is None:
            return
        if fp[0].strip() == '' or fp[1].strip() == '':
            return
        with open(fp[0], 'rb') as f_obj:
            self.fil_df = pickle.load(f_obj)
        self.write_table()

    def apply_filter(self):
        df_fil = ([True]*len(self.df))
        prv_relation = 'AND'
        self.read_table()
        for col, fil_type, vlu, relation in self.fil_df.itertuples(index=False, name=None):
            if fil_type == 'Equals':
                tmp_fil = self.df[col].astype(str) == vlu
            elif fil_type == 'Contains':
                tmp_fil = self.df[col].astype(str).str.contains(vlu, case=False)
            elif fil_type == 'Less Than':
                if 'Date' in col:
                    try:
                        tmp_fil = self.df[col] < datetime.datetime.strptime(vlu, '%d-%m-%Y').date()
                    except Exception as e:
                        print(e)
                        prv_relation = relation
                        continue
                else:
                    try:
                        tmp_fil = self.df[col].apply(try_float) < try_float(vlu)
                    except TypeError:
                        QtWidgets.QMessageBox.warning(self, 'Rule Error', f'Error in rule for column {col}, the rule will be ignored')
                        prev_relation = relation
                        continue
            elif fil_type == 'Larger Than':
                if 'Date' in col:
                    try:
                        tmp_fil = self.df[col] > datetime.datetime.strptime(vlu, '%d-%m-%Y').date()
                    except Exception as e:
                        print(e)
                        prv_relation = relation
                        continue
                else:
                    try:
                        tmp_fil = self.df[col].apply(try_float) > try_float(vlu)
                    except TypeError:
                        QtWidgets.QMessageBox.warning(self, 'Rule Error', f'Error in rule for column {col}, the rule will be ignored')
                        prev_relation = relation
                        continue
            elif fil_type == 'Starts with':
                tmp_fil = self.df[col].astype(str).str.startswith(vlu)
            elif fil_type == 'Ends with':
                tmp_fil = self.df[col].astype(str).str.endswith(vlu)
            if prv_relation == 'AND':
                df_fil = df_fil & tmp_fil
            elif prv_relation == 'OR':
                df_fil = df_fil | tmp_fil
            prv_relation = relation
        self.filtered_df = self.df[df_fil]
        self.accept()

    def clear_filters(self):
        self.filtered_df = self.df
        self.fil_df = None
        self.reject()

    def setup_row(self, row):
        columns_combo = QtWidgets.QComboBox()
        columns_combo.addItems(self.fil_dict['Column'])
        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(self.fil_types)
        value_edit = QtWidgets.QLineEdit()
        orand_combo = QtWidgets.QComboBox()
        orand_combo.addItems(self.fil_dict['OR/AND'])

        for c, wdgt in enumerate([columns_combo, type_combo, value_edit, orand_combo]):
            self.filter_table.setCellWidget(row, c, wdgt)

    def new_filter(self):
        self.filter_table.setRowCount(self.filter_table.rowCount()+1)
        self.setup_row(self.filter_table.rowCount()-1)

    def write_table(self):
        self.filter_table.clearContents()
        self.filter_table.setRowCount(len(self.fil_df))
        for row in range(len(self.fil_df)):
            self.setup_row(row)
            for col in range(self.filter_table.columnCount()):
                col_name = list(self.fil_dict.keys())[col]
                wdgt = self.filter_table.cellWidget(row, col)
                vlu = self.fil_df.iloc[row, col]
                if isinstance(wdgt, QtWidgets.QDateEdit):
                    wdgt.setDate(vlu)
                elif isinstance(wdgt, QtWidgets.QComboBox):
                    wdgt.setCurrentIndex(self.fil_dict[col_name].index(vlu))
                else:
                    wdgt.setText(str(vlu))

    def read_table(self):
        rowcount = self.filter_table.rowCount()
        colcount = self.filter_table.columnCount()
        fil_cols = list(self.fil_dict.keys())
        self.fil_df = pd.DataFrame(columns=fil_cols)
        for row in range(rowcount):
            for col, matcol in zip(list(range(colcount)), fil_cols):
                wdgt = self.filter_table.cellWidget(row, col)
                if not isinstance(wdgt, (QtWidgets.QDateEdit, QtWidgets.QComboBox)):
                    self.fil_df.loc[row, matcol] = wdgt.text()
                elif isinstance(wdgt, QtWidgets.QComboBox):
                    text = wdgt.currentText()
                    self.fil_df.loc[row, matcol] = text
                else:
                    date_obj = wdgt.date()
                    date_obj = datetime.datetime(
                        date_obj.year(), date_obj.month(), date_obj.day()).date()
                    self.fil_df.loc[row, matcol] = date_obj
        self.fil_df.index = range(len(self.fil_df))

    def del_filter(self):
        if self.filter_table.rowCount() == 1:
            QtWidgets.QMessageBox.critical(
                self, 'Cannot proceed',
                'There have to be one filter at least')
            return
        cur_row = self.filter_table.currentRow()
        if cur_row == -1:
            QtWidgets.QMessageBox.warning(self, 'No Row Selected', 'No row selected, nothing will be removed')
            return
        self.read_table()
        self.fil_df.drop(cur_row, inplace=True)
        self.fil_df.index = range(len(self.fil_df))
        # repopulate table
        self.write_table()
        self.filter_table.setCurrentCell(max(0, cur_row-1), 0)

    def exec_(self):
        super(AdvanceSearchDialog, self).exec_()


class tableView(QtWidgets.QWidget):
    def __init__(self, df=None, **kwargs):
        super().__init__()
        self.df = df
        if not self.df.empty:
            self.df.sort_values(self.df.columns[0], ascending=False, inplace=True)
        self.table_widget = tableWidget(df, **kwargs)
        self.sort_hist = [(self.df.columns[0], False),] if not self.df.empty else []
        self.advanced_search = False
        self.fil_rls = None
        self.initUI()

    def initUI(self):
        vbox = QtWidgets.QVBoxLayout()
        self.hbox = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText('Enter search term here')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.sort_fil_df)

        self.setup_toolbar_actions()
        self.setup_toolbar()

        self.table_widget.table.horizontalHeader().sectionClicked.connect(self.clicked_sort_hist)
        self.table_widget.table.currentCellChanged.connect(self.update_status)
        self.table_widget.table.cellActivated.connect(self.update_status)
        self.table_widget.table.cellClicked.connect(self.update_status)
        self.table_widget.table.setShowGrid(False)

        self.hbox.addWidget(self.search)
        self.status_bar = QtWidgets.QStatusBar()
        vbox.addLayout(self.hbox)
        hbox2 = QtWidgets.QHBoxLayout()
        hbox2.addWidget(self.toolbar)
        hbox2.addWidget(self.table_widget)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.status_bar)
        self.initialize_stats_box()
        self.populate_stats_box()
        self.setLayout(vbox)

    def initialize_stats_box(self):
        stats_hbox = QtWidgets.QHBoxLayout(self.table_widget.table)
        self.stats_frame = QtWidgets.QFrame()
        stats_hbox.addStretch()
        self.stats_frame.setMaximumWidth(0)
        stats_hbox.addWidget(self.stats_frame)
        self.stats_frame.setStyleSheet('background: rgba(246, 242, 249, 85%); color: Black;')
        stats_vbox = QtWidgets.QVBoxLayout()
        self.stats_lbl = QtWidgets.QLabel()
        self.stats_lbl.setStyleSheet('padding: 8px; margin: 15px; font-family: monospace')
        self.stats_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(self.stats_lbl)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet('background: transparent')
        stats_vbox.addWidget(scroll_area)
        self.stats_frame.setLayout(stats_vbox)

    def __prepare_html_stats(self, df):
        ignored_patterns = [
            'ID', 'Sales Person', 'SGAN', 'Discount (%)',
            'Ammendments', 'Intended For', 'Notes', 'Warning Level']
        stats_txt = ''
        for col in df.columns:
            if any(x in col for x in ignored_patterns):
                continue
            stats_txt += f'<br><big><b>{col}</b></big>'
            describe_df = df[col].describe()
            if 'mean' in describe_df.index:
                describe_df.rename({'50%': 'median'}, inplace=True)
                describe_df['sum'] = describe_df['count'] * describe_df['mean']
            describe_df = describe_df.reset_index()
            float_fmt = lambda x: f'{x:.4f}'
            stats_txt += describe_df.to_html(
                index=False, header=False, border=0, float_format=float_fmt)
            stats_txt += '<br>'
        return stats_txt

    def populate_stats_box(self):
        if self.df.empty:
            return
        fil_df = self.get_fil_df()

        stats_txt = '<h2><b>All Data</b></h2><br><br>'
        stats_txt += self.__prepare_html_stats(self.df)
        if fil_df is not None and len(fil_df) == len(self.df):
            self.stats_lbl.setText(stats_txt)
            return
        stats_txt += '<br><h2><b>Filtered Data</b></h2><br><br>'
        stats_txt += self.__prepare_html_stats(fil_df)
        self.stats_lbl.setText(stats_txt)

    def display_stats(self):
        self.populate_stats_box()
        if hasattr(self, 'stats_animation'):
            self.stats_animation.stop()
        width = self.stats_frame.width()
        new_width = min((self.table_widget.table.width()*0.4,
                         self.stats_lbl.sizeHint().width()*1.2))
        new_width = new_width if width == 0 else 0

        self.stats_animation = QtCore.QPropertyAnimation(self.stats_frame, b"maximumWidth")
        self.stats_animation.setDuration(150)
        self.stats_animation.setStartValue(width)
        self.stats_animation.setEndValue(new_width)
        self.stats_animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.stats_animation.start()

    def setup_toolbar_actions(self):
        self.btns = [
            [('Refresh', self.refresh_from_file, 'Reread PRT file',
              ICONS['Refresh'])],
            [('Advance Search', self.advance_search,
              'Filter the table with more control', ICONS['Search']),
             ('Statistics', self.display_stats,
              'Display various statistical analysis of the data in this table',
              ICONS['Analysis'])],
            [('To Excel', self.export_df,
              'Export table to Excel file for sharing or analyzing',
              ICONS['Excel_']),
             ('Fil To Excel', self.export_fil_df,
              'Export the visible (filtered) table to Excel file for sharing or analyzing',
              ICONS['Excel_dim']),
             ('Copy Rows', self.selected_to_clipboard,
              'Copy the selected rows to clipboard', ICONS['Copies']),
             ],
        ]

    def selected_to_clipboard(self):
        if not self.table_widget.table.selectedRanges():
            # if no selection
            QtWidgets.QMessageBox.information(self, 'Nothing Selected', 'No rows were selected, nothing was copied to clipboard')
            return
        locs = []
        for selection_range in self.table_widget.table.selectedRanges():
            # when using ctrl for selection
            toprow = selection_range.topRow()
            botrow = selection_range.bottomRow()
            locs += list(range(toprow, botrow+1))
        fil_df = self.get_fil_df().iloc[locs]
        fil_df.to_clipboard(index=False)

    def resize_toolbar(self, hovered):
        if hasattr(self, 'animation'):
            self.animation.stop()
        width = self.toolbar.width()
        new_width = 126 if hovered else 48

        self.animation = QtCore.QPropertyAnimation(self.toolbar, b"maximumWidth")
        self.animation.setDuration(150)
        self.animation.setStartValue(width)
        self.animation.setEndValue(new_width)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()
        print(self.toolbar.width())

    def setup_toolbar(self):
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setOrientation(Qt.Qt.Vertical)
        # self.toolbar.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
        self.toolbar.setFont(Qt.QFont('sans', 7))
        icn_size = 24
        icn_size = QtCore.QSize(icn_size, icn_size)
        self.toolbar.setIconSize(icn_size)
        self.toolbar.setMaximumWidth(48)
        btn_width = 120

        for grp in self.btns:
            for i, btn in enumerate(grp):
                btn_obj = toolButtonHover()
                btn_obj.setText(btn[0])
                btn_obj.clicked.connect(btn[1])
                btn_obj.hovered.connect(lambda: self.resize_toolbar(True))
                btn_obj.hoveredout.connect(lambda: self.resize_toolbar(False))
                btn_obj.setIcon(QtGui.QIcon(btn[3]))
                base_style = 'border-width: 1px; border-bottom-width:0px; border-radius: 0px; font: 8pt;'
                btn_obj.setStyleSheet(base_style)
                radius = '3px;'
                if i == 0:
                    btn_obj.setStyleSheet(f'{base_style}; border-top-left-radius: {radius} border-top-right-radius: {radius}')
                if i == len(grp)-1:
                    btn_obj.setStyleSheet(f'{base_style}; border-width: 1px; border-bottom-right-radius: {radius} border-bottom-left-radius: {radius}')
                btn_obj.setToolButtonStyle(Qt.Qt.ToolButtonTextBesideIcon)
                btn_obj.setMaximumWidth(btn_width*2)
                self.toolbar.addWidget(btn_obj)
            self.toolbar.addSeparator()

    def advance_search(self):
        self.fil_rls = None if not hasattr(self, 'fil_rls') else self.fil_rls
        dlg = AdvanceSearchDialog(self.df, self.fil_rls)
        dlg.exec_()
        if dlg.result() == 1:
            self.advanced_search = True
            self.advance_filtered_df = dlg.filtered_df
            self.fil_rls = dlg.fil_df
            self.sort_fil_df()
        else:
            self.advanced_search = False
            self.fil_rls = None
            self.sort_fil_df()

    def export_df(self, btn_text=None):
        fp = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export Table', '', 'Excel Files (*.xlsx)')
        if fp is None:
            return
        if fp[0].strip() == '' or fp[1].strip() == '':
            return
        fp = fp[0] if fp[0].endswith('.xlsx') else fp[0] + '.xlsx'
        self.df.to_excel(fp)

    def export_fil_df(self, btn_text=None):
        fp = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export Table', '', 'Excel Files (*.xlsx)')
        if fp is None:
            return
        if fp[0].strip() == '' or fp[1].strip() == '':
            return
        fp = fp[0] if fp[0].endswith('.xlsx') else fp[0] + '.xlsx'
        fil_df = self.get_fil_df()
        fil_df.to_excel(fp, index=False)

    def update_status(self, *args):
        # if no selection
        if not self.table_widget.table.selectedRanges():
            return
        locs = []
        for selection_range in self.table_widget.table.selectedRanges():
            # when using ctrl for selection
            toprow = selection_range.topRow()
            botrow = selection_range.bottomRow()
            leftcol = selection_range.leftColumn()
            rightcol = selection_range.rightColumn()
            locs += list((c, r) for r in range(toprow, botrow+1) for c in
                         range(leftcol, rightcol+1))
        min_col = min(c for c, r in locs)
        rows = [r for c, r in locs if c == min_col]
        first_col = self.df.columns[min_col]
        series = self.get_fil_df()[first_col].iloc[rows]
        describe = series.describe(percentiles=[])
        if 'mean' in describe.index:
            describe.rename({'50%': 'median'}, inplace=True)
            describe['sum'] = describe['count'] * describe['mean']
        try_round = lambda v: v if not isinstance(v, float) else round(v, ROUND_DECIMALS)
        statusrow = ' | '.join(f'{i.title()}: {try_round(v)}' for i, v in describe.items())
        self.status_bar.showMessage(statusrow)

    def refresh_from_file(self):
        df = read_input_file()
        self.update_df(df)

    def update_df(self, df):
        self.df = df
        self.sort_hist = [(self.df.columns[0], False),] if not self.df.empty else []
        self.table_widget.update_df(self.df)
        self.table_widget.load_df()
        self.sort_fil_df()

    def sort_history_load_df(self, sort_col, sort_order):
        self.update_sort_hist(sort_col, sort_order)
        self.sort_fil_df()

    def clicked_sort_hist(self, sort_col):
        sort_col = self.df.columns[sort_col]
        sort_order = False
        if (sort_col, sort_order) in self.sort_hist:
            self.sort_hist.remove((sort_col, sort_order))
            sort_order = True
        elif (sort_col, not sort_order) in self.sort_hist:
            self.sort_hist.remove((sort_col, not sort_order))
        self.sort_hist.insert(0, (sort_col, sort_order))
        sort_len = min(len(self.sort_hist), len(self.df.columns))
        self.sort_hist = self.sort_hist[:sort_len]
        self.sort_fil_df()

    def get_fil_df(self):
        '''
        Taking care of multiindexes and named indexes, filter the table
        dataframe using the search box and sort it according to table
        sort history
        '''
        if self.df.empty:
            return
        search_txt = self.search.text()
        multiindex_flag = False
        if isinstance(self.df.index, pd.MultiIndex):
            multiindex_flag = True
            index_cols = list(self.df.index.names)
            self.df.reset_index(inplace=True)
        if not self.advanced_search:
            fil_df = self.df[self.df.stack().astype(str).str.contains(
                search_txt, case=False, na=False, regex=False).any(level=0)].sort_values(
                    by=[s[0] for s in self.sort_hist],
                    ascending=[s[1] for s in self.sort_hist])
        else:
            fil_df = self.advance_filtered_df
        if multiindex_flag:
            fil_df.set_index(index_cols, inplace=True)
            self.df.set_index(index_cols, inplace=True)
        return fil_df

    def sort_fil_df(self, text=None):
        if self.df.empty:
            return
        fil_df = self.get_fil_df()
        self.table_widget.update_df(fil_df)
        self.table_widget.load_df()


class tableWidget(QtWidgets.QWidget):
    def __init__(self, df=None, include_index=False, merge_single_index=False, color_func=None, smart_alternate=False):
        super().__init__()
        self.df = df
        self.include_index = include_index
        self.merge_single_index = merge_single_index
        self.smart_alternate = smart_alternate
        self.initUI()

    def initUI(self):
        vbox = QtWidgets.QVBoxLayout()
        no_columns = len(self.df.columns)
        self.multiindex = isinstance(self.df.index, pd.MultiIndex)
        if self.multiindex and self.include_index:
            no_columns += len(self.df.index.levels)
        elif self.include_index:
            no_columns += 1
        self.table = QtWidgets.QTableWidget(len(self.df), no_columns)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.load_df()
        self.table.setAlternatingRowColors(not self.smart_alternate)
        vbox.addWidget(self.table)
        self.setLayout(vbox)

    def smart_alternating(self):
        if self.df.empty or len(self.df) == 0:
            return
        if not self.df.iloc[:, 0].duplicated().any():
            self.table.setAlternatingRowColors(True)
            return

        colors_dict = {
            'Type': {
                'DARK': {
                    'WARNING': 'Yellow', 'COMMENT': 'Green',
                    'MESSAGE': 'Turquoise', 'PROBLEM': 'Orange', 'ERROR': 'Red'},
                'LIGHT': {
                    'WARNING': 'Yellow', 'COMMENT': 'Green',
                    'MESSAGE': 'Turquoise', 'PROBLEM': 'Orange',
                    'ERROR': 'Red'}},
             }
        default_colors = {'DARK': '#282828', 'LIGHT': '#474747'}
        prev_ref_no = self.df.iloc[0, 0]

        i = 1
        for row, (_, ref_no) in enumerate(self.df.iloc[:, 0].iteritems()):
            if ref_no != prev_ref_no:
                i += 1
            bg_variant = 'DARK' if i % 2 == 0 else 'LIGHT' 
            # bg_color = 'GainsBoro' if i % 2 == 0 else 'white'
            # old_bg_color = bg_color
            # print(i, bg_color, ref_no, prev_ref_no)
            prev_ref_no = ref_no
            for col, col_name in enumerate(self.df.columns):
                item = self.table.item(row, col)
                default_bg_color = default_colors[bg_variant]
                bg_color = default_bg_color
                if col_name in colors_dict.keys():
                    bg_color = colors_dict[col_name][bg_variant].get(
                        item.text(), default_bg_color)
                bg_color = QtGui.QColor(bg_color)
                # item.setData(Qt.Qt.BackgroundRole, bg_color)
                item.setBackground(bg_color)
        self.table.update()
    
    def update_df(self, df):
        self.df = df

    def smart_cols(self):
        mat_col = [i for i, c in enumerate(self.df.columns) if c == 'Material']
        mat_col = mat_col[0] if mat_col else len(self.df.columns)+2
        self.table.resizeColumnsToContents()
        if self.table.columnCount() < 3:
            return
        widths = {c: self.table.columnWidth(c)+1 for c in
                  range(self.table.columnCount())}
        sorted_keys = sorted(widths, key=lambda k: widths[k], reverse=True)
        width = widths[sorted_keys[0]]
        for c in sorted_keys[:1]:
            if c != mat_col:
                self.table.setColumnWidth(c, width)
        self.table.resizeRowsToContents()

    def load_df(self):
        self.table.clearContents()
        cols = []
        if self.include_index:
            self.table.verticalHeader().hide()
        if self.multiindex and self.include_index:
            # cols = self.df.index.to_frame().columns.tolist()
            cols = list(self.df.index.names)
            shift = len(cols)
        elif self.include_index:
            shift = 1
            index_col = self.df.index.name
            index_col = 'Index' if not index_col else index_col
            cols = [index_col]
        else:
            shift = 0
        cols += self.df.columns.tolist()

        self.table.setHorizontalHeaderLabels(cols)

        if len(cols) != self.table.columnCount():
            self.table.setColumnCount(len(cols))
        if len(self.df) != self.table.rowCount():
            self.table.setRowCount(len(self.df))
        for idx, row in enumerate(self.df.iterrows()):
            items = [QtWidgets.QTableWidgetItem() for _ in range(len(row[1]))]
            for item, vlu in zip(items, row[1]):
                # this will enable nums sorting
                if isinstance(vlu, (datetime.date, pd.Timestamp, datetime.datetime)):
                    item.setData(Qt.Qt.DisplayRole, str(vlu))
                else:
                    item.setData(Qt.Qt.DisplayRole, vlu)
            for col, item in enumerate(items):
                self.table.setItem(idx, col + shift, item)

        if self.multiindex and self.include_index:
            idxs = []
            for level, code in enumerate(self.df.index.codes):
                temp = list(self.df.index.levels[level][i] for i in code)
                idxs.append(list((i, len(list(l))) for i, l in groupby(temp)))
            for col, idx in enumerate(idxs):
                row = 0
                for data in idx:
                    if data[1] > 1:
                        self.table.setSpan(row, col, data[1], 1)
                    self.table.setItem(
                        row, col, QtWidgets.QTableWidgetItem(str(data[0])))
                    row += data[1]
        elif self.include_index and self.merge_single_index:
            idxs = [(idx, len(list(l))) for idx, l in groupby(self.df.index)]
            for row, data in enumerate(idxs):
                if data[1] > 1:
                    self.table.setSpan(row, 0, data[1], 1)
                self.table.setItem(
                    row, 0, QtWidgets.QTableWidgetItem(str(data[0])))
                row += data[1]
        elif self.include_index:
            for i, idx in enumerate(self.df.index):
                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(idx)))
        self.smart_cols()
        if self.smart_alternate:
            self.smart_alternating()


class tableViewDlg(QtWidgets.QDialog):
    def __init__(self, df=None, win_title='Table Viewer', **kwargs):
        super().__init__()
        self.tableview = tableView(df=df, **kwargs)
        self.setGeometry(
            self.x(), self.y(),
            self.tableview.table_widget.table.width() * 1.2,
            self.height() * 1.2)
        self.setWindowTitle(win_title)
        self.vbox = QtWidgets.QVBoxLayout()
        self.addTable()
        self.setWindowFlag(Qt.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.Qt.WindowMaximizeButtonHint, True)
        self.setWindowState(Qt.Qt.WindowMaximized)
        self.setModal(True)

    def addTable(self):
        self.vbox.addWidget(self.tableview)
        self.setLayout(self.vbox)

    def exec_(self):
        super(tableViewDlg, self).exec_()


def main():
    '''
    Main entry point of the program
    '''
    df = read_input_file()
    viewer = tableView(df, smart_alternate=True)
    viewer.show()




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.Qt.AA_UseHighDpiPixmaps)
    main()
    sys.exit(app.exec_())
