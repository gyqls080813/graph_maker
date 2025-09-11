import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Draggable Artist Class (Upgraded to handle Lines and Text) ---
class DraggableArtist:
    """
    An artist that can be dragged by the mouse.
    This upgraded version can handle both Text and Line2D artists.
    """
    def __init__(self, artists, update_callback):
        self.artists = artists
        self.update_callback = update_callback
        self.current_artist = None
        self.press_info = None
        if not self.artists:
            return
        for artist in self.artists:
            artist.set_picker(5)  # How close the mouse needs to be to 'pick' the artist
        self.connect()

    def connect(self):
        """Connects the matplotlib event callbacks."""
        self.cid_pick = self.artists[0].figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid_motion = self.artists[0].figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.artists[0].figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_pick(self, event):
        """Callback for when an artist is picked."""
        if event.artist in self.artists:
            self.current_artist = event.artist
            mouse_event = event.mouseevent
            if mouse_event.xdata is None or mouse_event.ydata is None:
                self.current_artist = None
                return

            x0, y0 = mouse_event.xdata, mouse_event.ydata

            # Store information based on the artist type
            if isinstance(self.current_artist, plt.Text):
                self.press_info = (self.current_artist.get_position(), (x0, y0))
            elif isinstance(self.current_artist, plt.Line2D):
                self.press_info = ((self.current_artist.get_xdata(), self.current_artist.get_ydata()), (x0, y0))

    def on_motion(self, event):
        """Callback for when the mouse is moved."""
        if self.current_artist is None or event.inaxes != self.current_artist.axes or self.press_info is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        start_artist_info, (start_x, start_y) = self.press_info
        dx = event.xdata - start_x
        dy = event.ydata - start_y

        # Update artist based on its type
        if isinstance(self.current_artist, plt.Text):
            start_pos = start_artist_info
            self.current_artist.set_position((start_pos[0] + dx, start_pos[1] + dy))
        elif isinstance(self.current_artist, plt.Line2D):
            start_xdata, start_ydata = start_artist_info
            self.current_artist.set_data(start_xdata + dx, start_ydata + dy)

        self.current_artist.figure.canvas.draw_idle()

    def on_release(self, event):
        """Callback for when the mouse button is released."""
        if self.current_artist is None:
            return

        # Only save the new position for Text objects to persist changes.
        # Dragging the trendline is a visual-only adjustment that resets on redraw.
        if isinstance(self.current_artist, plt.Text):
            new_pos = self.current_artist.get_position()
            self.update_callback(self.current_artist, new_pos)

        self.current_artist = None
        self.press_info = None

    def disconnect(self):
        """Disconnects the matplotlib event callbacks."""
        if not self.artists or not hasattr(self, 'cid_pick'):
            return
        self.artists[0].figure.canvas.mpl_disconnect(self.cid_pick)
        self.artists[0].figure.canvas.mpl_disconnect(self.cid_motion)
        self.artists[0].figure.canvas.mpl_disconnect(self.cid_release)


# --- Data Selector Window (Drag Fixed) ---
class DataSelector(tk.Toplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.title("Select Data Rows")
        self.geometry("700x500")
        self.df = df
        self.parent = parent
        self.start_row_idx = None

        info_label = ttk.Label(self, text="Click and drag to select rows.", font=("Helvetica", 10, "italic"))
        info_label.pack(pady=5)
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.tree = ttk.Treeview(tree_frame, show="headings", selectmode="extended")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        self._populate_tree()

        apply_button = ttk.Button(self, text="Apply Selection", command=self.apply_selection)
        apply_button.pack(pady=10)

        self.tree.bind("<ButtonPress-1>", self.on_press)
        self.tree.bind("<B1-Motion>", self.on_drag)

    def _populate_tree(self):
        # 0) 기존 내용 초기화
        self.tree.delete(*self.tree.get_children())
        # 행 번호 컬럼을 맨 앞에 추가
        data_cols = list(self.df.columns)
        columns = ["#"] + data_cols

        self.tree["columns"] = columns
        self.tree["show"] = "headings"

        # 1) 헤더 설정 + 기본 폭
        for col in columns:
            self.tree.heading(col, text=col, anchor="w")
            # 폭은 일단 최소값, 뒤에서 샘플로 재조정
            self.tree.column(col, anchor="w", width=80, stretch=False)

        # 2) 문자열 변환 유틸 (NaN → "", float 깔끔 포맷)
        def to_str(v):
            if pd.isna(v):
                return ""
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)

        # 3) 폭 계산: 헤더 길이 vs 샘플 데이터(최대 200행) 길이
        sample = self.df.head(200)
        widths_chars = {"#": max(3, len(str(len(self.df))))}  # 행 수 자리수만큼
        for col in data_cols:
            max_len = len(col)
            if not sample.empty:
                max_len = max(max_len, *(len(to_str(x)) for x in sample[col]))
            # 너무 길면 줄이고, 너무 짧으면 최소 보장
            widths_chars[col] = min(30, max(6, max_len))

        # Tkinter 픽셀 환산(대략 글자폭 7px + 패딩)
        for col in columns:
            self.tree.column(col, width=int(widths_chars[col] * 7 + 16))

        # 4) 행 삽입 (배치로 성능 개선)
        batch, B = [], 1000
        # itertuples(index=False)로 빠르게 순회
        for one_based_idx, row_vals in enumerate(self.df.itertuples(index=False, name=None), 1):
            display_vals = (one_based_idx,) + tuple(to_str(v) for v in row_vals)
            iid = str(one_based_idx - 1)  # 기존 로직과 호환(0-based 문자열)
            batch.append((iid, display_vals))
            if len(batch) >= B:
                for _iid, _vals in batch:
                    self.tree.insert("", "end", iid=_iid, values=_vals)
                batch.clear()
        # 남은 것 플러시
        for _iid, _vals in batch:
            self.tree.insert("", "end", iid=_iid, values=_vals)

    def on_press(self, event):
        self.tree.selection_set()
        iid = self.tree.identify_row(event.y)
        if iid:
            self.start_row_idx = self.tree.index(iid)
            self.tree.selection_add(iid)

    def on_drag(self, event):
        if self.start_row_idx is None:
            return
        iid = self.tree.identify_row(event.y)
        if iid:
            end_idx = self.tree.index(iid)
            self.tree.selection_set()
            min_idx, max_idx = sorted([self.start_row_idx, end_idx])
            for i in range(min_idx, max_idx + 1):
                self.tree.selection_add(self.tree.get_children()[i])

    def apply_selection(self):
        selections = self.tree.selection()
        if not selections:
            self.destroy()
            return
        row_indices = sorted([int(s) for s in selections])
        self.parent.start_row.set(str(row_indices[0] + 1))
        self.parent.end_row.set(str(row_indices[-1] + 1))
        self.destroy()


# --- Data & Style Classes ---
class StyleConfig:
    def __init__(self, size='12', weight='normal', style='normal', color='black', linestyle='solid', width='1'):
        self.size = tk.StringVar(value=size)
        self.weight = tk.StringVar(value=weight)
        self.style = tk.StringVar(value=style)
        self.color = tk.StringVar(value=color)
        self.linestyle = tk.StringVar(value=linestyle)
        self.width = tk.StringVar(value=width)

    def get_font_dict(self):
        return {
            'fontsize': self._parse_float(self.size.get(), 12),
            'fontweight': self.weight.get(),
            'fontstyle': self.style.get(),
            'color': self.color.get()
        }

    def get_line_dict(self):
        return {
            'linestyle': self.linestyle.get(),
            'color': self.color.get(),
            'linewidth': self._parse_float(self.width.get(), 1)
        }

    def _parse_float(self, value, default):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


class StyleableObject:
    def __init__(self, name, obj_type, style_config, text_var=None, x_var=None, y_var=None):
        self.name = name
        self.obj_type = obj_type
        self.style = style_config
        self.text = text_var
        self.x = x_var
        self.y = y_var


# --- Main Application ---
class GraphMaker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Maker - Final Version")
        self.geometry("1400x850")  # Increased width

        # Matplotlib defaults
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['axes.unicode_minus'] = False

        # State
        self.df, self.figure, self.canvas = None, None, None
        self.x_col, self.y_col = tk.StringVar(), tk.StringVar()
        self.start_row, self.end_row = tk.StringVar(), tk.StringVar()
        self.x_min, self.x_max = tk.StringVar(), tk.StringVar()
        self.y_min, self.y_max = tk.StringVar(), tk.StringVar()
        self.x_interval, self.y_interval = tk.StringVar(), tk.StringVar()
        self.show_x_grid = tk.BooleanVar(value=True)
        self.show_y_grid = tk.BooleanVar(value=True)
        self.show_trendline = tk.BooleanVar(value=False)
        self.show_equation = tk.BooleanVar(value=False)
        self.fit_intercept = tk.BooleanVar(value=True)

        # Graph size inputs (cm)
        self.figure_width_cm = tk.StringVar(value="15")
        self.figure_height_cm = tk.StringVar(value="10")
        self.size_target = tk.StringVar(value="Entire Figure")

        # Preview controls
        self.screen_dpi = 100                           # Preview DPI
        self.preview_exact = tk.BooleanVar(value=True)  # Exact-size preview toggle

        # Styling registry
        self.styleable_objects = {}
        self.annotations_counter = 0
        self.dragger = None
        self.artist_map = {}
        self._init_style_objects()
        self._create_widgets()
        self.plot_graph()  # Initial plot

    def _init_style_objects(self):
        self.styleable_objects["Title"] = StyleableObject(
            "Title", 'title',
            StyleConfig(size='16', weight='bold'),
            text_var=tk.StringVar(value="Generated Graph")
        )
        self.styleable_objects["X-Axis Label"] = StyleableObject(
            "X-Axis Label", 'ax_label', StyleConfig(), text_var=tk.StringVar()
        )
        self.styleable_objects["Y-Axis Label"] = StyleableObject(
            "Y-Axis Label", 'ax_label', StyleConfig(), text_var=tk.StringVar()
        )
        self.styleable_objects["X-Tick Labels"] = StyleableObject("X-Tick Labels", 'tick', StyleConfig())
        self.styleable_objects["Y-Tick Labels"] = StyleableObject("Y-Tick Labels", 'tick', StyleConfig())
        self.styleable_objects["Trendline"] = StyleableObject("Trendline", 'line', StyleConfig(color='red', linestyle='--', width='2'))
        self.styleable_objects["Regression Equation"] = StyleableObject(
            "Regression Equation", 'text',
            StyleConfig(color='red'),
            text_var=tk.StringVar(), x_var=tk.DoubleVar(), y_var=tk.DoubleVar()
        )

    # For Entry binds
    def _plot_graph_event(self, event=None):
        self.plot_graph()

    def _mount_canvas(self):
        widget = self.canvas.get_tk_widget()
        widget.pack_forget()  # clear previous packing

        if self.preview_exact.get():
            # figure inches × preview DPI = widget pixels
            fig_w_in, fig_h_in = self.figure.get_size_inches()
            dpi = self.figure.dpi
            w_px = int(fig_w_in * dpi)
            h_px = int(fig_h_in * dpi)
            widget.config(width=w_px, height=h_px)
            widget.pack(anchor="center", padx=10, pady=10)  # no fill/expand -> exact size
        else:
            widget.pack(fill="both", expand=True)

    def _create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Inspector with scroll
        inspector_container = ttk.Frame(main_frame)
        inspector_container.grid(row=0, column=1, sticky="ns", pady=10)

        inspector_canvas = tk.Canvas(inspector_container, width=380)
        inspector_scrollbar = ttk.Scrollbar(inspector_container, orient="vertical", command=inspector_canvas.yview)
        self.inspector_frame = ttk.Frame(inspector_canvas)
        self.inspector_frame.bind("<Configure>", lambda e: inspector_canvas.configure(scrollregion=inspector_canvas.bbox("all")))

        inspector_canvas.create_window((0, 0), window=self.inspector_frame, anchor="nw")
        inspector_canvas.configure(yscrollcommand=inspector_scrollbar.set)

        inspector_scrollbar.pack(side="right", fill="y")
        inspector_canvas.pack(side="left", fill="both", expand=True)

        self._populate_inspector()

    def _populate_inspector(self):
        for widget in self.inspector_frame.winfo_children():
            widget.destroy()

        # 1. Data
        file_frame = ttk.LabelFrame(self.inspector_frame, text="1. Data", padding=10)
        file_frame.pack(fill="x", pady=5, padx=5)
        ttk.Button(file_frame, text="Load Excel File", command=self.load_excel).pack(fill="x")
        self.file_label = ttk.Label(file_frame, text="No file selected.", wraplength=300)
        self.file_label.pack(pady=5)
        self.data_selector_button = ttk.Button(file_frame, text="Select Rows Visually...", command=self.open_data_selector, state="disabled")
        self.data_selector_button.pack(fill="x")
        ttk.Label(file_frame, text="X-Axis Column:").pack(anchor="w", pady=(5, 0))
        self.x_col_menu = ttk.OptionMenu(file_frame, self.x_col, "Select X-Column")
        self.x_col_menu.pack(fill="x")
        ttk.Label(file_frame, text="Y-Axis Column:").pack(anchor="w", pady=(5, 0))
        self.y_col_menu = ttk.OptionMenu(file_frame, self.y_col, "Select Y-Column")
        self.y_col_menu.pack(fill="x")
        ttk.Label(file_frame, text="Start Data Row:").pack(anchor="w", pady=(5, 0))
        ttk.Entry(file_frame, textvariable=self.start_row).pack(fill="x")
        ttk.Label(file_frame, text="End Data Row:").pack(anchor="w", pady=(5, 0))
        ttk.Entry(file_frame, textvariable=self.end_row).pack(fill="x")

        # 2. Axis Scale
        scale_frame = ttk.LabelFrame(self.inspector_frame, text="2. Axis Scale", padding=10)
        scale_frame.pack(fill="x", pady=5, padx=5)
        self._populate_scale_tab(scale_frame)

        # 3. Graph Size
        size_frame = ttk.LabelFrame(self.inspector_frame, text="3. Graph Size (cm)", padding=10)
        size_frame.pack(fill="x", pady=5, padx=5)
        self._populate_size_tab(size_frame)

        # 4. Formatting
        format_frame = ttk.LabelFrame(self.inspector_frame, text="4. Formatting", padding=10)
        format_frame.pack(fill="x", pady=5, padx=5)
        self._populate_formatting_tab(format_frame)

        # 5. Styles
        obj_frame = ttk.LabelFrame(self.inspector_frame, text="5. Edit Styles", padding=10)
        obj_frame.pack(fill="x", pady=5, padx=5)
        ttk.Label(obj_frame, text="Select object to edit:").pack(anchor="w")
        self.inspector_combo = ttk.Combobox(obj_frame, values=list(self.styleable_objects.keys()), state="readonly")
        self.inspector_combo.pack(fill="x")
        self.inspector_combo.bind("<<ComboboxSelected>>", self.on_inspector_select)
        self.dynamic_style_frame = ttk.Frame(obj_frame, padding=(0, 10))
        self.dynamic_style_frame.pack(fill="x")

        # 6. Actions
        action_frame = ttk.LabelFrame(self.inspector_frame, text="6. Actions", padding=10)
        action_frame.pack(fill="x", pady=5, padx=5)
        ttk.Button(action_frame, text="Add Text to Graph", command=self.add_annotation).pack(fill="x", pady=2)
        ttk.Button(action_frame, text="Generate Graph", command=self.plot_graph, style="Accent.TButton").pack(fill="x", pady=2)
        ttk.Button(action_frame, text="Save Graph", command=self.save_graph).pack(fill="x", pady=2)

        self.x_col.trace_add("write", self.on_column_change)
        self.y_col.trace_add("write", self.on_column_change)

    def _populate_scale_tab(self, parent):
        ttk.Label(parent, text="X-Min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.x_min, width=8).grid(row=0, column=1)
        ttk.Label(parent, text="X-Max:").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Entry(parent, textvariable=self.x_max, width=8).grid(row=0, column=3)
        ttk.Label(parent, text="X-Interval:").grid(row=0, column=4, sticky="w", padx=5)
        ttk.Entry(parent, textvariable=self.x_interval, width=8).grid(row=0, column=5)
        ttk.Label(parent, text="Y-Min:").grid(row=1, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.y_min, width=8).grid(row=1, column=1)
        ttk.Label(parent, text="Y-Max:").grid(row=1, column=2, sticky="w", padx=5)
        ttk.Entry(parent, textvariable=self.y_max, width=8).grid(row=1, column=3)
        ttk.Label(parent, text="Y-Interval:").grid(row=1, column=4, sticky="w", padx=5)
        ttk.Entry(parent, textvariable=self.y_interval, width=8).grid(row=1, column=5)

    def _populate_size_tab(self, parent):
        ttk.Label(parent, text="Size Target:").grid(row=0, column=0, sticky="w", pady=2)
        size_combo = ttk.Combobox(parent, textvariable=self.size_target,
                                  values=["Entire Figure", "Inner Plot"], width=15, state="readonly")
        size_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)
        self.size_target.trace_add("write", lambda *args: self.plot_graph())

        ttk.Label(parent, text="Width (cm):").grid(row=1, column=0, sticky="w", pady=2)
        width_entry = ttk.Entry(parent, textvariable=self.figure_width_cm, width=8)
        width_entry.grid(row=1, column=1, pady=2)

        ttk.Label(parent, text="Height (cm):").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        height_entry = ttk.Entry(parent, textvariable=self.figure_height_cm, width=8)
        height_entry.grid(row=1, column=3, pady=2)

        # Update on Enter/FocusOut
        width_entry.bind("<Return>", self._plot_graph_event)
        width_entry.bind("<FocusOut>", self._plot_graph_event)
        height_entry.bind("<Return>", self._plot_graph_event)
        height_entry.bind("<FocusOut>", self._plot_graph_event)

        # Exact size preview toggle
        ttk.Checkbutton(
            parent,
            text="Exact size preview (no stretch)",
            variable=self.preview_exact,
            command=self.plot_graph
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

    def _populate_formatting_tab(self, parent):
        ttk.Checkbutton(parent, text="Show X-Axis Grid", variable=self.show_x_grid, command=self.plot_graph).pack(anchor="w")
        ttk.Checkbutton(parent, text="Show Y-Axis Grid", variable=self.show_y_grid, command=self.plot_graph).pack(anchor="w")
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=5)
        ttk.Checkbutton(parent, text="Show Trendline", variable=self.show_trendline, command=self.plot_graph).pack(anchor="w")
        ttk.Checkbutton(parent, text="Show Equation", variable=self.show_equation, command=self.plot_graph).pack(anchor="w")
        ttk.Checkbutton(parent, text="Fit Intercept (y=mx+c)", variable=self.fit_intercept, command=self.plot_graph).pack(anchor="w")

    def on_inspector_select(self, event):
        for widget in self.dynamic_style_frame.winfo_children():
            widget.destroy()
        selection = self.inspector_combo.get()
        if not selection:
            return
        obj = self.styleable_objects.get(selection)
        if not obj:
            return
        if obj.text is not None:
            ttk.Label(self.dynamic_style_frame, text="Text:").pack(anchor="w", pady=(10, 0))
            entry = ttk.Entry(self.dynamic_style_frame, textvariable=obj.text)
            entry.pack(fill="x")
            entry.bind("<Return>", self._plot_graph_event)
        if obj.x is not None and obj.y is not None:
            pos_frame = ttk.Frame(self.dynamic_style_frame)
            pos_frame.pack(fill="x", pady=5)
            ttk.Label(pos_frame, text="X-Pos:").pack(side="left")
            x_entry = ttk.Entry(pos_frame, textvariable=obj.x, width=8)
            x_entry.pack(side="left")
            x_entry.bind("<Return>", self._plot_graph_event)
            ttk.Label(pos_frame, text="Y-Pos:").pack(side="left", padx=(10, 0))
            y_entry = ttk.Entry(pos_frame, textvariable=obj.y, width=8)
            y_entry.pack(side="left")
            y_entry.bind("<Return>", self._plot_graph_event)
        if obj.obj_type == 'line':
            self._create_line_style_widgets(self.dynamic_style_frame, obj.style)
        else:
            self._create_font_style_widgets(self.dynamic_style_frame, obj.style)

    def _create_font_style_widgets(self, parent, config):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text="Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=config.size, width=8).grid(row=0, column=1)
        ttk.Label(frame, text="Color:").grid(row=0, column=2, sticky="w", padx=5)
        entry = ttk.Entry(frame, textvariable=config.color, width=8)
        entry.grid(row=0, column=3)
        ttk.Button(frame, text="...", width=2, command=lambda: self._choose_color(config.color)).grid(row=0, column=4)
        ttk.Label(frame, text="Weight:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=config.weight, values=['normal', 'bold'], width=6, state="readonly").grid(row=1, column=1)
        ttk.Label(frame, text="Style:").grid(row=1, column=2, sticky="w", padx=5)
        ttk.Combobox(frame, textvariable=config.style, values=['normal', 'italic'], width=6, state="readonly").grid(row=1, column=3)
        for var in [config.size, config.weight, config.style, config.color]:
            var.trace_add("write", lambda *args: self.plot_graph())

    def _create_line_style_widgets(self, parent, config):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text="Width:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=config.width, width=8).grid(row=0, column=1)
        ttk.Label(frame, text="Color:").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=config.color, width=8).grid(row=0, column=3)
        ttk.Button(frame, text="...", width=2, command=lambda: self._choose_color(config.color)).grid(row=0, column=4)
        ttk.Label(frame, text="Line Style:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=config.linestyle, values=['solid', 'dashed', 'dotted', 'dashdot'], width=10, state="readonly").grid(row=1, column=1, columnspan=3, sticky="ew")
        for var in [config.width, config.linestyle, config.color]:
            var.trace_add("write", lambda *args: self.plot_graph())

    def _choose_color(self, color_var):
        color_code = colorchooser.askcolor(title="Choose color")
        if color_code:
            color_var.set(color_code[1])

    def add_annotation(self):
        if not self.figure:
            return
        self.annotations_counter += 1
        name = f"Custom Text {self.annotations_counter}"
        ax = self.figure.axes[0]
        center_x, center_y = np.mean(ax.get_xlim()), np.mean(ax.get_ylim())
        new_ann = StyleableObject(
            name, 'text', StyleConfig(),
            text_var=tk.StringVar(value="New Text"),
            x_var=tk.DoubleVar(value=round(center_x, 2)),
            y_var=tk.DoubleVar(value=round(center_y, 2))
        )
        self.styleable_objects[name] = new_ann
        self.inspector_combo['values'] = list(self.styleable_objects.keys())
        self.inspector_combo.set(name)
        self.on_inspector_select(None)
        self.plot_graph()

    def open_data_selector(self):
        if self.df is None:
            return
        selector = DataSelector(self, self.df)
        self.wait_window(selector)

    def load_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if not file_path:
            return
        try:
            self.df = pd.read_excel(file_path)
        except Exception as e:
            self.file_label.config(text=f"Error: {e}")
            return
        self.file_label.config(text="Loaded: " + file_path.split('/')[-1])
        self.data_selector_button.config(state="normal")
        self.start_row.set("1")
        self.end_row.set(str(len(self.df)))
        columns = list(self.df.columns)
        self.x_col_menu['menu'].delete(0, 'end')
        self.y_col_menu['menu'].delete(0, 'end')
        for col in columns:
            self.x_col_menu['menu'].add_command(label=col, command=tk._setit(self.x_col, col))
            self.y_col_menu['menu'].add_command(label=col, command=tk._setit(self.y_col, col))
        if columns:
            self.x_col.set(columns[0])
            self.y_col.set(columns[1] if len(columns) > 1 else columns[0])
        self.on_column_change()

    def _calculate_axis_defaults(self, data_min, data_max):
        if data_min is None or data_max is None or pd.isna(data_min) or pd.isna(data_max):
            return "0", "10", "2"
        nice_min = 0
        range_val = data_max - nice_min
        if range_val <= 0:
            range_val = data_max
        power = 10 ** math.floor(math.log10(range_val if range_val > 0 else 1))
        nice_max = math.ceil(data_max / power) * power
        if nice_max <= nice_min:
            nice_max = data_max + power
        interval = (nice_max - nice_min) / 10.0
        if interval <= 0:
            interval = 1.0
        return str(nice_min), str(round(nice_max, 2)), str(round(interval, 2))

    def on_column_change(self, *args):
        if self.df is None or not self.x_col.get() or not self.y_col.get():
            return
        self.styleable_objects["X-Axis Label"].text.set(self.x_col.get())
        self.styleable_objects["Y-Axis Label"].text.set(self.y_col.get())
        try:
            plotting_df = self._get_plotting_df()
            if plotting_df.empty:
                return
            x_min_val, x_max_val, x_interval_val = self._calculate_axis_defaults(
                plotting_df[self.x_col.get()].min(), plotting_df[self.x_col.get()].max())
            y_min_val, y_max_val, y_interval_val = self._calculate_axis_defaults(
                plotting_df[self.y_col.get()].min(), plotting_df[self.y_col.get()].max())
            self.x_min.set(x_min_val); self.x_max.set(x_max_val); self.x_interval.set(x_interval_val)
            self.y_min.set(y_min_val); self.y_max.set(y_max_val); self.y_interval.set(y_interval_val)
        except (KeyError, TypeError, IndexError):
            pass
        self.plot_graph()

    def _get_plotting_df(self):
        if self.df is None:
            return pd.DataFrame()
        try:
            start = int(self.start_row.get()); end = int(self.end_row.get())
        except (ValueError, TypeError):
            return self.df
        start_index = max(0, start - 1)
        end_index = min(len(self.df), end)
        return self.df.iloc[start_index:end_index]

    def update_object_position(self, artist, pos):
        name = self.artist_map.get(artist)
        if name:
            obj = self.styleable_objects[name]
            if obj.x and obj.y:
                obj.x.set(round(pos[0], 2))
                obj.y.set(round(pos[1], 2))

    def plot_graph(self, *args):
        # disconnect dragger
        if self.dragger:
            self.dragger.disconnect()
            self.dragger = None

        # destroy old canvas widget
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        # --- Graph Size Logic ---
        try:
            target_w_cm = float(self.figure_width_cm.get())
            target_h_cm = float(self.figure_height_cm.get())
            cm_to_inch = 2.54
            target_w_in = max(target_w_cm / cm_to_inch, 2.0)  # min 2 inches
            target_h_in = max(target_h_cm / cm_to_inch, 2.0)

            # Preview DPI + constrained layout
            self.figure, ax = plt.subplots(
                figsize=(target_w_in, target_h_in),
                dpi=self.screen_dpi,
                constrained_layout=True
            )
        except (ValueError, TypeError):
            self.figure, ax = plt.subplots(dpi=self.screen_dpi, constrained_layout=True)

        # No data yet
        if self.df is None:
            ax.text(0.5, 0.5, "Please load an Excel file.", ha='center')
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.draw()
            self._mount_canvas()
            return

        plotting_df = self._get_plotting_df()
        if plotting_df.empty or not self.x_col.get() or not self.y_col.get():
            ax.text(0.5, 0.5, "No data to plot.", ha='center')
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.draw()
            self._mount_canvas()
            return

        # Plot data
        try:
            x_data = plotting_df[self.x_col.get()].astype(float)
            y_data = plotting_df[self.y_col.get()].astype(float)
            ax.plot(x_data, y_data, 'ks--', linewidth=1.5, markersize=4, label='Data')
        except (KeyError, ValueError) as e:
            ax.text(0.5, 0.5, f"Error plotting data:\n{e}", ha='center', color='red')
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.draw()
            self._mount_canvas()
            return

        # Titles & labels
        title_obj = self.styleable_objects["Title"]
        xlabel_obj = self.styleable_objects["X-Axis Label"]
        ylabel_obj = self.styleable_objects["Y-Axis Label"]
        xtick_obj = self.styleable_objects["X-Tick Labels"]
        ytick_obj = self.styleable_objects["Y-Tick Labels"]
        ax.set_title(title_obj.text.get(), **title_obj.style.get_font_dict())
        ax.set_xlabel(xlabel_obj.text.get(), **xlabel_obj.style.get_font_dict())
        ax.set_ylabel(ylabel_obj.text.get(), **ylabel_obj.style.get_font_dict())

        # Tick styles
        xtick_styles = xtick_obj.style.get_font_dict()
        ax.tick_params(axis='x', labelsize=xtick_styles['fontsize'], labelcolor=xtick_styles['color'])
        plt.setp(ax.get_xticklabels(), fontweight=xtick_styles['fontweight'], fontstyle=xtick_styles['fontstyle'])
        ytick_styles = ytick_obj.style.get_font_dict()
        ax.tick_params(axis='y', labelsize=ytick_styles['fontsize'], labelcolor=ytick_styles['color'])
        plt.setp(ax.get_yticklabels(), fontweight=ytick_styles['fontweight'], fontstyle=ytick_styles['fontstyle'])

        # Axis limits & intervals
        try:
            x_min = float(self.x_min.get()); x_max = float(self.x_max.get())
            y_min = float(self.y_min.get()); y_max = float(self.y_max.get())
            x_interval = float(self.x_interval.get()); y_interval = float(self.y_interval.get())
            ax.set_xlim(left=x_min, right=x_max)
            ax.set_ylim(bottom=y_min, top=y_max)
            if x_interval > 0:
                ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
            if y_interval > 0:
                ax.yaxis.set_major_locator(mticker.MultipleLocator(y_interval))
        except (ValueError, TypeError):
            pass

        # Trendline & equation
        draggable_artists = []
        self.artist_map = {}
        trendline_artist = None

        if self.show_trendline.get() and not x_data.empty:
            try:
                trendline_obj = self.styleable_objects["Trendline"]
                equation_obj = self.styleable_objects["Regression Equation"]
                if self.fit_intercept.get():
                    m, c = np.polyfit(x_data, y_data, 1)
                    equation = f'y = {m:.2f}x + {c:.2f}'
                else:
                    x_reshaped = x_data.values.reshape(-1, 1)
                    m_tuple = np.linalg.lstsq(x_reshaped, y_data, rcond=None)[0]
                    m = m_tuple[0]; c = 0
                    equation = f'y = {m:.2f}x'
                x_trend = np.array([x_data.min(), x_data.max()])
                y_trend = m * x_trend + c
                trendline_artist, = ax.plot(x_trend, y_trend, **trendline_obj.style.get_line_dict(), label='Trendline')
                if self.show_equation.get():
                    equation_obj.text.set(equation)
                    if not equation_obj.x.get() and not equation_obj.y.get():
                        equation_obj.x.set(x_data.mean())
                        equation_obj.y.set(y_data.mean())
            except (np.linalg.LinAlgError, ValueError):
                pass

        if trendline_artist:
            draggable_artists.append(trendline_artist)
            self.artist_map[trendline_artist] = "Trendline"

        # Annotations (text)
        for name, obj in self.styleable_objects.items():
            if obj.obj_type == 'text':
                if self.show_equation.get() or name != "Regression Equation":
                    text_artist = ax.text(obj.x.get(), obj.y.get(), obj.text.get(), **obj.style.get_font_dict())
                    draggable_artists.append(text_artist)
                    self.artist_map[text_artist] = name

        # Grids
        ax.grid(False)
        if self.show_x_grid.get():
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        if self.show_y_grid.get():
            ax.yaxis.grid(True, linestyle='--', alpha=0.5)

        # Inner-plot size targeting (optional)
        if self.size_target.get() == "Inner Plot":
            try:
                ax_bbox = ax.get_position()  # fraction [0..1]
                new_fig_w_in = target_w_in / ax_bbox.width
                new_fig_h_in = target_h_in / ax_bbox.height
                self.figure.set_size_inches(new_fig_w_in, new_fig_h_in, forward=True)
            except (ZeroDivisionError, ValueError, TypeError):
                pass

        # Mount canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self._mount_canvas()

        # Enable dragging
        if draggable_artists:
            self.dragger = DraggableArtist(draggable_artists, self.update_object_position)

    def save_graph(self):
        if not self.figure:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", '*.png'),
                ("JPEG", '*.jpg'),
                ("SVG (Vector)", "*.svg"),
                ("PDF (Vector)", "*.pdf"),
                ("EPS (Vector)", "*.eps")
            ])
        if file_path:
            # 정확한 픽셀/인치 보장을 위해 bbox_inches='tight' 제거
            self.figure.savefig(file_path, dpi=300)


if __name__ == "__main__":
    app = GraphMaker()
    app.mainloop()
