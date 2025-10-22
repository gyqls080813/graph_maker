import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
import pandas as pd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# Bio-Logic .mpr 지원(있으면 사용, 없으면 폴백)
try:
    from biologic import eclabfiles as _eclab  # pip install biologic-toolbox
except Exception:
    _eclab = None

# ------------------------------- 
# Draggable for Text / Lines
# ------------------------------- 
class DraggableArtist:
    """
    텍스트/선(Line2D) 드래그 지원.
    - 축이 datetime이어도 안전하게 동작하도록 date2num/num2date로 변환하여 이동.
    """
    def __init__(self, artists, update_callback):
        self.artists = artists or []
        self.update_callback = update_callback
        self.current_artist = None
        self.press_info = None
        if not self.artists:
            return
        for a in self.artists:
            try:
                a.set_picker(5)  # 집기 반경
            except Exception:
                pass
        self.connect()

    # ---------- helpers (datetime 처리) ----------
    @staticmethod
    def _is_dt_value(v):
        import numpy as np
        import datetime as _dt
        return isinstance(v, (_dt.datetime, _dt.date, np.datetime64))

    @staticmethod
    def _is_dt_array(arr):
        import numpy as np
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.datetime64):
            return True
        if a.dtype == object and a.size > 0:
            from datetime import date, datetime
            return isinstance(a.flat[0], (datetime, date))
        return False

    @staticmethod
    def _to_num_value(v):
        # 단일 값 -> float(날짜번호)
        from matplotlib import dates as mdates
        import numpy as np
        import pandas as pd
        if isinstance(v, np.datetime64):
            v = pd.to_datetime(v).to_pydatetime()
        return mdates.date2num(v)

    @staticmethod
    def _to_num_array(arr):
        # 배열 -> float 배열(날짜번호)
        from matplotlib import dates as mdates
        import numpy as np
        import pandas as pd
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.datetime64):
            pydt = pd.to_datetime(a).to_pydatetime()
            return mdates.date2num(pydt)
        # object 배열(파이썬 datetime)도 처리
        return mdates.date2num(list(a))

    @staticmethod
    def _num_to_dt_value(xnum):
        # float(날짜번호) -> datetime
        from matplotlib import dates as mdates
        return mdates.num2date(xnum)

    @staticmethod
    def _num_to_dt_array(xnums):
        # float 배열(날짜번호) -> datetime 배열
        from matplotlib import dates as mdates
        import numpy as np
        return np.asarray(mdates.num2date(xnums))

    # ---------- mpl events ----------
    def connect(self):
        fig = self.artists[0].figure
        self.cid_pick = fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_pick(self, event):
        import numpy as np
        import matplotlib.pyplot as plt

        if event.artist not in self.artists:
            return

        self.current_artist = event.artist
        me = event.mouseevent
        if me.xdata is None or me.ydata is None:
            self.current_artist = None
            return

        sx, sy = me.xdata, me.ydata

        # 텍스트
        if isinstance(self.current_artist, plt.Text):
            x0, y0 = self.current_artist.get_position()
            x_dt = self._is_dt_value(x0)
            y_dt = self._is_dt_value(y0)
            x0n = self._to_num_value(x0) if x_dt else float(x0)
            y0n = self._to_num_value(y0) if y_dt else float(y0)
            self.press_info = ({
                "kind": "text",
                "x0": x0n, "y0": y0n,
                "x_dt": x_dt, "y_dt": y_dt,
            }, (sx, sy))

        # 선(Line2D)
        elif isinstance(self.current_artist, plt.Line2D):
            xdata = self.current_artist.get_xdata()
            ydata = self.current_artist.get_ydata()

            x_dt = self._is_dt_array(xdata)
            y_dt = self._is_dt_array(ydata)

            x0n = self._to_num_array(xdata) if x_dt else np.asarray(xdata, dtype=float)
            y0n = self._to_num_array(ydata) if y_dt else np.asarray(ydata, dtype=float)

            self.press_info = ({
                "kind": "line",
                "x0": x0n, "y0": y0n,
                "x_dt": x_dt, "y_dt": y_dt,
            }, (sx, sy))

    def on_motion(self, event):
        if self.current_artist is None or self.press_info is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        info, (sx, sy) = self.press_info
        dx = event.xdata - sx
        dy = event.ydata - sy

        kind = info["kind"]
        if kind == "text":
            xn = info["x0"] + dx
            yn = info["y0"] + dy
            x_set = self._num_to_dt_value(xn) if info["x_dt"] else xn
            y_set = self._num_to_dt_value(yn) if info["y_dt"] else yn
            self.current_artist.set_position((x_set, y_set))

        elif kind == "line":
            xarr = info["x0"] + dx
            yarr = info["y0"] + dy
            if info["x_dt"]:
                xarr = self._num_to_dt_array(xarr)
            if info["y_dt"]:
                yarr = self._num_to_dt_array(yarr)
            self.current_artist.set_data(xarr, yarr)

        self.current_artist.figure.canvas.draw_idle()

    def on_release(self, event):
        import matplotlib.pyplot as plt
        if self.current_artist is None:
            return
        # 위치 저장 콜백은 텍스트일 때만(좌표 보존 UI와 연계)
        if isinstance(self.current_artist, plt.Text):
            self.update_callback(self.current_artist, self.current_artist.get_position())
        self.current_artist = None
        self.press_info = None

    def disconnect(self):
        if not self.artists or not hasattr(self, 'cid_pick'):
            return
        fig = self.artists[0].figure
        fig.canvas.mpl_disconnect(self.cid_pick)
        fig.canvas.mpl_disconnect(self.cid_motion)
        fig.canvas.mpl_disconnect(self.cid_release)


# ------------------------------- 
# Data Selector: Excel-like drag (row+column rectangle)
# ------------------------------- 
# ------------------------------- 
# Data Selector: Excel-like drag (row+column rectangle)
# ------------------------------- 
class DataSelector(tk.Toplevel):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        self.title("Select Rows & Columns")
        self.geometry("980x560")
        self.parent = parent
        self.df = df

        self._dragging = False
        self._scroll_job = None
        self._edge_px = 24

        self.start_row_idx = None
        self.start_col_idx = None
        self.sel_row_lo = None
        self.sel_row_hi = None
        self.sel_col_lo = None
        self.sel_col_hi = None

        self.assign_first_as_x = tk.BooleanVar(value=True)  # 첫 선택을 X, 나머지 Y

        # Paned
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: table
        left = ttk.Frame(self.paned_window)
        self.paned_window.add(left, weight=1)
        ttk.Label(left, text="Drag like Excel to select a rectangle (rows & columns)").pack(anchor="w", padx=4)
        tf = ttk.Frame(left); tf.pack(fill="both", expand=True, pady=(4,8))
        self.tree = ttk.Treeview(tf, show="headings", selectmode="extended")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tf, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y"); hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # Right: status + options + apply
        right = ttk.Frame(self.paned_window, width=200)
        self.paned_window.add(right, weight=0)
        self.status = ttk.Label(right, text="Selected: (none)", wraplength=160, justify="left")
        self.status.pack(anchor="n", pady=(2,8), fill="x", expand=True)

        ttk.Checkbutton(
            right,
            text="Use first selected\ncolumn as X",
            variable=self.assign_first_as_x
        ).pack(anchor="w", pady=(0,8))

        ttk.Button(right, text="Apply Selection", command=self.apply_selection).pack(fill="x", side="bottom")

        # build table
        self._populate_tree()

        # Bind mouse drag
        self.tree.bind("<ButtonPress-1>", self.on_press)
        self.tree.bind("<B1-Motion>", self.on_drag)
        self.tree.bind("<ButtonRelease-1>", self.on_release)

    # ---------- utils ----------
    def _colid_to_index(self, colid: str):
        try:
            return int(colid.replace('#',''))
        except Exception:
            return None

    def _selected_data_slice(self):
        """header idx (#1..#N) -> data_cols slice [lo:hi_excl]; #1은 index '#'"""
        if self.sel_col_lo is None or self.sel_col_hi is None:
            return (0, 0)
        H_lo = min(self.sel_col_lo, self.sel_col_hi)
        H_hi = max(self.sel_col_lo, self.sel_col_hi)

        # clamp to data headers (#2..#len(columns))
        H_lo = max(2, H_lo)
        H_hi = min(len(self.columns), H_hi)
        if H_lo > H_hi:
            return (0, 0)

        lo = H_lo - 2
        hi_excl = H_hi - 1
        lo = max(0, min(lo, len(self.data_cols)))
        hi_excl = max(lo, min(hi_excl, len(self.data_cols)))
        return (lo, hi_excl)

    # ---------- table ----------
    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        self.data_cols = [str(c) for c in self.df.columns]
        self.columns = ["#"] + self.data_cols
        self.tree["columns"] = self.columns
        self.tree["show"] = "headings"

        self._heading_texts = {f"#{i+1}": col for i, col in enumerate(self.columns)}

        # headers
        for i, col in enumerate(self.columns):
            self.tree.heading(f"#{i+1}", text=col, anchor="w")
            self.tree.column(f"#{i+1}", anchor="w", width=120, stretch=False)

        # rows
        for one_based_idx, row_vals in enumerate(self.df.itertuples(index=False, name=None), 1):
            display_vals = (one_based_idx,) + tuple("" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v)) for v in row_vals)
            self.tree.insert("", "end", iid=str(one_based_idx-1), values=display_vals)

    # ---------- mouse ----------
    def on_press(self, event):
        row_iid = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if not row_iid or not col_id:
            return
        self._dragging = True
        self._last_event_x = event.x
        self._last_event_y = event.y
        self.tree.selection_set()
        self.start_row_idx = self.tree.index(row_iid)
        self.start_col_idx = self._colid_to_index(col_id)
        self.sel_row_lo = self.start_row_idx
        self.sel_row_hi = self.start_row_idx
        self.sel_col_lo = self.start_col_idx
        self.sel_col_hi = self.start_col_idx
        self._update_visual_selection()
        self._ensure_auto_scroll()

    def on_drag(self, event):
        if self.start_row_idx is None or self.start_col_idx is None:
            return
        self._last_event_x = event.x
        self._last_event_y = event.y
        row_iid = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if row_iid:
            end_idx = self.tree.index(row_iid)
            self.sel_row_lo, self.sel_row_hi = sorted([self.start_row_idx, end_idx])
        if col_id:
            end_col = self._colid_to_index(col_id)
            if end_col is not None:
                self.sel_col_lo, self.sel_col_hi = sorted([self.start_col_idx, end_col])
        self._update_visual_selection()
        self._ensure_auto_scroll()

    def on_release(self, event):
        self._dragging = False
        if self._scroll_job is not None:
            try: self.after_cancel(self._scroll_job)
            except Exception: pass
            self._scroll_job = None

    def _ensure_auto_scroll(self):
        if not self._dragging: return
        if self._scroll_job is None:
            self._scroll_job = self.after(30, self._auto_scroll_tick)

    def _auto_scroll_tick(self):
        self._scroll_job = None
        if not self._dragging: return
        try:
            h = self.tree.winfo_height()
            w = self.tree.winfo_width()
        except Exception:
            return
        y = getattr(self, "_last_event_y", 0)
        x = getattr(self, "_last_event_x", 0)
        edge = self._edge_px
        did_scroll = False
        if y < edge:
            self.tree.yview_scroll(-1, "units"); did_scroll = True
        elif y > h - edge:
            self.tree.yview_scroll(1, "units"); did_scroll = True
        if x < edge:
            try: self.tree.xview_scroll(-1, "units"); did_scroll = True
            except Exception: pass
        elif x > w - edge:
            try: self.tree.xview_scroll(1, "units"); did_scroll = True
            except Exception: pass
        if did_scroll:
            row_iid = self.tree.identify_row(y)
            col_id = self.tree.identify_column(x)
            if row_iid:
                end_idx = self.tree.index(row_iid)
                self.sel_row_lo, self.sel_row_hi = sorted([self.start_row_idx, end_idx])
            if col_id:
                end_col = self._colid_to_index(col_id)
                if end_col is not None:
                    self.sel_col_lo, self.sel_col_hi = sorted([self.start_col_idx, end_col])
            self._update_visual_selection()
        if self._dragging and (did_scroll or y < edge or y > h - edge or x < edge or x > w - edge):
            self._scroll_job = self.after(30, self._auto_scroll_tick)

    # ---------- UI update ----------
    def _update_visual_selection(self):
        self.tree.selection_set()
        if self.sel_row_lo is not None and self.sel_row_hi is not None:
            for i in range(self.sel_row_lo, self.sel_row_hi + 1):
                try:
                    self.tree.selection_add(self.tree.get_children()[i])
                except Exception:
                    pass

        sel_col_start, sel_col_end = -1, -1
        if self.sel_col_lo is not None and self.sel_col_hi is not None:
            sel_col_start, sel_col_end = self.sel_col_lo, self.sel_col_hi

        for i, _ in enumerate(self.columns):
            col_id = f"#{i+1}"
            base_text = self._heading_texts.get(col_id, self.columns[i])
            disp_text = f"★ {base_text}" if sel_col_start <= (i+1) <= sel_col_end and i >= 1 else base_text
            self.tree.heading(col_id, text=disp_text)

        lo, hi_excl = self._selected_data_slice()
        sel_names = self.data_cols[lo:hi_excl]
        cols_str = ", ".join(sel_names) if sel_names else "(none)"
        rows_str = "(none)"
        if self.sel_row_lo is not None and self.sel_row_hi is not None:
            rows_str = f"{self.sel_row_lo+1}..{self.sel_row_hi+1}"

        if self.assign_first_as_x.get() and sel_names:
            preview = f"X = {sel_names[0]}\nY = {', '.join(sel_names[1:]) or '(none)'}"
        else:
            preview = "(no auto mapping)"

        self.status.config(text=f"Selected rows:\n{rows_str}\n\nSelected cols:\n{cols_str}\n\nMapping:\n{preview}")

    # ---------- apply ----------
    def apply_selection(self):
        if self.sel_row_lo is not None and self.sel_row_hi is not None:
            self.parent.start_row.set(str(self.sel_row_lo + 1))
            self.parent.end_row.set(str(self.sel_row_hi + 1))

        lo, hi_excl = self._selected_data_slice()
        self.parent.selected_columns = self.data_cols[lo:hi_excl]

        # 자동 매핑
        if self.assign_first_as_x.get():
            try: self.parent._map_columns_from_selection()
            except Exception: pass

        self.destroy()


class HeaderRowDialog(tk.Toplevel):
    """헤더 후보 행을 사용자에게 선택받는 다이얼로그."""

    def __init__(self, parent, df: pd.DataFrame, recommended: int):
        super().__init__(parent)
        self.title("Select Header Row")
        self.geometry("820x520")
        self.transient(parent)
        self.grab_set()
        self.result = None
        self.recommended = max(0, min(int(recommended or 0), max(len(df) - 1, 0)))
        self.selected_row = None

        info = ttk.Label(
            self,
            text=(
                "데이터의 어느 행이 컬럼 헤더인지 선택하세요.\n"
                "권장 행: #{rec} (더블클릭 또는 버튼으로 선택 가능)"
            ).format(rec=self.recommended + 1),
            justify="left",
            wraplength=760,
        )
        info.pack(padx=10, pady=(12, 6), anchor="w")

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        columns = ["Row #"] + [f"Col {i+1}" for i in range(df.shape[1])]
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        for i, col in enumerate(columns):
            heading = col if i == 0 else f"Column {i}"
            width = 70 if i == 0 else 140
            self.tree.heading(col, text=heading, anchor="w")
            self.tree.column(col, width=width, anchor="w", stretch=i != 0)

        display_rows = df.copy()
        if len(display_rows) > 200 and self.recommended < len(display_rows):
            hi = max(self.recommended + 30, 200)
            display_rows = display_rows.iloc[:hi]
        for idx, row in display_rows.iterrows():
            values = [idx + 1]
            for val in row:
                if pd.isna(val):
                    values.append("")
                else:
                    values.append(str(val))
            self.tree.insert("", "end", iid=str(idx), values=values)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(0, 12))

        self.use_recommended_btn = ttk.Button(
            btn_frame,
            text=f"권장 행 사용 (#{self.recommended + 1})",
            command=self._use_recommended,
        )
        self.use_recommended_btn.pack(side="left")

        self.use_selected_btn = ttk.Button(
            btn_frame,
            text="선택한 행 사용",
            state="disabled",
            command=self._use_selected,
        )
        self.use_selected_btn.pack(side="left", padx=(8, 0))

        ttk.Button(btn_frame, text="취소", command=self._cancel).pack(side="right")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-Button-1>", self._on_double_click)

        if str(self.recommended) in self.tree.get_children():
            self.tree.selection_set(str(self.recommended))
            self.tree.see(str(self.recommended))
            self.selected_row = self.recommended
            self.use_selected_btn.config(state="normal")

        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            self.selected_row = None
            self.use_selected_btn.config(state="disabled")
            return
        self.selected_row = int(sel[0])
        self.use_selected_btn.config(state="normal")

    def _on_double_click(self, event=None):
        self._use_selected()

    def _use_recommended(self):
        self.result = self.recommended
        self.destroy()

    def _use_selected(self):
        if self.selected_row is None:
            return
        self.result = self.selected_row
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

# ------------------------------- 
# Style helpers
# ------------------------------- 
class StyleConfig:
    def __init__(self, size='12', weight='normal', style='normal',
                 color='black', linestyle='solid', width='1', underline=False):
        self.size = tk.StringVar(value=size)
        self.weight = tk.StringVar(value=weight)
        self.style = tk.StringVar(value=style)
        self.color = tk.StringVar(value=color)
        self.linestyle = tk.StringVar(value=linestyle)
        self.width = tk.StringVar(value=width)
        self.underline = tk.BooleanVar(value=underline)

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

    def _parse_float(self, v, d):
        try: return float(v)
        except Exception: return d


class StyleableObject:
    def __init__(self, name, obj_type, style_config, text_var=None, x_var=None, y_var=None):
        self.name = name
        self.obj_type = obj_type
        self.style = style_config
        self.text = text_var
        self.x = x_var
        self.y = y_var


# ------------------------------- 
# Main App
# ------------------------------- 
class GraphMaker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Maker - Multi-Series")
        self.geometry("1400x860")

        # ---- 부팅/콜백 제어 ----
        self._booting = True
        self._suspend_callbacks = False

        # Matplotlib 기본 폰트
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['axes.unicode_minus'] = False

        # ---- 상태 변수 ----
        self.df = None
        self.figure = None
        self.canvas = None

        # 데이터/축 관련
        self.x_col = tk.StringVar()
        self.start_row, self.end_row = tk.StringVar(), tk.StringVar()
        self.x_min, self.x_max = tk.StringVar(), tk.StringVar()
        self.y_min, self.y_max = tk.StringVar(), tk.StringVar()
        self.x_interval, self.y_interval = tk.StringVar(), tk.StringVar()

        # 그리드/ 회귀/ 캔버스 크기
        self.show_x_grid = tk.BooleanVar(value=True)
        self.show_y_grid = tk.BooleanVar(value=True)
        self.fit_intercept = tk.BooleanVar(value=True)

        self.figure_width_cm = tk.StringVar(value="15")
        self.figure_height_cm = tk.StringVar(value="10")
        self.size_target = tk.StringVar(value="Entire Figure")   # "Inner Plot"도 지원
        self.preview_exact = tk.BooleanVar(value=True)
        self.screen_dpi = 100

        # DataSelector에서 넘어온 선택 컬럼
        self.selected_columns = []
        self._underline_lines = []

        # ---- 스타일 오브젝트(타이틀/라벨/틱 등) ----
        self.styleable_objects = {}
        self._init_style_objects()

        # ---- 레전드 스타일 ----
        self.show_legend = tk.BooleanVar(value=True)
        self.legend_loc = tk.StringVar(value="best")
        self.legend_fontsize = tk.StringVar(value="10")
        self.legend_frameon = tk.BooleanVar(value=True)
        self.legend_fancybox = tk.BooleanVar(value=True)
        self.legend_shadow = tk.BooleanVar(value=False)
        self.legend_facecolor = tk.StringVar(value='white')
        self.legend_edgecolor = tk.StringVar(value='black')
        self.legend_linewidth = tk.StringVar(value='0.8')

        # 레전드 라벨 템플릿(디폴트: "축이름 (식)")
        self.legend_label_template = tk.StringVar(value="{col} ({eq})")

        # y축 눈금 회전 옵션
        self.rotate_y_ticks_90 = tk.BooleanVar(value=False)

        # ---- 시리즈별 저장소 ----
        self.series_styles = {}               # {col: StyleableObject(line)}
        self.trendline_styles = {}            # {col: StyleableObject(line)}
        self.equation_styles = {}             # {col: StyleableObject(text)}  # on-plot 편집용
        self.series_show_trend = {}           # {col: BooleanVar}

        # ★ 회귀식 위치/포맷/표시 제어(신규)
        self.series_eq_place = {}             # {col: StringVar('legend'|'onplot'|'none')}
        self.series_custom_legend_text = {}   # {col: StringVar}
        self.series_use_custom_legend = {}    # {col: BooleanVar}
        self.series_last_eq = {}              # {col: StringVar}  # 계산된 최신 식(읽기)
        self.series_eq_auto_update = {}       # {col: BooleanVar} # 계산식→편집식 자동 동기화

        self.series_eq_include_r2 = {}        # {col: BooleanVar}
        self.series_eq_y_symbol = {}          # {col: StringVar}  # 기본 'y'
        self.series_eq_x_symbol = {}          # {col: StringVar}  # 기본 'x'
        self.series_eq_format_mode = {}       # {col: StringVar('sigfigs'|'fixed'|'sci')}
        self.series_eq_precision = {}         # {col: StringVar}  # 정밀도(문자열로 Entry 바인딩)
        self.series_eq_hide_c_eps = {}        # {col: StringVar}  # |c| < eps이면 +c 숨김

        # ---- 내부 플래그 ----
        self._is_plotting = False
        self.auto_x_label = True
        self.auto_y_label = True

        # ---- 드래그 관련 ----
        self.dragger = None
        self.artist_map = {}
        self.annotations_counter = 0

        # ---- UI 생성 ----
        self._create_widgets()

        # ---- 디바운스(스로틀링) 타이머 ----
        self._debounce_ms = 120
        self._pending_plot_job = None
        self._pending_series_job = None

        # ---- 트레이스/초기 렌더 ----
        self._booting = False
        self.x_col.trace_add("write", self.on_x_change)
        self._schedule_plot()

    def _add_underline_line(self, artist, ax, pad_px=2):
        """
        주어진 Text artist 바로 아래에 밑줄(Line2D)을 하나 그려준다.
        1) 텍스트와 동일한 transform 좌표계에 그리려고 시도
        2) 실패 시 figure/axes 좌표로 변환해 폴백
        """
        # 1) 렌더러 확보 (1차 draw 이후여야 bbox가 정확)
        renderer = None
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            try:
                renderer = self.figure.canvas.get_renderer()
            except Exception:
                renderer = None
        if renderer is None:
            return

        # 2) 텍스트 bbox(디스플레이 좌표, 픽셀 단위)
        try:
            bbox = artist.get_window_extent(renderer=renderer)
        except Exception:
            return
        if bbox.width <= 0:
            return

        x0_disp, x1_disp = bbox.x0, bbox.x1
        y_disp = bbox.y0 - float(pad_px)  # 텍스트 하단보다 pad_px만큼 아래

        # 3) 선 스타일(글자 색/크기 기반)
        try:
            fs = float(getattr(artist, "get_fontsize", lambda: 10.0)())
        except Exception:
            fs = 10.0
        try:
            color = getattr(artist, "get_color", lambda: "black")()
        except Exception:
            color = "black"
        z = (artist.get_zorder() or 2) + 0.3
        lw = max(0.8, fs / 12.0)  # 글자 크기에 비례

        # 4) 텍스트와 같은 transform 좌표에 그리기(가장 이상적)
        tr = artist.get_transform()
        try:
            inv = tr.inverted()
            (x0_u, y_u) = inv.transform((x0_disp, y_disp))
            (x1_u, _)   = inv.transform((x1_disp, y_disp))
            line = Line2D([x0_u, x1_u], [y_u, y_u],
                        transform=tr,
                        color=color,
                        linewidth=lw,
                        solid_capstyle="butt",
                        zorder=z)
            ax.add_line(line)
            self._underline_lines.append(line)
            return
        except Exception:
            pass

        # 5) 폴백: 디스플레이→figure→axes로 변환하여 ax.transAxes로 그림(비율 좌표)
        try:
            # 디스플레이 → figure
            to_fig = self.figure.transFigure.inverted()
            (x0_fig, y_fig) = to_fig.transform((x0_disp, y_disp))
            (x1_fig, _)     = to_fig.transform((x1_disp, y_disp))

            # figure → axes (비율 좌표)
            to_ax = ax.transAxes.inverted()
            (x0_ax, y_ax) = to_ax.transform(self.figure.transFigure.transform((x0_fig, y_fig)))
            (x1_ax, _)    = to_ax.transform(self.figure.transFigure.transform((x1_fig, y_fig)))

            line = Line2D([x0_ax, x1_ax], [y_ax, y_ax],
                        transform=ax.transAxes,
                        color=color,
                        linewidth=lw,
                        solid_capstyle="butt",
                        zorder=z)
            ax.add_line(line)
            self._underline_lines.append(line)
        except Exception:
            return



    def remove_custom_text(self, name: str):
        """Actions 탭의 Custom Text 편집 블록에서 '삭제' 눌렀을 때 호출"""
        if name in self.styleable_objects:
            del self.styleable_objects[name]
            # 에디터 영역 갱신 + 그래프 즉시 리렌더
            self._refresh_custom_text_editor()
            self.plot_graph()

    # ---------- utility ----------
    def _schedule_plot(self, delay=None):
        if delay is None:
            delay = self._debounce_ms
        if self._pending_plot_job is not None:
            try: self.after_cancel(self._pending_plot_job)
            except Exception: pass
        self._pending_plot_job = self.after(delay, self._do_plot)

    def _do_plot(self):
        self._pending_plot_job = None
        self.plot_graph()

    def _schedule_axis_and_plot(self):
        if self._pending_plot_job is not None:
            try: self.after_cancel(self._pending_plot_job)
            except Exception: pass
        self._pending_plot_job = self.after(self._debounce_ms, self._do_axis_and_plot)

    def _do_axis_and_plot(self):
        self._pending_plot_job = None
        self._update_axis_defaults()
        self.plot_graph()

    def _schedule_series_panel_build(self, col):
        if self._pending_series_job is not None:
            try: self.after_cancel(self._pending_series_job)
            except Exception: pass
        self._pending_series_job = self.after(self._debounce_ms, lambda: self._do_build_series_panel(col))

    def _do_build_series_panel(self, col):
        self._pending_series_job = None
        self._build_series_panel(col)

    def _apply_underline(self, artist, enabled: bool):
        """가능하면 Text.set_underline(True) 사용, 안 되면 False 반환하여 폴백(라인)을 쓰게 한다."""
        try:
            if hasattr(artist, "set_underline"):
                artist.set_underline(bool(enabled))
                return True
        except Exception:
            pass
        return False


    def _cycle_color(self, i):
        palette = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                   "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
        return palette[i % len(palette)]

    def _available_columns(self):
        if self.df is None: return []
        cols = list(self.df.columns)
        if self.selected_columns:
            return [c for c in self.selected_columns if c in cols]
        return cols

    def _get_selected_y_cols(self):
        if not hasattr(self, "y_listbox"):
            return []
        x = self.x_col.get()
        selected = [self.y_listbox.get(i) for i in self.y_listbox.curselection()]
        # X축 컬럼은 제외
        return [c for c in selected if c != x]


    def _ensure_series_state_attrs(self):
        """시리즈 관련 상태 딕셔너리가 항상 존재하도록 보장"""
        if not hasattr(self, "series_styles"):               self.series_styles = {}
        if not hasattr(self, "trendline_styles"):            self.trendline_styles = {}
        if not hasattr(self, "equation_styles"):             self.equation_styles = {}
        if not hasattr(self, "series_show_trend"):           self.series_show_trend = {}
        if not hasattr(self, "series_show_eq"):              self.series_show_eq = {}
        if not hasattr(self, "series_eq_in_legend"):         self.series_eq_in_legend = {}

        # 레전드/식 관련 부가 상태
        if not hasattr(self, "series_custom_legend_text"):   self.series_custom_legend_text = {}
        if not hasattr(self, "series_use_custom_legend"):    self.series_use_custom_legend = {}
        if not hasattr(self, "series_last_eq"):              self.series_last_eq = {}
        if not hasattr(self, "series_eq_auto_update"):       self.series_eq_auto_update = {}

    def _ensure_series_objects(self, y_cols):
        # 기본 딕셔너리 존재 보장
        if not hasattr(self, "series_styles"):               self.series_styles = {}
        if not hasattr(self, "trendline_styles"):            self.trendline_styles = {}
        if not hasattr(self, "equation_styles"):             self.equation_styles = {}
        if not hasattr(self, "series_show_trend"):           self.series_show_trend = {}

        # 예전 호환(있어도 무시 가능)
        if not hasattr(self, "series_show_eq"):              self.series_show_eq = {}
        if not hasattr(self, "series_eq_in_legend"):         self.series_eq_in_legend = {}

        # 새 옵션들(식 위치/서식 등)
        if not hasattr(self, "series_eq_place"):             self.series_eq_place = {}
        if not hasattr(self, "series_custom_legend_text"):   self.series_custom_legend_text = {}
        if not hasattr(self, "series_use_custom_legend"):    self.series_use_custom_legend = {}
        if not hasattr(self, "series_last_eq"):              self.series_last_eq = {}
        if not hasattr(self, "series_eq_auto_update"):       self.series_eq_auto_update = {}

        if not hasattr(self, "series_eq_include_r2"):        self.series_eq_include_r2 = {}
        if not hasattr(self, "series_eq_y_symbol"):          self.series_eq_y_symbol = {}
        if not hasattr(self, "series_eq_x_symbol"):          self.series_eq_x_symbol = {}
        if not hasattr(self, "series_eq_format_mode"):       self.series_eq_format_mode = {}
        if not hasattr(self, "series_eq_precision"):         self.series_eq_precision = {}
        if not hasattr(self, "series_eq_hide_c_eps"):        self.series_eq_hide_c_eps = {}

        for i, col in enumerate(y_cols):
            if col not in self.series_styles:
                self.series_styles[col] = StyleableObject(
                    f"Series [{col}]", 'line',
                    StyleConfig(color=self._cycle_color(i), width='1.8', linestyle='solid')
                )
            if col not in self.trendline_styles:
                self.trendline_styles[col] = StyleableObject(
                    f"Trendline [{col}]", 'line',
                    StyleConfig(color=self._cycle_color(i), width='1.2', linestyle='--')
                )
            if col not in self.equation_styles:
                self.equation_styles[col] = StyleableObject(
                    f"Equation [{col}]", 'text',
                    StyleConfig(color=self._cycle_color(i), size='10'),
                    text_var=tk.StringVar(value=""),
                    x_var=tk.DoubleVar(value=0), y_var=tk.DoubleVar(value=0)
                )
            if col not in self.series_show_trend:
                self.series_show_trend[col] = tk.BooleanVar(value=False)

            # 구형 호환
            if col not in self.series_show_eq:
                self.series_show_eq[col] = tk.BooleanVar(value=False)
            if col not in self.series_eq_in_legend:
                self.series_eq_in_legend[col] = tk.BooleanVar(value=False)

            # --- 신규 옵션들 ---
            if col not in self.series_eq_place:
                # 기본값: 범례에 "열이름 (식)" 형태로 들어가게
                self.series_eq_place[col] = tk.StringVar(value="legend")
            if col not in self.series_custom_legend_text:
                self.series_custom_legend_text[col] = tk.StringVar(value="")
            if col not in self.series_use_custom_legend:
                self.series_use_custom_legend[col] = tk.BooleanVar(value=False)
            if col not in self.series_last_eq:
                self.series_last_eq[col] = tk.StringVar(value="")
            if col not in self.series_eq_auto_update:
                self.series_eq_auto_update[col] = tk.BooleanVar(value=True)

            if col not in self.series_eq_include_r2:
                self.series_eq_include_r2[col] = tk.BooleanVar(value=False)
            if col not in self.series_eq_y_symbol:
                self.series_eq_y_symbol[col] = tk.StringVar(value="y")
            if col not in self.series_eq_x_symbol:
                self.series_eq_x_symbol[col] = tk.StringVar(value="x")
            if col not in self.series_eq_format_mode:
                self.series_eq_format_mode[col] = tk.StringVar(value="sigfigs")  # 'sigfigs'|'fixed'|'sci'
            if col not in self.series_eq_precision:
                self.series_eq_precision[col] = tk.StringVar(value="3")
            if col not in self.series_eq_hide_c_eps:
                self.series_eq_hide_c_eps[col] = tk.StringVar(value="0")




    def _safe_int(self, var_or_str, default):
        try:
            v = var_or_str.get() if hasattr(var_or_str, "get") else var_or_str
            return int(v)
        except Exception:
            return default

    def _safe_float(self, var_or_str, default):
        try:
            v = var_or_str.get() if hasattr(var_or_str, "get") else var_or_str
            return float(v)
        except Exception:
            return default

    def _format_equation(self, col, m, c, r2):
        mode = (self.series_eq_fmt_mode[col].get() or 'sigfigs').lower()
        prec = max(0, self._safe_int(self.series_eq_precision[col], 3))
        xs  = self.series_eq_xsym[col].get().strip() or 'x'
        ys  = self.series_eq_ysym[col].get().strip() or 'y'
        thr = abs(self._safe_float(self.series_eq_zero_thr[col], 0.0))

        fmt = f".{prec}g" if mode == 'sigfigs' else f".{prec}f"
        m_s  = format(m, fmt)
        c_s  = format(c, fmt)

        # intercept 표시 여부
        show_c = (abs(c) > thr)
        sign = " + " if c >= 0 else " - "
        eq = f"{ys} = {m_s}{xs}"
        if show_c:
            eq += f"{sign}{c_s.lstrip('-')}"

        if self.series_eq_show_r2[col].get() and (r2 is not None):
            r2_s = format(r2, fmt)
            eq += f"  (R²={r2_s})"
        return eq

    def _refresh_axis_selectors(self):
        cols = self._available_columns()

        # X 선택 콤보박스 갱신
        if hasattr(self, "x_col_combo"):
            self.x_col_combo["values"] = cols
            # 현재 값이 후보에 없으면 첫 컬럼으로
            if cols and self.x_col.get() not in cols:
                self.x_col.set(cols[0])
            # 콤보박스 표시값 을 변수와 동기화(필요 시)
            try:
                self.x_col_combo.set(self.x_col.get() if self.x_col.get() in cols else (cols[0] if cols else ""))
            except Exception:
                pass

        # Y 리스트 — 현재 X는 제외
        self.y_listbox.delete(0, "end")
        x_now = self.x_col.get()
        for c in cols:
            if c != x_now:
                self.y_listbox.insert("end", c)
        

    def _plot_graph_event(self, event=None):
        self._schedule_plot()

    # ---------- UI ----------
    def _mount_canvas(self):
        widget = self.canvas.get_tk_widget()
        widget.pack_forget()
        if self.preview_exact.get():
            w_in, h_in = self.figure.get_size_inches()
            dpi = self.figure.dpi
            widget.config(width=int(w_in * dpi), height=int(h_in * dpi))
            widget.pack(anchor="center", padx=10, pady=10)
        else:
            widget.pack(fill="both", expand=True)

    def _create_widgets(self):
        main = ttk.Frame(self); main.pack(fill="both", expand=True, padx=10, pady=10)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1, minsize=400)
        main.grid_columnconfigure(1, weight=0)

        self.plot_frame = ttk.Frame(main);
        self.plot_frame.grid(row=0, column=0, sticky="nsew")

        inspector_container = ttk.Frame(main, width=430)
        inspector_container.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        inspector_container.grid_propagate(False)

        self.notebook = ttk.Notebook(inspector_container)
        self.notebook.pack(fill="both", expand=True)

        self._populate_inspector()
        self.x_col.trace_add("write", self.on_x_change)

    def _populate_inspector(self):
        tab_data = ttk.Frame(self.notebook, padding=10)
        tab_global_style = ttk.Frame(self.notebook, padding=10)
        tab_axis_style = ttk.Frame(self.notebook, padding=10)
        tab_series_style = ttk.Frame(self.notebook, padding=10)
        tab_actions = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(tab_data, text="Data")
        self.notebook.add(tab_global_style, text="Graph Style")
        self.notebook.add(tab_axis_style, text="Axis Style")
        self.notebook.add(tab_series_style, text="Series Style")
        self.notebook.add(tab_actions, text="Actions")

        self._create_data_tab(tab_data)
        self._create_global_style_tab(tab_global_style)
        self._create_axis_style_tab(tab_axis_style)
        self._create_series_style_tab(tab_series_style)
        self._create_actions_tab(tab_actions)

    def _create_axis_style_tab(self, parent):   
        # ----- X-Axis -----
        f_x = ttk.LabelFrame(parent, text="X-Axis", padding=10)
        f_x.pack(fill="x", pady=6)

        # Label
        f_xlabel = ttk.LabelFrame(f_x, text="Label Style", padding=8)
        f_xlabel.pack(fill="x", pady=4)
        self._create_font_editor(f_xlabel, self.styleable_objects["X-Axis Label"], show_text=True)

        # Tick
        f_xticks = ttk.LabelFrame(f_x, text="Tick Label Style", padding=8)
        f_xticks.pack(fill="x", pady=4)
        self._create_font_editor(f_xticks, self.styleable_objects["X-Tick Labels"], show_text=False)

        # Range & Interval (날짜축일 땐 Interval은 무시)
        f_xrange = ttk.LabelFrame(f_x, text="Range & Interval", padding=8)
        f_xrange.pack(fill="x", pady=4)
        ttk.Label(f_xrange, text="Min:").grid(row=0, column=0, sticky="w")
        x_min_e = ttk.Entry(f_xrange, textvariable=self.x_min, width=12)
        x_min_e.grid(row=0, column=1, sticky="ew", padx=(4,6))
        ttk.Label(f_xrange, text="Max:").grid(row=0, column=2, sticky="w")
        x_max_e = ttk.Entry(f_xrange, textvariable=self.x_max, width=12)
        x_max_e.grid(row=0, column=3, sticky="ew", padx=(4,6))
        ttk.Label(f_xrange, text="Interval:").grid(row=0, column=4, sticky="w")
        x_int_e = ttk.Entry(f_xrange, textvariable=self.x_interval, width=8)
        x_int_e.grid(row=0, column=5, sticky="ew", padx=(4,0))
        for i in [1,3,5]:
            f_xrange.grid_columnconfigure(i, weight=1)

        # ----- Y-Axis -----
        f_y = ttk.LabelFrame(parent, text="Y-Axis", padding=10)
        f_y.pack(fill="x", pady=6)

        # Label
        f_ylabel = ttk.LabelFrame(f_y, text="Label Style", padding=8)
        f_ylabel.pack(fill="x", pady=4)
        self._create_font_editor(f_ylabel, self.styleable_objects["Y-Axis Label"], show_text=True)

        # Tick
        f_yticks = ttk.LabelFrame(f_y, text="Tick Label Style", padding=8)
        f_yticks.pack(fill="x", pady=4)
        self._create_font_editor(f_yticks, self.styleable_objects["Y-Tick Labels"], show_text=False)
        ttk.Checkbutton(
            f_yticks, text="Rotate Y ticks 90°", variable=self.rotate_y_ticks_90, command=self.plot_graph
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(6,0))

        # Range & Interval
        f_yrange = ttk.LabelFrame(f_y, text="Range & Interval", padding=8)
        f_yrange.pack(fill="x", pady=4)
        ttk.Label(f_yrange, text="Min:").grid(row=0, column=0, sticky="w")
        y_min_e = ttk.Entry(f_yrange, textvariable=self.y_min, width=12)
        y_min_e.grid(row=0, column=1, sticky="ew", padx=(4,6))
        ttk.Label(f_yrange, text="Max:").grid(row=0, column=2, sticky="w")
        y_max_e = ttk.Entry(f_yrange, textvariable=self.y_max, width=12)
        y_max_e.grid(row=0, column=3, sticky="ew", padx=(4,6))
        ttk.Label(f_yrange, text="Interval:").grid(row=0, column=4, sticky="w")
        y_int_e = ttk.Entry(f_yrange, textvariable=self.y_interval, width=8)
        y_int_e.grid(row=0, column=5, sticky="ew", padx=(4,0))
        for i in [1,3,5]:
            f_yrange.grid_columnconfigure(i, weight=1)

        # ★ 엔터/포커스아웃 시 즉시 반영: 축 범위/간격 변경 → 바로 재그리기
        def _axis_changed(_evt=None):
            self._schedule_plot()  # 여기서는 기본값 재계산 없이, 입력값을 그대로 적용하여 그리기
        for e in (x_min_e, x_max_e, x_int_e, y_min_e, y_max_e, y_int_e):
            e.bind("<Return>", _axis_changed)
            e.bind("<FocusOut>", _axis_changed)

    def _create_data_tab(self, parent):
        f1 = ttk.LabelFrame(parent, text="Data Source", padding=10)
        f1.pack(fill="x", pady=6)
        ttk.Button(f1, text="Load Data File", command=self.load_data_file).pack(fill="x")
        self.file_label = ttk.Label(f1, text="No file selected.", wraplength=360)
        self.file_label.pack(pady=5, fill="x")

        f2 = ttk.LabelFrame(parent, text="Data Selection", padding=10)
        f2.pack(fill="x", pady=6)
        self.select_btn = ttk.Button(f2, text="Select Rows & Columns...", command=self.open_data_selector, state="disabled")
        self.select_btn.pack(fill="x", pady=(2,8))

        # ▶ X-Axis Column (統一된 디자인: Combobox, readonly)
        ttk.Label(f2, text="X-Axis Column:").pack(anchor="w")
        self.x_col_combo = ttk.Combobox(f2, textvariable=self.x_col, state="readonly")
        self.x_col_combo.pack(fill="x")
        # 선택 변경 시 즉시 반영
        self.x_col_combo.bind("<<ComboboxSelected>>", self.on_x_change)

        ttk.Label(f2, text="Y-Axis Columns (multi-select):").pack(anchor="w", pady=(6,0))
        self.y_listbox = tk.Listbox(f2, selectmode="extended", height=6, exportselection=False)
        self.y_listbox.pack(fill="x", expand=True)
        self.y_listbox.bind("<<ListboxSelect>>", self._on_y_change)

        row_frame = ttk.Frame(f2)
        row_frame.pack(fill="x", pady=(6,0))
        ttk.Label(row_frame, text="Start Row:").grid(row=0, column=0, sticky="w")
        e_start = ttk.Entry(row_frame, textvariable=self.start_row)
        e_start.grid(row=0, column=1, sticky="ew", padx=(4,10))
        ttk.Label(row_frame, text="End Row:").grid(row=0, column=2, sticky="w")
        e_end = ttk.Entry(row_frame, textvariable=self.end_row)
        e_end.grid(row=0, column=3, sticky="ew", padx=(4,0))
        row_frame.grid_columnconfigure(1, weight=1)
        row_frame.grid_columnconfigure(3, weight=1)

        # 엔터/포커스아웃 시 즉시 반영: 행 범위 변경 → 축 기본값 재계산 + 재그리기
        def _rows_changed(_evt=None):
            self._schedule_axis_and_plot()
        e_start.bind("<Return>", _rows_changed)
        e_end.bind("<Return>", _rows_changed)
        e_start.bind("<FocusOut>", _rows_changed)
        e_end.bind("<FocusOut>", _rows_changed)


    def _create_global_style_tab(self, parent):
        f_title = ttk.LabelFrame(parent, text="Title", padding=10)
        f_title.pack(fill="x", pady=6)
        self._create_font_editor(f_title, self.styleable_objects["Title"], show_text=True)

        f_legend = ttk.LabelFrame(parent, text="Legend", padding=10)
        f_legend.pack(fill="x", pady=6, expand=True)
        f_legend.grid_columnconfigure(1, weight=1)
        f_legend.grid_columnconfigure(3, weight=1)

        ttk.Checkbutton(f_legend, text="Show legend", variable=self.show_legend, command=self.plot_graph)\
            .grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(f_legend, text="Draw Frame", variable=self.legend_frameon, command=self.plot_graph)\
            .grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(f_legend, text="Fancy Box (rounded)", variable=self.legend_fancybox, command=self.plot_graph)\
            .grid(row=1, column=1, columnspan=3, sticky="w", padx=10)
        ttk.Checkbutton(f_legend, text="Shadow", variable=self.legend_shadow, command=self.plot_graph)\
            .grid(row=2, column=0, sticky="w")

        ttk.Label(f_legend, text="Location").grid(row=3, column=0, sticky="w", pady=(8,0))
        loc_cb = ttk.Combobox(f_legend, textvariable=self.legend_loc, values=[
            "best","upper right","upper left","lower left","lower right",
            "right","center left","center right","lower center","upper center","center"
        ], state="readonly")
        loc_cb.grid(row=3, column=1, columnspan=3, sticky="ew", padx=(6,0), pady=(8,0))
        loc_cb.bind("<<ComboboxSelected>>", lambda _e: self.plot_graph())

        ttk.Label(f_legend, text="Fontsize").grid(row=4, column=0, sticky="w", pady=(6,0))
        fs_e = ttk.Entry(f_legend, textvariable=self.legend_fontsize, width=6)
        fs_e.grid(row=4, column=1, sticky="ew", padx=(6,0), pady=(6,0))
        ttk.Label(f_legend, text="Border").grid(row=4, column=2, sticky="w", padx=(10,0))
        lw_e = ttk.Entry(f_legend, textvariable=self.legend_linewidth, width=6)
        lw_e.grid(row=4, column=3, sticky="ew", padx=(6,0), pady=(6,0))

        ttk.Label(f_legend, text="Face Color").grid(row=5, column=0, sticky="w", pady=(6,0))
        fc_e = ttk.Entry(f_legend, textvariable=self.legend_facecolor)
        fc_e.grid(row=5, column=1, sticky="ew", padx=(6,0))
        ttk.Button(f_legend, text="...", width=2,
                command=lambda: (self._choose_color(self.legend_facecolor), self.plot_graph()))\
            .grid(row=5, column=2, sticky="w", padx=(4,0))

        ttk.Label(f_legend, text="Edge Color").grid(row=6, column=0, sticky="w", pady=(6,0))
        ec_e = ttk.Entry(f_legend, textvariable=self.legend_edgecolor)
        ec_e.grid(row=6, column=1, sticky="ew", padx=(6,0))
        ttk.Button(f_legend, text="...", width=2,
                command=lambda: (self._choose_color(self.legend_edgecolor), self.plot_graph()))\
            .grid(row=6, column=2, sticky="w", padx=(4,0))

        # ★ 엔터/포커스아웃 시 즉시 반영
        def _legend_changed(_evt=None):
            self._schedule_plot()
        for e in (fs_e, lw_e, fc_e, ec_e):
            e.bind("<Return>", _legend_changed)
            e.bind("<FocusOut>", _legend_changed)

        f_grid = ttk.LabelFrame(parent, text="Grid & Size", padding=10)
        f_grid.pack(fill="x", pady=6)
        ttk.Checkbutton(f_grid, text="Show X-Axis Grid", variable=self.show_x_grid, command=self.plot_graph).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(f_grid, text="Show Y-Axis Grid", variable=self.show_y_grid, command=self.plot_graph).grid(row=0, column=1, sticky="w", padx=(10,0))

        ttk.Label(f_grid, text="Width (cm):").grid(row=1, column=0, sticky="w", pady=(8,0))
        we = ttk.Entry(f_grid, textvariable=self.figure_width_cm, width=8); we.grid(row=2, column=0, sticky="ew")
        ttk.Label(f_grid, text="Height (cm):").grid(row=1, column=1, sticky="w", padx=(10,0), pady=(8,0))
        he = ttk.Entry(f_grid, textvariable=self.figure_height_cm, width=8); he.grid(row=2, column=1, sticky="ew", padx=(10,0))
        we.bind("<Return>", self._plot_graph_event); we.bind("<FocusOut>", self._plot_graph_event)
        he.bind("<Return>", self._plot_graph_event); he.bind("<FocusOut>", self._plot_graph_event)

        ttk.Label(f_grid, text="Size Target:").grid(row=3, column=0, sticky="w", pady=(8,0))
        size_combo = ttk.Combobox(f_grid, textvariable=self.size_target, values=["Entire Figure","Inner Plot"], state="readonly")
        size_combo.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.size_target.trace_add("write", lambda *a: self.plot_graph())
        ttk.Checkbutton(f_grid, text="Exact size preview (no stretch)", variable=self.preview_exact, command=self.plot_graph)\
            .grid(row=5, column=0, columnspan=2, sticky="w", pady=(8,0))
        f_grid.grid_columnconfigure(0, weight=1)
        f_grid.grid_columnconfigure(1, weight=1)

    def _calculate_axis_defaults(self, data_min, data_max):
        """숫자형 데이터용 '예쁜' 기본 범위/간격."""
        if data_min is None or data_max is None or pd.isna(data_min) or pd.isna(data_max):
            return "0", "10", "2"
        nice_min = 0
        rng = data_max - nice_min
        if rng <= 0:
            rng = data_max if data_max > 0 else 1.0
        power = 10 ** math.floor(math.log10(rng))
        nice_max = math.ceil(data_max / power) * power
        if nice_max <= nice_min:
            nice_max = data_max + power
        interval = (nice_max - nice_min) / 10.0
        if interval <= 0:
            interval = 1.0
        return str(nice_min), str(round(nice_max, 6)), str(round(interval, 6))


    def _update_axis_defaults(self):
        """데이터를 보고 X/Y 기본 범위를 채움. X가 날짜면 ISO 문자열로 채움."""
        if getattr(self, "_booting", True) or getattr(self, "_suspend_callbacks", False):
            return
        if self.df is None or not self.x_col.get():
            return

        pdf = self._get_plotting_df()
        if pdf.empty:
            return

        xser_raw = pdf[self.x_col.get()]
        is_datetime = pd.api.types.is_datetime64_any_dtype(xser_raw) or (
            xser_raw.dtype == object and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(xser_raw, errors='coerce'))
        )

        # X min/max
        if is_datetime:
            xdt = pd.to_datetime(xser_raw, errors="coerce").dropna()
            if not xdt.empty:
                self.x_min.set(xdt.min().strftime("%Y-%m-%d"))
                self.x_max.set(xdt.max().strftime("%Y-%m-%d"))
                self.x_interval.set("")  # 날짜축은 Interval 입력 미사용
        else:
            x = pd.to_numeric(xser_raw, errors="coerce").dropna()
            if not x.empty:
                xmin, xmax, xint = self._calculate_axis_defaults(x.min(), x.max())
                self.x_min.set(xmin); self.x_max.set(xmax); self.x_interval.set(xint)

        # Y min/max
        y_cols = self._get_selected_y_cols() or []
        if not y_cols:
            y_cols = [c for c in self._available_columns() if c != self.x_col.get()][:1]
            self._ensure_series_objects(y_cols)

        ys = []
        for c in y_cols:
            ser = pd.to_numeric(pdf[c], errors="coerce").dropna()
            if not ser.empty:
                ys.append(ser)
        if not ys:
            return

        y_all_min = min(s.min() for s in ys)
        y_all_max = max(s.max() for s in ys)
        y_min, y_max, y_int = self._calculate_axis_defaults(y_all_min, y_all_max)
        self.y_min.set(y_min); self.y_max.set(y_max); self.y_interval.set(y_int)


    def _create_series_style_tab(self, parent):
        f_selector = ttk.LabelFrame(parent, text="Series Selection", padding=10)
        f_selector.pack(fill="x", pady=6)
        ttk.Label(f_selector, text="Series to edit:").grid(row=0, column=0, sticky="w")
        self.series_combo = ttk.Combobox(f_selector, values=[], state="readonly")
        self.series_combo.grid(row=0, column=1, sticky="ew", padx=(6,0))
        self.series_combo.bind("<<ComboboxSelected>>", self._on_series_select)
        f_selector.grid_columnconfigure(1, weight=1)

        self.series_dyn = ttk.Frame(parent)
        self.series_dyn.pack(fill="x", expand=True)

    def _create_actions_tab(self, parent):
        f_actions = ttk.LabelFrame(parent, text="Actions", padding=10)
        f_actions.pack(fill="x", pady=6)
        ttk.Button(f_actions, text="Add Text to Graph", command=self.add_annotation).pack(fill="x", pady=4)
        ttk.Button(f_actions, text="Save Graph As Image...", command=self.save_graph).pack(fill="x", pady=4)

        self.custom_text_frame = ttk.LabelFrame(parent, text="Custom Text Objects", padding=10)
        self.custom_text_frame.pack(fill="x", pady=6)
        self.custom_text_dyn_frame = ttk.Frame(self.custom_text_frame)
        self.custom_text_dyn_frame.pack(fill="x")
        self._refresh_custom_text_editor()

    def _refresh_custom_text_editor(self):
        # 기존 위젯 정리
        for w in self.custom_text_dyn_frame.winfo_children():
            w.destroy()

        has_custom = False
        for name, obj in sorted(self.styleable_objects.items()):
            if obj.obj_type == 'text' and name.startswith("Custom Text"):
                has_custom = True

                # 한 개 블록 컨테이너
                outer = ttk.Frame(self.custom_text_dyn_frame)
                outer.pack(fill="x", pady=4)

                # 라벨 프레임 (여기에는 pack만 사용)
                editor_frame = ttk.LabelFrame(outer, text=name, padding=8)
                editor_frame.pack(fill="x")

                # ---- Toolbar(삭제 버튼) : pack 사용 ----
                toolbar = ttk.Frame(editor_frame)
                toolbar.pack(fill="x", pady=(0, 6))
                ttk.Button(
                    toolbar, text="삭제", width=6,
                    command=lambda n=name: self.remove_custom_text(n)
                ).pack(side="right")

                # ---- 콘텐츠 프레임: 여기 안에서만 grid 사용 ----
                content = ttk.Frame(editor_frame)
                content.pack(fill="x")

                # _create_font_editor는 grid를 쓰므로 editor_frame 말고 content를 부모로 전달
                self._create_font_editor(content, obj, show_text=True, allow_pos=True)

        if not has_custom:
            ttk.Label(self.custom_text_dyn_frame, text="No custom text objects added.", foreground="gray").pack()

    def _create_font_editor(self, parent, obj: StyleableObject, show_text=True, allow_pos=False):
        # 1) 본문 텍스트: 엔터로만 갱신
        if show_text and obj.text is not None:
            row = 0
            ttk.Label(parent, text="Text").grid(row=row, column=0, sticky="w")
            e = ttk.Entry(parent, textvariable=obj.text)
            e.grid(row=row, column=1, columnspan=3, sticky="ew", padx=(6,4))
            ttk.Button(parent, text="⏎", width=3, command=self.plot_graph).grid(row=row, column=4, sticky="e")
            e.bind("<Return>", lambda _e: self.plot_graph())

            def disable_auto(_evt=None):
                if obj is self.styleable_objects.get("X-Axis Label"): self.auto_x_label = False
                if obj is self.styleable_objects.get("Y-Axis Label"): self.auto_y_label = False
            e.bind("<Key>", disable_auto)
        else:
            row = -1

        # 2) 글꼴 크기/색상: 엔터/포커스아웃에만 반영
        row += 1
        ttk.Label(parent, text="Size").grid(row=row, column=0, sticky="w", pady=(6,0))
        se = ttk.Entry(parent, textvariable=obj.style.size, width=6); se.grid(row=row, column=1, sticky="w", padx=(6,0), pady=(6,0))
        ttk.Label(parent, text="Color").grid(row=row, column=2, sticky="w", padx=(6,0))
        ce = ttk.Entry(parent, textvariable=obj.style.color, width=10); ce.grid(row=row, column=3, sticky="w")
        ttk.Button(parent, text="...", width=2, command=lambda: (self._choose_color(obj.style.color), self.plot_graph())).grid(row=row, column=4, sticky="w", padx=(4,0))
        for entry in (se, ce):
            entry.bind("<Return>", lambda _e: self.plot_graph())
            entry.bind("<FocusOut>", lambda _e: self.plot_graph())

        # 3) 굵기/스타일: 선택 완료 시 반영
        row += 1
        ttk.Label(parent, text="Weight").grid(row=row, column=0, sticky="w", pady=(6,0))
        weight_cb = ttk.Combobox(parent, textvariable=obj.style.weight, values=['normal','bold'], width=8, state="readonly")
        weight_cb.grid(row=row, column=1, sticky="w", pady=(6,0))
        ttk.Label(parent, text="Style").grid(row=row, column=2, sticky="w", padx=(6,0))
        style_cb = ttk.Combobox(parent, textvariable=obj.style.style, values=['normal','italic'], width=8, state="readonly")
        style_cb.grid(row=row, column=3, sticky="w")
        weight_cb.bind("<<ComboboxSelected>>", lambda _e: self.plot_graph())
        style_cb.bind("<<ComboboxSelected>>",  lambda _e: self.plot_graph())

        # 4) 밑줄: 체크 즉시 반영 (토글류는 즉시 적용 UX)
        row += 1
        ttk.Checkbutton(parent, text="Underline", variable=obj.style.underline, command=self.plot_graph)\
            .grid(row=row, column=0, columnspan=2, sticky="w", pady=(6,0))

        # 5) 위치(x, y): 엔터/포커스아웃 시 반영
        if allow_pos and obj.x is not None and obj.y is not None:
            row += 1
            ttk.Label(parent, text="x").grid(row=row, column=0, sticky="w", pady=(8,0))
            xe = ttk.Entry(parent, textvariable=obj.x, width=8); xe.grid(row=row, column=1, sticky="w", padx=(6,0), pady=(8,0))
            ttk.Label(parent, text="y").grid(row=row, column=2, sticky="w", padx=(6,0))
            ye = ttk.Entry(parent, textvariable=obj.y, width=8); ye.grid(row=row, column=3, sticky="w")
            xe.bind("<Return>", self._plot_graph_event); xe.bind("<FocusOut>", self._plot_graph_event)
            ye.bind("<Return>", self._plot_graph_event); ye.bind("<FocusOut>", self._plot_graph_event)

        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(3, weight=1)


    def _create_line_editor(self, parent, style: StyleConfig, title):
        box = ttk.LabelFrame(parent, text=title, padding=8); box.pack(fill="x", pady=(6,0))

        ttk.Label(box, text="Width").grid(row=0, column=0, sticky="w")
        we = ttk.Entry(box, textvariable=style.width, width=6); we.grid(row=0, column=1, sticky="w", padx=(6,0))

        ttk.Label(box, text="Color").grid(row=0, column=2, sticky="w", padx=(10,0))
        ce = ttk.Entry(box, textvariable=style.color, width=10); ce.grid(row=0, column=3, sticky="w")
        ttk.Button(box, text="...", width=2, command=lambda: (self._choose_color(style.color), self.plot_graph()))\
            .grid(row=0, column=4, sticky="w", padx=(4,0))

        ttk.Label(box, text="Linestyle").grid(row=1, column=0, sticky="w", pady=(6,0))
        ls_cb = ttk.Combobox(box, textvariable=style.linestyle, values=['solid','dashed','dotted','dashdot'], state="readonly", width=10)
        ls_cb.grid(row=1, column=1, columnspan=2, sticky="w", pady=(6,0))

        # ★ 엔터/포커스아웃/선택 완료 시에만 반영
        for entry in (we, ce):
            entry.bind("<Return>", lambda _e: self.plot_graph())
            entry.bind("<FocusOut>", lambda _e: self.plot_graph())
        ls_cb.bind("<<ComboboxSelected>>", lambda _e: self.plot_graph())

        box.grid_columnconfigure(1, weight=1)
        box.grid_columnconfigure(3, weight=1)


    def _build_series_panel(self, col):
        # 패널 초기화
        for w in self.series_dyn.winfo_children():
            w.destroy()
        self._ensure_series_objects([col])

        # ── Regression & Equation ──
        f_trend = ttk.LabelFrame(self.series_dyn, text="Regression & Equation", padding=10)
        f_trend.pack(fill="x", pady=6)
        f_trend.grid_columnconfigure(1, weight=1)

        ttk.Checkbutton(f_trend, text="Show trendline",
            variable=self.series_show_trend[col], command=self._schedule_plot
        ).grid(row=0, column=0, sticky="w")

        ttk.Checkbutton(f_trend, text="Fit Intercept (y = m x + c)",
            variable=self.fit_intercept, command=self._schedule_plot
        ).grid(row=0, column=1, sticky="w", padx=(10,0))

        ttk.Checkbutton(f_trend, text="Include R²",
            variable=self.series_eq_include_r2[col], command=self._schedule_plot
        ).grid(row=0, column=2, sticky="w", padx=(10,0))

        ttk.Label(f_trend, text="y symbol").grid(row=1, column=0, sticky="w", pady=(6,0))
        ysym = ttk.Entry(f_trend, textvariable=self.series_eq_y_symbol[col], width=6)
        ysym.grid(row=1, column=1, sticky="w", padx=(6,0), pady=(6,0))
        ysym.bind("<Return>", self._plot_graph_event); ysym.bind("<FocusOut>", self._plot_graph_event)

        ttk.Label(f_trend, text="x symbol").grid(row=1, column=2, sticky="w")
        xsym = ttk.Entry(f_trend, textvariable=self.series_eq_x_symbol[col], width=6)
        xsym.grid(row=1, column=3, sticky="w", padx=(6,0))
        xsym.bind("<Return>", self._plot_graph_event); xsym.bind("<FocusOut>", self._plot_graph_event)

        ttk.Label(f_trend, text="Number format").grid(row=2, column=0, sticky="w", pady=(6,0))
        ttk.Combobox(f_trend, state="readonly",
                    values=["sigfigs","fixed","sci"],
                    textvariable=self.series_eq_format_mode[col], width=10
        ).grid(row=2, column=1, sticky="w", padx=(6,0), pady=(6,0))

        ttk.Label(f_trend, text="Precision").grid(row=2, column=2, sticky="w")
        prec = ttk.Entry(f_trend, textvariable=self.series_eq_precision[col], width=6)
        prec.grid(row=2, column=3, sticky="w", padx=(6,0))
        prec.bind("<Return>", self._plot_graph_event); prec.bind("<FocusOut>", self._plot_graph_event)

        ttk.Label(f_trend, text="Hide +c if |c| ≤").grid(row=3, column=0, sticky="w", pady=(6,0))
        eps = ttk.Entry(f_trend, textvariable=self.series_eq_hide_c_eps[col], width=6)
        eps.grid(row=3, column=1, sticky="w", padx=(6,0))
        eps.bind("<Return>", self._plot_graph_event); eps.bind("<FocusOut>", self._plot_graph_event)

        # ── Equation Placement ──
        f_place = ttk.LabelFrame(self.series_dyn, text="Equation Placement", padding=10)
        f_place.pack(fill="x", pady=6)

        def _on_place_change():
            self._schedule_plot()
            self._build_series_panel(col)  # 활성/비활성 재구성

        for i, (lbl, val) in enumerate([("None","none"),("In Legend","legend"),("On Plot (draggable)","onplot")]):
            ttk.Radiobutton(f_place, text=lbl, value=val,
                variable=self.series_eq_place[col], command=_on_place_change
            ).grid(row=0, column=i, padx=(0,10), sticky="w")

        place = self.series_eq_place[col].get()

        # ── In Legend 선택 시: 템플릿/커스텀 편집 ──
        if place == "legend":
            f_leg = ttk.LabelFrame(self.series_dyn, text=f"Legend Label [{col}]", padding=10)
            f_leg.pack(fill="x", pady=6)
            f_leg.grid_columnconfigure(1, weight=1)

            ttk.Label(f_leg, text="Legend label template (use {col}, {eq})").grid(row=0, column=0, columnspan=2, sticky="w")
            e_tpl = ttk.Entry(f_leg, textvariable=self.legend_label_template)
            e_tpl.grid(row=1, column=0, sticky="ew", padx=(0,6))
            e_tpl.bind("<Return>", self._plot_graph_event); e_tpl.bind("<FocusOut>", self._plot_graph_event)
            ttk.Button(f_leg, text="Reset",
                command=lambda:(self.legend_label_template.set("{col} ({eq})"), self._schedule_plot())
            ).grid(row=1, column=1, sticky="w")

            ttk.Checkbutton(f_leg, text="Use custom legend text (override template)",
                variable=self.series_use_custom_legend[col], command=self._schedule_plot
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8,0))

            ttk.Label(f_leg, text="Custom legend:").grid(row=3, column=0, sticky="w", pady=(6,0))
            e_leg = ttk.Entry(f_leg, textvariable=self.series_custom_legend_text[col])
            e_leg.grid(row=3, column=1, sticky="ew", padx=(6,0), pady=(6,0))
            e_leg.bind("<Return>", self._plot_graph_event); e_leg.bind("<FocusOut>", self._plot_graph_event)
            ttk.Button(f_leg, text="↵", width=3, command=self._schedule_plot).grid(row=3, column=2, sticky="w", padx=(6,0))

            def _fill_default_custom():
                colname = col
                eqtxt = self.series_last_eq[col].get().strip() or self.equation_styles[col].text.get().strip() or "(no eq)"
                self.series_custom_legend_text[col].set(f"{colname} ({eqtxt})")
                self.series_use_custom_legend[col].set(True)
                self._schedule_plot()
            ttk.Button(f_leg, text="Fill with current '{col} (eq)'", command=_fill_default_custom
            ).grid(row=4, column=2, sticky="e")

        # ── On Plot 선택 시: 편집 텍스트/스타일/좌표 ──
        if place == "onplot":
            f_eq = ttk.LabelFrame(self.series_dyn, text=f"On-plot Equation [{col}]", padding=10)
            f_eq.pack(fill="x", pady=6)
            f_eq.grid_columnconfigure(1, weight=1)

            ttk.Label(f_eq, text="Computed eq:").grid(row=0, column=0, sticky="w")
            ttk.Entry(f_eq, textvariable=self.series_last_eq[col], state="readonly"
            ).grid(row=0, column=1, sticky="ew", padx=(6,0))

            ttk.Label(f_eq, text="Editable eq (on-plot):").grid(row=1, column=0, sticky="w", pady=(6,0))
            e_user = ttk.Entry(f_eq, textvariable=self.equation_styles[col].text)
            e_user.grid(row=1, column=1, sticky="ew", padx=(6,0), pady=(6,0))
            e_user.bind("<Return>", self._plot_graph_event); e_user.bind("<FocusOut>", self._plot_graph_event)
            ttk.Button(f_eq, text="↵", width=3, command=self._schedule_plot).grid(row=1, column=2, sticky="w")

            def _toggle_auto():
                if self.series_eq_auto_update[col].get():
                    self.equation_styles[col].text.set(self.series_last_eq[col].get())
                self._schedule_plot()
            ttk.Checkbutton(f_eq, text="Keep editable eq synced with computed",
                variable=self.series_eq_auto_update[col], command=_toggle_auto
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,0))
            ttk.Button(f_eq, text="Copy computed → editable",
                command=lambda:(self.equation_styles[col].text.set(self.series_last_eq[col].get()), self._schedule_plot())
            ).grid(row=2, column=2, sticky="e")

            f_eqstyle = ttk.LabelFrame(self.series_dyn, text="On-plot Style / Position", padding=10)
            f_eqstyle.pack(fill="x", pady=6)
            self._create_font_editor(f_eqstyle, self.equation_styles[col], show_text=False, allow_pos=True)

        # ── 항상: 선 스타일 섹션 ──
        f_styles = ttk.LabelFrame(self.series_dyn, text="Styles", padding=10)
        f_styles.pack(fill="x", pady=6)
        self._create_line_editor(f_styles, self.series_styles[col].style, f"Line Style [{col}]")
        self._create_line_editor(f_styles, self.trendline_styles[col].style, f"Trendline Style [{col}]")

        self._schedule_plot()



    # ---------- style objects ----------
    def _init_style_objects(self):
        self.styleable_objects["Title"] = StyleableObject(
            "Title", 'title', StyleConfig(size='16', weight='bold'), text_var=tk.StringVar(value="Generated Graph")
        )
        self.styleable_objects["X-Axis Label"] = StyleableObject(
            "X-Axis Label", 'ax_label', StyleConfig(), text_var=tk.StringVar(value="X")
        )
        self.styleable_objects["Y-Axis Label"] = StyleableObject(
            "Y-Axis Label", 'ax_label', StyleConfig(), text_var=tk.StringVar(value="Y")
        )
        self.styleable_objects["X-Tick Labels"] = StyleableObject("X-Tick Labels", 'tick', StyleConfig())
        self.styleable_objects["Y-Tick Labels"] = StyleableObject("Y-Tick Labels", 'tick', StyleConfig())

    def open_data_selector(self):
        if self.df is None:
            return
        sel = DataSelector(self, self.df)
        self.wait_window(sel)

        # 선택된 컬럼 기준으로 X/Y 동기화
        self._map_columns_from_selection()
        self._auto_pick_xy_from_selection()
        self._schedule_axis_and_plot()

    def _map_columns_from_selection(self):
        """self.selected_columns: X=첫 번째, Y=그 외로 UI 반영"""
        if self.df is None:
            return
        cols = [c for c in (self.selected_columns or []) if c in self.df.columns]
        if not cols:
            return
        self.x_col.set(cols[0])
        self._refresh_axis_selectors()
        if hasattr(self, "y_listbox") and self.y_listbox.size() > 0:
            self.y_listbox.selection_clear(0, "end")
            items = [self.y_listbox.get(i) for i in range(self.y_listbox.size())]
            for c in cols[1:]:
                if c in items:
                    self.y_listbox.selection_set(items.index(c))


    def _auto_pick_xy_from_selection(self):
        """
        self.selected_columns와 현재 선택된 행 범위를 기준으로
        - EIS 규칙 우선(X=Re(Z)/Ohm, Y=-Im(Z)/Ohm)
        - 아니면 숫자열 2개 선택(경로/문자열 컬럼 자동 제외)
        를 적용해 X/Y를 설정하고, Y 리스트박스도 적절히 프리셀렉트한다.
        """
        if self.df is None:
            return

        cols = self._available_columns()  # selected_columns가 있으면 그것만 반환
        if not cols:
            return

        # 현재 plotting 구간만 사용(선택된 행 반영)
        pdf = self._get_plotting_df()
        if pdf.empty:
            pdf = self.df

        # 1) EIS 규칙 최우선
        eis_x = "Re(Z)/Ohm" if "Re(Z)/Ohm" in cols else None
        eis_y = "-Im(Z)/Ohm" if "-Im(Z)/Ohm" in cols else None
        if eis_x and eis_y:
            self.x_col.set(eis_x)
            # Y 리스트 갱신 전에 프리셋 보관
            preselect_y = eis_y
        else:
            # 2) 숫자열 2개를 자동 선정 (문자열/경로/메타 열 제외)
            numeric_cols = []
            for c in cols:
                s = pd.to_numeric(pdf[c], errors="coerce")
                # 숫자 비율이 일정 이상이면 숫자열로 간주
                if s.notna().sum() >= max(2, int(len(s) * 0.6)):
                    numeric_cols.append(c)

            # 후보가 부족하면 전체 중에서라도 베스트를 고름
            if len(numeric_cols) >= 2:
                x_cand, y_cand = numeric_cols[0], numeric_cols[1]
            elif len(numeric_cols) == 1:
                x_cand, y_cand = numeric_cols[0], next((c for c in cols if c != numeric_cols[0]), cols[0])
            else:
                # 전부 문자열이면 일단 첫 컬럼을 X, 두 번째를 Y로 (최소 동작 보장)
                x_cand, y_cand = cols[0], (cols[1] if len(cols) > 1 else cols[0])

            self.x_col.set(x_cand)
            preselect_y = y_cand

        # 3) X/Y UI 동기화: 리스트 갱신 후 Y를 프리셀렉트
        self._refresh_axis_selectors()  # X 콤보 & Y 리스트 재구성
        if hasattr(self, "y_listbox") and self.y_listbox.size() > 0:
            self.y_listbox.selection_clear(0, "end")
            items = [self.y_listbox.get(i) for i in range(self.y_listbox.size())]
            if preselect_y in items:
                self.y_listbox.selection_set(items.index(preselect_y))
            else:
                # EIS가 아니거나 후보가 없을 때 안전 기본값
                self.y_listbox.selection_set(0)



    def _is_numeric_series(self, s) -> bool:
        try:
            x = pd.to_numeric(s, errors="coerce")
            return x.notna().sum() >= max(2, int(len(x) * 0.6))
        except Exception:
            return False

    def _find_header_row(self, df_raw: pd.DataFrame) -> int:
        s = df_raw.astype(str).applymap(lambda v: v.strip())
        nrows = min(len(s), 100)

        re_keys = ("re(z", "rez", "zreal", "real", "re(")
        im_keys = ("-im(z", "im(z", "zimg", "zimag", "imag", "img", "-im", "im(")

        def has_keywords(vals):
            low = [str(v).lower().replace(" ", "") for v in vals]
            has_re = any(any(k in c for k in re_keys) for c in low)
            has_im = any(any(k in c for k in im_keys) for c in low)
            return has_re or has_im

        for i in range(nrows):
            if has_keywords(s.iloc[i].tolist()):
                return i

        for i in range(max(nrows - 1, 0)):
            # i를 헤더로 가정했을 때 i+1 이후 행에서 숫자열이 2개 이상이면 채택
            num_cols = 0
            for c in range(s.shape[1]):
                col_after = pd.to_numeric(df_raw.iloc[i + 1 :, c], errors="coerce")
                if col_after.notna().sum() >= max(2, int(len(col_after) * 0.6)):
                    num_cols += 1
            if num_cols >= 2:
                return i

        return 0

    def _coerce_eis_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [str(c).strip() for c in df.columns]
        low  = [c.lower().replace(" ", "") for c in cols]

        re_idx, im_idx = None, None
        re_keys = ("re(z", "rez", "zreal", "real", "re")
        im_keys = ("-im(z", "im(z", "zimg", "zimag", "imag", "img", "im")
        for i, c in enumerate(low):
            if re_idx is None and any(k in c for k in re_keys):
                re_idx = i
            if im_idx is None and any(k in c for k in im_keys):
                im_idx = i

        new_cols = cols[:]
        if re_idx is not None: new_cols[re_idx] = "Re(Z)/Ohm"
        if im_idx is not None: new_cols[im_idx] = "-Im(Z)/Ohm"
        df.columns = new_cols

        if "Re(Z)/Ohm" not in df.columns or "-Im(Z)/Ohm" not in df.columns:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 2:
                if "Re(Z)/Ohm" not in df.columns:
                    df.rename(columns={num_cols[0]: "Re(Z)/Ohm"}, inplace=True)
                if "-Im(Z)/Ohm" not in df.columns:
                    base = num_cols[1] if num_cols[1] != "Re(Z)/Ohm" else (num_cols[2] if len(num_cols) > 2 else num_cols[1])
                    df["-Im(Z)/Ohm"] = -pd.to_numeric(df[base], errors="coerce")

        if "-Im(Z)/Ohm" not in df.columns:
            for cand in ["Im(Z)", "Imag", "Zimag", "Zimg", "Im", "Img"]:
                if cand in df.columns:
                    df["-Im(Z)/Ohm"] = -pd.to_numeric(df[cand], errors="coerce")
                    break

        return df

    def _normalize_column_names(self, columns) -> list[str]:
        fixed = []
        seen = set()
        for i, name in enumerate(columns):
            nm = str(name).strip() if name is not None else ""
            if not nm:
                nm = f"col{i+1}"
            if nm in seen:
                nm = f"{nm}_{i+1}"
            seen.add(nm)
            fixed.append(nm)
        return fixed

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in df.columns:
            try:
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() > 0:
                    df[c] = conv
            except Exception:
                pass
        return df

    def _prompt_header_row(self, df0: pd.DataFrame, recommended: int) -> int:
        dialog = HeaderRowDialog(self, df0, recommended)
        self.wait_window(dialog)
        if dialog.result is None:
            return recommended
        return max(0, min(int(dialog.result), len(df0) - 1))

    def _cleanup_table(
        self,
        df_raw: pd.DataFrame,
        *,
        headerless: bool = True,
        allow_prompt: bool = True,
    ) -> pd.DataFrame:
        df0 = df_raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df0.empty:
            return df0
        df0 = df0.reset_index(drop=True)

        if headerless:
            hdr = self._find_header_row(df0)
            if allow_prompt and len(df0) > 1:
                hdr = self._prompt_header_row(df0, hdr)
            hdr = max(0, min(int(hdr), len(df0) - 1))
            header_vals = [str(v).strip() for v in df0.iloc[hdr].tolist()]
            fixed = self._normalize_column_names(header_vals)
            df = df0.iloc[hdr + 1 :].reset_index(drop=True)
            df.columns = fixed[: df.shape[1]]
        else:
            df = df0.copy()
            df.columns = self._normalize_column_names(df.columns)

        df = self._convert_numeric_columns(df)
        return self._coerce_eis_headers(df)

    def load_data_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Supported files", "*.xml *.csv *.txt *.mpr *.tsv *.xlsx *.xls *.xlsm"),
                ("XML", "*.xml"),
                ("CSV", "*.csv"),
                ("Text", "*.txt *.tsv"),
                ("Bio-Logic MPR", "*.mpr"),
                ("Excel", "*.xlsx *.xls *.xlsm"),
                ("All files", "*.*"),
            ]
        )
        if not path: return

        try:
            self._suspend_callbacks = True
            lower = path.lower()
            if lower.endswith((".xlsx", ".xls", ".xlsm")):
                df = self._read_excel_smart(path)
            elif lower.endswith(".xml"):
                df = self._read_xml(path)
            elif lower.endswith(".tsv"):
                df = self._read_txt_like(path)
            elif lower.endswith(".csv") or lower.endswith(".txt"):
                df = self._read_txt_like(path)
            elif lower.endswith(".mpr"):
                df = self._read_mpr(path)
            else:
                try: df = self._read_excel_smart(path)
                except Exception: df = self._read_txt_like(path)

            # 공통 후처리
            df.columns = [str(c).strip() for c in df.columns]
            if "Unnamed: 0" in df.columns:
                try:
                    if (df["Unnamed: 0"].reset_index(drop=True) == pd.RangeIndex(len(df))).all():
                        df = df.drop(columns=["Unnamed: 0"])
                except Exception: pass

            self.df = df
            self.file_label.config(text="Loaded: " + path.replace("\\","/").split("/")[-1])
            self.start_row.set("1"); self.end_row.set(str(len(self.df)))
            self.selected_columns = list(self.df.columns)
            self.select_btn.config(state="normal")

            self._auto_pick_xy_from_selection()

        except Exception as e:
            self.file_label.config(text=f"Error: {e}")
            return
        finally:
            self._suspend_callbacks = False

        self._schedule_axis_and_plot()


    def _read_csv_smart(self, path, sep=None):
        encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]
        last_err = None
        for enc in encodings:
            try:
                if sep is None:
                    return pd.read_csv(path, sep=None, engine="python", encoding=enc, header=None)
                else:
                    return pd.read_csv(path, sep=sep, encoding=enc, header=None)
            except Exception as e:
                last_err = e
                continue
        if last_err: raise last_err
        raise ValueError("Failed to read text file.")

    def _read_txt_like(self, path: str) -> pd.DataFrame:
        try:
            df_raw = self._read_csv_smart(path, sep=None)  # 구분자/인코딩 자동 + header=None
        except Exception:
            df_raw = pd.read_csv(path, delim_whitespace=True, engine="python", header=None)
        return self._cleanup_table(df_raw, headerless=True, allow_prompt=True)

    def _read_excel_smart(self, path: str) -> pd.DataFrame:
        try:
            df_raw = pd.read_excel(path, header=None)
        except Exception:
            df_raw = pd.read_excel(path, header=None, engine="openpyxl")
        return self._cleanup_table(df_raw, headerless=True, allow_prompt=True)

    def _read_xml(self, path: str) -> pd.DataFrame:
        try:
            df_raw = pd.read_xml(path)
            if df_raw is not None and not df_raw.empty:
                return self._cleanup_table(df_raw, headerless=False, allow_prompt=False)
        except Exception:
            df_raw = None

        if df_raw is None or df_raw.empty:
            try:
                import xml.etree.ElementTree as ET

                tree = ET.parse(path)
                root = tree.getroot()

                def collect_records(node):
                    children = list(node)
                    if not children:
                        return []
                    records = []
                    for child in children:
                        record = {}
                        if child.attrib:
                            record.update({k: v for k, v in child.attrib.items()})
                        sub_children = list(child)
                        if sub_children:
                            for sub in sub_children:
                                if list(sub):
                                    continue
                                text = (sub.text or "").strip()
                                if text:
                                    record[sub.tag] = text
                                for ak, av in sub.attrib.items():
                                    record[f"{sub.tag}_{ak}"] = av
                        else:
                            text = (child.text or "").strip()
                            if text:
                                record[child.tag] = text
                        if record:
                            records.append(record)
                    if records:
                        return records
                    for child in children:
                        nested = collect_records(child)
                        if nested:
                            return nested
                    return []

                records = collect_records(root)
                if not records:
                    raise ValueError("Failed to parse XML file.")
                df_raw = pd.DataFrame(records)
            except Exception as err:
                raise ValueError(f"Failed to read XML file: {err}") from err

        return self._cleanup_table(df_raw, headerless=False, allow_prompt=False)

    def _read_mpr(self, path: str) -> pd.DataFrame:
        if _eclab is not None:
            try:
                mpr = _eclab.MPRfile(path)
                dat = mpr.data
                dat = dat.rename(columns={
                    "Zreal": "Re(Z)/Ohm", "Zimg": "-Im(Z)/Ohm", "Zimag": "-Im(Z)/Ohm",
                    "Re(Z)": "Re(Z)/Ohm", "-Im(Z)": "-Im(Z)/Ohm",
                })
                if "Re(Z)/Ohm" in dat.columns and "-Im(Z)/Ohm" not in dat.columns:
                    for cand in ["Im(Z)", "Imag", "Zimag", "Zimg", "Im", "Img"]:
                        if cand in dat.columns:
                            dat["-Im(Z)/Ohm"] = -pd.to_numeric(dat[cand], errors="coerce"); break
                if "Re(Z)/Ohm" not in dat.columns or "-Im(Z)/Ohm" not in dat.columns:
                    num_cols = [c for c in dat.columns if pd.api.types.is_numeric_dtype(dat[c])]
                    if len(num_cols) >= 2:
                        dat = dat.copy()
                        dat.rename(columns={num_cols[0]: "Re(Z)/Ohm"}, inplace=True)
                        if "Re(Z)/Ohm" in dat.columns and num_cols[1] != "Re(Z)/Ohm":
                            dat["-Im(Z)/Ohm"] = -pd.to_numeric(dat[num_cols[1]], errors="coerce")
                return dat
            except Exception:
                pass
        # 텍스트로 내보낸 형태 폴백
        return self._read_txt_like(path)


    def _get_plotting_df(self):
        if self.df is None:
            return pd.DataFrame()
        try:
            s1 = int(self.start_row.get())
            e1 = int(self.end_row.get())
        except Exception:
            return self.df.copy()
        s0 = max(0, s1 - 1)
        e0 = min(len(self.df) - 1, e1 - 1)
        if e0 < s0:
            return pd.DataFrame()
        return self.df.iloc[s0:e0+1].copy()

    # ---------- axis defaults ----------
    def _calculate_axis_defaults(self, data_min, data_max):
        if data_min is None or data_max is None or pd.isna(data_min) or pd.isna(data_max):
            return "0", "10", "2"
        nice_min = 0
        rng = data_max - nice_min
        if rng <= 0: rng = data_max
        power = 10 ** math.floor(math.log10(rng if rng > 0 else 1))
        nice_max = math.ceil(data_max / power) * power
        if nice_max <= nice_min: nice_max = data_max + power
        interval = (nice_max - nice_min) / 10.0
        if interval <= 0: interval = 1.0
        return str(nice_min), str(round(nice_max, 2)), str(round(interval, 2))

    def on_x_change(self, *a):
        if getattr(self, "_booting", True) or getattr(self, "_suspend_callbacks", False):
            return

        # 이전 Y 선택값 기억
        prev_selected = set(self._get_selected_y_cols())

        # X 바뀌면 Y 후보 즉시 재구성(현재 X는 제외)
        self._refresh_axis_selectors()

        # 이전 선택 복원(여전히 존재하는 것만)
        all_items = [self.y_listbox.get(i) for i in range(self.y_listbox.size())]
        self.y_listbox.selection_clear(0, "end")
        for i, c in enumerate(all_items):
            if c in prev_selected:
                self.y_listbox.selection_set(i)

        # 자동 라벨
        if self.auto_x_label and self.x_col.get():
            self.styleable_objects["X-Axis Label"].text.set(self.x_col.get())
        y_cols = self._get_selected_y_cols()
        if self.auto_y_label and y_cols:
            self.styleable_objects["Y-Axis Label"].text.set(", ".join(y_cols))

        # 축 기본값 재계산 + 리플롯
        self._schedule_axis_and_plot()



    def _on_y_change(self, event=None):
        if getattr(self, "_booting", True) or getattr(self, "_suspend_callbacks", False):
            return

        y_cols = self._get_selected_y_cols()
        # 선택이 비면 패널 비우고 종료
        if not y_cols:
            self.series_combo["values"] = []
            self.series_combo.set("")
            for w in getattr(self, "series_dyn", []).winfo_children():
                w.destroy()
            self._schedule_axis_and_plot()
            return

        self._ensure_series_objects(y_cols)

        self.series_combo["values"] = y_cols
        if self.series_combo.get() not in y_cols:
            self.series_combo.set(y_cols[0])

        # 자동 Y 라벨
        if self.auto_y_label:
            self.styleable_objects["Y-Axis Label"].text.set(", ".join(y_cols))

        # 패널/축 갱신
        self._schedule_series_panel_build(self.series_combo.get())
        self._schedule_axis_and_plot()


    def _on_series_select(self, event=None):
        col = self.series_combo.get()
        if not col:
            return
        # 혹시라도 X축을 선택하면 무시
        if col == self.x_col.get():
            return
        # 방어적으로 상태 보장
        self._ensure_series_objects([col])
        self._schedule_series_panel_build(col)

    # ---------- color picker ----------
    def _choose_color(self, var: tk.StringVar):
        c = colorchooser.askcolor(title="Choose color")
        if c and c[1]: var.set(c[1])

    # ---------- add annotation ----------
    def add_annotation(self):
        if not self.figure: return
        self.annotations_counter += 1
        name = f"Custom Text {self.annotations_counter}"
        ax = self.figure.axes[0]
        cx, cy = np.mean(ax.get_xlim()), np.mean(ax.get_ylim())
        so = StyleableObject(
            name, 'text', StyleConfig(),
            text_var=tk.StringVar(value="New Text"),
            x_var=tk.DoubleVar(value=round(cx, 2)),
            y_var=tk.DoubleVar(value=round(cy, 2))
        )
        self.styleable_objects[name] = so
        self._refresh_custom_text_editor()
        self.plot_graph()

    def _format_number(self, value, mode, prec):
        try:
            p = int(float(prec))
        except Exception:
            p = 3
        if mode == "fixed":
            return f"{value:.{p}f}"
        if mode == "sci":
            return f"{value:.{p}e}"
        # sigfigs
        return f"{value:.{p}g}"

    def _format_equation(self, m, c, r2, ysym, xsym, mode, prec, hide_c_eps, include_r2):
        try:
            eps = abs(float(hide_c_eps))
        except Exception:
            eps = 0.0
        c_use = 0.0 if abs(c) <= eps else c

        ms = self._format_number(m, mode, prec)
        body = f"{ysym} = {ms}{xsym}"

        if c_use != 0:
            cs = self._format_number(c_use, mode, prec)
            sign = "+" if c_use >= 0 else ""
            body += f" {sign} {cs}"

        if include_r2 and r2 is not None:
            r2s = self._format_number(r2, "fixed", 3)
            body += f", R²={r2s}"
        return body

    # ---------- plotting ----------
    def update_object_position(self, artist, pos):
        name = self.artist_map.get(artist)
        if name and name in self.styleable_objects:
            obj = self.styleable_objects[name]
            if obj.x and obj.y:
                obj.x.set(round(pos[0], 2)); obj.y.set(round(pos[1], 2))
        for col, so in self.equation_styles.items():
            if self.artist_map.get(artist) == f"Equation[{col}]":
                so.x.set(round(pos[0], 2)); so.y.set(round(pos[1], 2))

    def plot_graph(self, *args):
        if self._is_plotting:
            return
        self._is_plotting = True
        old_fig = self.figure

        try:
            # ── 이전 밑줄 라인 제거 ──
            if hasattr(self, "_underline_lines") and self._underline_lines:
                for ln in self._underline_lines:
                    try:
                        ln.remove()
                    except Exception:
                        pass
                self._underline_lines.clear()

            # 드래그/캔버스 정리
            if self.dragger:
                self.dragger.disconnect()
                self.dragger = None
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None

            # 새 Figure
            try:
                w_in = max(float(self.figure_width_cm.get()) / 2.54, 2.0)
                h_in = max(float(self.figure_height_cm.get()) / 2.54, 2.0)
                self.figure, ax = plt.subplots(figsize=(w_in, h_in),
                                            dpi=self.screen_dpi,
                                            constrained_layout=True)
            except Exception:
                self.figure, ax = plt.subplots(dpi=self.screen_dpi, constrained_layout=True)

            # 데이터 체크
            if self.df is None:
                t = ax.text(0.5, 0.5, "Please load a file.", ha='center', va='center', transform=ax.transAxes)
                underline_targets = []
                # (필요 시) 타이틀 등에도 밑줄을 추가하려면 targets에 append
                self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
                self.canvas.draw()  # 1차 draw
                for a, aax, pad in underline_targets:
                    self._add_underline_line(a, aax, pad_px=pad)
                self.canvas.draw()  # 2차 draw
                self._mount_canvas()
                return

            pdf   = self._get_plotting_df()
            xname = self.x_col.get()
            y_cols = self._get_selected_y_cols()

            if pdf.empty or (not xname) or (not y_cols):
                t = ax.text(0.5, 0.5, "Select X and Y columns to plot.", ha='center', va='center', transform=ax.transAxes)
                underline_targets = []
                self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
                self.canvas.draw()  # 1차 draw
                for a, aax, pad in underline_targets:
                    self._add_underline_line(a, aax, pad_px=pad)
                self.canvas.draw()  # 2차 draw
                self._mount_canvas()
                return

            self._ensure_series_objects(y_cols)

            # ---------- X 처리 ----------
            xraw = pdf[xname]
            x_is_datetime = False
            try:
                xdt = pd.to_datetime(xraw, errors="coerce")
                if xdt.notna().sum() >= max(2, int(len(xdt) * 0.6)):
                    x_is_datetime = True
                    xnum = mdates.date2num(xdt.to_numpy())
                else:
                    xnum = pd.to_numeric(xraw, errors="coerce").to_numpy(dtype=float)
            except Exception:
                xnum = pd.to_numeric(xraw, errors="coerce").to_numpy(dtype=float)

            draggable_artists = []
            self.artist_map = {}
            underline_targets = []  # (artist, ax, pad_px)

            # ---------- 시리즈 ----------
            for i, col in enumerate(y_cols):
                yarr = pd.to_numeric(pdf[col], errors="coerce").to_numpy(dtype=float)

                mask = np.isfinite(xnum) & np.isfinite(yarr)
                xv = xnum[mask]
                yv = yarr[mask]
                if xv.size == 0:
                    continue

                series_label = str(col)

                # 회귀 (m, c, R^2)
                m = c = None
                r2 = None
                eq_text = ""
                if xv.size >= 2:
                    try:
                        if self.fit_intercept.get():
                            m, c = np.polyfit(xv, yv, 1)
                            y_pred = m * xv + c
                        else:
                            m = np.linalg.lstsq(xv.reshape(-1, 1), yv, rcond=None)[0][0]
                            c = 0.0
                            y_pred = m * xv
                        y_mean = np.mean(yv)
                        ss_tot = np.sum((yv - y_mean) ** 2)
                        ss_res = np.sum((yv - y_pred) ** 2)
                        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

                        ysym = (self.series_eq_y_symbol[col].get() or "y")
                        xsym = (self.series_eq_x_symbol[col].get() or "x")
                        mode = (self.series_eq_format_mode[col].get() or "sigfigs").lower()
                        prec = self.series_eq_precision[col].get()
                        hide_eps = self.series_eq_hide_c_eps[col].get()
                        include_r2 = self.series_eq_include_r2[col].get()

                        eq_text = self._format_equation(
                            m=m, c=c, r2=r2,
                            ysym=ysym, xsym=xsym,
                            mode=mode, prec=prec,
                            hide_c_eps=hide_eps,
                            include_r2=include_r2
                        )
                    except Exception:
                        m = c = None
                        r2 = None
                        eq_text = ""

                # 최신 계산식 저장 및 자동 동기화
                self.series_last_eq[col].set(eq_text)
                if self.series_eq_auto_update[col].get() and eq_text:
                    self.equation_styles[col].text.set(eq_text)

                # 데이터 라인
                style = self.series_styles[col].style.get_line_dict()
                line, = ax.plot(xv, yv, label=series_label, **style)

                # 트렌드라인
                if self.series_show_trend[col].get() and (m is not None):
                    xt = np.array([np.nanmin(xv), np.nanmax(xv)], dtype=float)
                    yt = m * xt + c
                    tline = ax.plot(xt, yt, **self.trendline_styles[col].style.get_line_dict())[0]
                    try: tline.set_picker(5)
                    except Exception: pass
                    draggable_artists.append(tline)
                    self.artist_map[tline] = f"Trendline[{col}]"

                # 범례 라벨 구성
                place = (self.series_eq_place[col].get() or "legend").lower()
                if self.series_use_custom_legend[col].get():
                    lbl = (self.series_custom_legend_text[col].get() or "").strip()
                    line.set_label(lbl if lbl else series_label)
                else:
                    if place == "legend" and eq_text:
                        tpl = (self.legend_label_template.get() or "{col} ({eq})").strip()
                        try:
                            line.set_label(tpl.format(col=col, eq=eq_text))
                        except Exception:
                            line.set_label(f"{col} ({eq_text})")
                    else:
                        line.set_label(series_label)

                # on-plot 텍스트
                if place == "onplot":
                    so = self.equation_styles[col]
                    text_to_show = (so.text.get() or "").strip() or eq_text or f"{col}: insufficient data"
                    try:
                        x0 = float(so.x.get()); y0 = float(so.y.get())
                    except Exception:
                        x0 = y0 = 0.0
                    if (x0 == 0.0 and y0 == 0.0) and xv.size > 0:
                        try:
                            so.x.set(float(np.nanmean(xv)))
                            so.y.set(float(np.nanmax(yv)))
                        except Exception:
                            pass
                    a = ax.text(so.x.get(), so.y.get(), text_to_show,
                                **so.style.get_font_dict(), clip_on=True)
                    try: a.set_in_layout(False)
                    except Exception: pass
                    if so.style.underline.get():
                        underline_targets.append((a, ax, 2))
                    draggable_artists.append(a)
                    self.artist_map[a] = f"Equation[{col}]"

            # ── 타이틀/축 라벨 ──
            for key in ["Title", "X-Axis Label", "Y-Axis Label"]:
                obj = self.styleable_objects[key]
                a = None
                if key == "Title":
                    a = ax.set_title(obj.text.get(), **obj.style.get_font_dict())
                elif key == "X-Axis Label":
                    a = ax.set_xlabel(obj.text.get(), **obj.style.get_font_dict())
                elif key == "Y-Axis Label":
                    a = ax.set_ylabel(obj.text.get(), **obj.style.get_font_dict())
                if a and obj.style.underline.get():
                    underline_targets.append((a, ax, 2))

            # ── Tick 스타일 ──
            xts = self.styleable_objects["X-Tick Labels"].style.get_font_dict()
            yts = self.styleable_objects["Y-Tick Labels"].style.get_font_dict()
            ax.tick_params(axis='x', labelsize=xts['fontsize'], labelcolor=xts['color'])
            plt.setp(ax.get_xticklabels(), fontweight=xts['fontweight'], fontstyle=xts['fontstyle'])
            ax.tick_params(axis='y', labelsize=yts['fontsize'], labelcolor=yts['color'])
            plt.setp(ax.get_yticklabels(), fontweight=yts['fontweight'], fontstyle=yts['fontstyle'])
            if self.rotate_y_ticks_90.get():
                plt.setp(ax.get_yticklabels(), rotation=90, va='center')

            # ── 축 범위/간격 ──
            try:
                ymin, ymax = float(self.y_min.get()), float(self.y_max.get())
                yint = float(self.y_interval.get())
                ax.set_ylim(ymin, ymax)
                if yint > 0:
                    ax.yaxis.set_major_locator(mticker.MultipleLocator(yint))
            except Exception:
                pass

            try:
                if x_is_datetime:
                    xmin_dt = pd.to_datetime(self.x_min.get(), errors="coerce")
                    xmax_dt = pd.to_datetime(self.x_max.get(), errors="coerce")
                    if pd.notna(xmin_dt) and pd.notna(xmax_dt):
                        ax.set_xlim(mdates.date2num(xmin_dt.to_pydatetime()),
                                    mdates.date2num(xmax_dt.to_pydatetime()))
                    locator = mdates.AutoDateLocator()
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
                    ax.xaxis_date()
                else:
                    xmin, xmax = float(self.x_min.get()), float(self.x_max.get())
                    xint = float(self.x_interval.get())
                    ax.set_xlim(xmin, xmax)
                    if xint > 0:
                        ax.xaxis.set_major_locator(mticker.MultipleLocator(xint))
            except Exception:
                pass

            # ── 그리드/범례 ──
            ax.grid(False)
            if self.show_x_grid.get(): ax.xaxis.grid(True, linestyle='--', alpha=0.5)
            if self.show_y_grid.get(): ax.yaxis.grid(True, linestyle='--', alpha=0.5)

            if self.show_legend.get():
                try: fs = float(self.legend_fontsize.get())
                except Exception: fs = 10.0
                try: lw = float(self.legend_linewidth.get())
                except Exception: lw = 0.8
                leg = ax.legend(loc=self.legend_loc.get(),
                                fontsize=fs,
                                frameon=self.legend_frameon.get(),
                                fancybox=self.legend_fancybox.get(),
                                shadow=self.legend_shadow.get())
                if leg:
                    leg.set_draggable(True)
                    try: leg.set_in_layout(False)
                    except Exception: pass
                    frame = leg.get_frame()
                    frame.set_linewidth(lw)
                    frame.set_edgecolor(self.legend_edgecolor.get())
                    frame.set_facecolor(self.legend_facecolor.get())

            # ── 1차 렌더: 텍스트 bbox 확정 ──
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.draw()

            # ── 밑줄 생성 ──
            for a, aax, pad in underline_targets:
                self._add_underline_line(a, aax, pad_px=pad)

            # ── 2차 렌더: 밑줄 반영 ──
            self.canvas.draw()
            self._mount_canvas()

            # 드래그 연결
            if draggable_artists:
                self.dragger = DraggableArtist(draggable_artists, self.update_object_position)

        finally:
            self._is_plotting = False
            if old_fig is not None and old_fig is not self.figure:
                plt.close(old_fig)


    def save_graph(self):
        if not self.figure: return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("SVG (Vector)","*.svg"),
                    ("PDF (Vector)","*.pdf"),("EPS (Vector)","*.eps")]
        )
        if path:
            self.figure.savefig(path, dpi=300)

# ---------- run ----------
if __name__ == "__main__":
    app = GraphMaker()
    app.mainloop()