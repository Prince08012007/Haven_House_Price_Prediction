"""Desktop GUI for Ames House Price Prediction ML.
Native Windows desktop application built with Tkinter and ttk.
"""

from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import predict_single, predict_batch, find_data_file

NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
    "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill",
    "NWAmes", "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer", "SawyerW",
    "Somerst", "StoneBr", "Timber", "Veenker"
]

PRESETS = {
    "💎 Luxury Estate": {
        "OverallQual": 9, "OverallCond": 6, "GrLivArea": 3200, "TotalBsmtSF": 1800,
        "1stFlrSF": 1800, "2ndFlrSF": 1400, "YearBuilt": 2016, "YearRemodAdd": 2018,
        "FullBath": 3, "HalfBath": 1, "TotRmsAbvGrd": 10, "BedroomAbvGr": 4,
        "GarageCars": 3, "GarageArea": 850, "Fireplaces": 2, "Neighborhood": "NridgHt",
        "KitchenQual": "Ex", "ExterQual": "Ex", "BsmtQual": "Ex", "CentralAir": "Y",
    },
    "🏡 Suburban Family": {
        "OverallQual": 7, "OverallCond": 5, "GrLivArea": 1950, "TotalBsmtSF": 1050,
        "1stFlrSF": 1050, "2ndFlrSF": 900, "YearBuilt": 2004, "YearRemodAdd": 2005,
        "FullBath": 2, "HalfBath": 1, "TotRmsAbvGrd": 7, "BedroomAbvGr": 3,
        "GarageCars": 2, "GarageArea": 500, "Fireplaces": 1, "Neighborhood": "CollgCr",
        "KitchenQual": "Gd", "ExterQual": "Gd", "BsmtQual": "Gd", "CentralAir": "Y",
    },
    "🏘️ Starter Home": {
        "OverallQual": 6, "OverallCond": 5, "GrLivArea": 1300, "TotalBsmtSF": 850,
        "1stFlrSF": 850, "2ndFlrSF": 450, "YearBuilt": 1996, "YearRemodAdd": 1998,
        "FullBath": 1, "HalfBath": 1, "TotRmsAbvGrd": 6, "BedroomAbvGr": 3,
        "GarageCars": 1, "GarageArea": 320, "Fireplaces": 0, "Neighborhood": "SawyerW",
        "KitchenQual": "TA", "ExterQual": "TA", "BsmtQual": "TA", "CentralAir": "Y",
    },
}


class HousePriceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🏡 House Price Prediction ML — Desktop Application")
        self.geometry("980x750")
        self.minsize(850, 650)
        self.configure(bg="#F3F4F6")

        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook", background="#F3F4F6")
        style.configure("TNotebook.Tab", padding=[16, 8], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#2563EB")], foreground=[("selected", "#FFFFFF")])

        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background="#F3F4F6", foreground="#1E3A8A")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 9), background="#F3F4F6", foreground="#4B5563")
        style.configure("Card.TFrame", background="#FFFFFF", relief="ridge")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#2563EB", foreground="#FFFFFF")
        style.map("Primary.TButton", background=[("active", "#1D4ED8")])

    def _build_ui(self):
        # Header banner
        header_frame = tk.Frame(self, bg="#1E3A8A", height=70)
        header_frame.pack(fill="x")

        title_lbl = tk.Label(
            header_frame,
            text="🏡 Ames House Price Valuation & Prediction Suite",
            font=("Segoe UI", 15, "bold"),
            fg="#FFFFFF",
            bg="#1E3A8A",
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        subtitle_lbl = tk.Label(
            header_frame,
            text="Powered by Ensemble Learning (CatBoost + XGBoost + GradientBoosting)",
            font=("Segoe UI", 9),
            fg="#93C5FD",
            bg="#1E3A8A",
        )
        subtitle_lbl.pack(side="right", padx=20, pady=12)

        # Tab notebook
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=15, pady=15)

        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        tab3 = ttk.Frame(notebook)

        notebook.add(tab1, text="  🏠 Property Valuation  ")
        notebook.add(tab2, text="  📁 Batch Predictions  ")
        notebook.add(tab3, text="  📊 Model Benchmarks  ")

        self._build_tab1(tab1)
        self._build_tab2(tab2)
        self._build_tab3(tab3)

    def _build_tab1(self, parent):
        # Top toolbar with presets and model selection
        toolbar = tk.Frame(parent, bg="#E5E7EB", pady=8, padx=10)
        toolbar.pack(fill="x", padx=10, pady=(10, 5))

        tk.Label(toolbar, text="Quick Presets:", bg="#E5E7EB", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))

        for name in PRESETS:
            btn = tk.Button(
                toolbar,
                text=name,
                font=("Segoe UI", 9),
                bg="#FFFFFF",
                relief="groove",
                command=lambda n=name: self._apply_preset(n),
            )
            btn.pack(side="left", padx=4)

        tk.Label(toolbar, text="Model:", bg="#E5E7EB", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(25, 6))
        self.model_var = tk.StringVar(value="Ensemble Blend")
        model_dropdown = ttk.Combobox(
            toolbar,
            textvariable=self.model_var,
            values=["Ensemble Blend", "CatBoost", "XGBoost", "GradientBoosting"],
            state="readonly",
            width=18,
        )
        model_dropdown.pack(side="left")

        # Main content area: Left column (Inputs) & Right column (Results)
        panes = tk.PanedWindow(parent, orient="horizontal", sashrelief="groove", sashwidth=4)
        panes.pack(fill="both", expand=True, padx=10, pady=5)

        input_frame = tk.Frame(panes, bg="#FFFFFF", padx=12, pady=12)
        result_frame = tk.Frame(panes, bg="#FFFFFF", padx=15, pady=15)

        panes.add(input_frame, minsize=480)
        panes.add(result_frame, minsize=320)

        # Build Form Inputs inside a scrollable frame or grouped sections
        self._build_form(input_frame)
        self._build_result_panel(result_frame)

    def _build_form(self, container):
        tk.Label(container, text="Property Specifications", font=("Segoe UI", 12, "bold"), bg="#FFFFFF", fg="#1E3A8A").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10)
        )

        self.inputs = {}

        fields = [
            # label, key, default, row, col, type, options/range
            ("Living Area (sq ft):", "GrLivArea", 1850, 1, 0, "spin", (400, 6000, 50)),
            ("Basement Area (sq ft):", "TotalBsmtSF", 1000, 1, 2, "spin", (0, 4000, 50)),
            ("1st Floor (sq ft):", "1stFlrSF", 1050, 2, 0, "spin", (400, 4000, 50)),
            ("2nd Floor (sq ft):", "2ndFlrSF", 800, 2, 2, "spin", (0, 3000, 50)),
            ("Overall Quality (1-10):", "OverallQual", 7, 3, 0, "spin", (1, 10, 1)),
            ("Overall Cond (1-10):", "OverallCond", 5, 3, 2, "spin", (1, 10, 1)),
            ("Year Built:", "YearBuilt", 2005, 4, 0, "spin", (1880, 2026, 1)),
            ("Year Remodeled:", "YearRemodAdd", 2006, 4, 2, "spin", (1950, 2026, 1)),
            ("Full Bathrooms:", "FullBath", 2, 5, 0, "spin", (0, 5, 1)),
            ("Half Bathrooms:", "HalfBath", 1, 5, 2, "spin", (0, 4, 1)),
            ("Bedrooms Above Grade:", "BedroomAbvGr", 3, 6, 0, "spin", (1, 8, 1)),
            ("Total Rooms Above:", "TotRmsAbvGrd", 7, 6, 2, "spin", (2, 16, 1)),
            ("Garage Cars:", "GarageCars", 2, 7, 0, "spin", (0, 5, 1)),
            ("Garage Area (sq ft):", "GarageArea", 500, 7, 2, "spin", (0, 1500, 50)),
            ("Fireplaces:", "Fireplaces", 1, 8, 0, "spin", (0, 4, 1)),
            ("Central Air:", "CentralAir", "Y", 8, 2, "combo", ["Y", "N"]),
            ("Neighborhood:", "Neighborhood", "CollgCr", 9, 0, "combo", NEIGHBORHOODS),
            ("Kitchen Quality:", "KitchenQual", "Gd", 9, 2, "combo", ["Ex", "Gd", "TA", "Fa", "Po"]),
            ("Exterior Quality:", "ExterQual", "Gd", 10, 0, "combo", ["Ex", "Gd", "TA", "Fa", "Po"]),
            ("Basement Quality:", "BsmtQual", "Gd", 10, 2, "combo", ["Ex", "Gd", "TA", "Fa", "Po"]),
        ]

        for label, key, default, r, c, kind, extra in fields:
            lbl = tk.Label(container, text=label, bg="#FFFFFF", font=("Segoe UI", 9))
            lbl.grid(row=r, column=c, sticky="e", padx=(8, 4), pady=5)

            if kind == "spin":
                low, high, step = extra
                var = tk.IntVar(value=default)
                widget = ttk.Spinbox(container, from_=low, to=high, increment=step, textvariable=var, width=12)
            elif kind == "combo":
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(container, textvariable=var, values=extra, state="readonly", width=12)
            else:
                var = tk.StringVar(value=str(default))
                widget = ttk.Entry(container, textvariable=var, width=12)

            widget.grid(row=r, column=c + 1, sticky="w", padx=(2, 10), pady=5)
            self.inputs[key] = var

        calc_btn = tk.Button(
            container,
            text="💰 Calculate Estimated House Value",
            bg="#2563EB",
            fg="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            pady=8,
            command=self._calculate_valuation,
        )
        calc_btn.grid(row=11, column=0, columnspan=4, sticky="we", padx=10, pady=(15, 5))

    def _build_result_panel(self, container):
        tk.Label(container, text="Valuation Results", font=("Segoe UI", 12, "bold"), bg="#FFFFFF", fg="#1E3A8A").pack(
            anchor="w", pady=(0, 10)
        )

        # Price Display Card
        price_card = tk.Frame(container, bg="#1E40AF", padx=15, pady=20)
        price_card.pack(fill="x", pady=5)

        tk.Label(price_card, text="ESTIMATED MARKET VALUATION", font=("Segoe UI", 9, "bold"), fg="#DBEAFE", bg="#1E40AF").pack()

        self.price_label = tk.Label(price_card, text="$0", font=("Segoe UI", 26, "bold"), fg="#FFFFFF", bg="#1E40AF")
        self.price_label.pack(pady=4)

        self.range_label = tk.Label(price_card, text="Expected Range: $0 - $0", font=("Segoe UI", 9), fg="#93C5FD", bg="#1E40AF")
        self.range_label.pack()

        # Stats breakdown
        self.stats_frame = tk.Frame(container, bg="#F9FAFB", relief="groove", bd=1, padx=10, pady=10)
        self.stats_frame.pack(fill="x", pady=10)

        self.sqft_rate_label = tk.Label(self.stats_frame, text="Price / sq ft: --", bg="#F9FAFB", font=("Segoe UI", 9, "bold"))
        self.sqft_rate_label.pack(anchor="w", pady=2)

        self.diff_median_label = tk.Label(self.stats_frame, text="Vs Ames Median ($163,000): --", bg="#F9FAFB", font=("Segoe UI", 9))
        self.diff_median_label.pack(anchor="w", pady=2)

        # Model Breakdown List
        tk.Label(container, text="Model Ensemble Breakdown:", font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#374151").pack(
            anchor="w", pady=(10, 4)
        )

        self.models_tree = ttk.Treeview(container, columns=("Model", "Price"), show="headings", height=4)
        self.models_tree.heading("Model", text="Algorithm")
        self.models_tree.heading("Price", text="Predicted Price")
        self.models_tree.column("Model", width=140)
        self.models_tree.column("Price", width=120, anchor="e")
        self.models_tree.pack(fill="x")

        # Initial trigger
        self.after(200, self._calculate_valuation)

    def _apply_preset(self, name):
        preset = PRESETS.get(name)
        if not preset:
            return
        for k, v in preset.items():
            if k in self.inputs:
                self.inputs[k].set(v)
        self._calculate_valuation()

    def _calculate_valuation(self):
        user_dict = {k: var.get() for k, var in self.inputs.items()}
        model_name = self.model_var.get()

        try:
            res = predict_single(user_dict, model_name=model_name)
            price = res["predicted_price"]
            low = res["price_low"]
            high = res["price_high"]

            self.price_label.config(text=f"${price:,.0f}")
            self.range_label.config(text=f"Expected Range: ${low:,.0f} – ${high:,.0f}")

            liv_area = max(int(user_dict.get("GrLivArea", 1)), 1)
            rate = price / liv_area
            self.sqft_rate_label.config(text=f"Price / sq ft: ${rate:.1f} / sq ft")

            ames_median = 163000
            diff = ((price - ames_median) / ames_median) * 100
            self.diff_median_label.config(text=f"Vs Ames Median ($163,000): {diff:+.1f}%")

            # Update Treeview
            for item in self.models_tree.get_children():
                self.models_tree.delete(item)

            for m_name, m_price in res["model_breakdown"].items():
                self.models_tree.insert("", "end", values=(m_name, f"${m_price:,.0f}"))

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error evaluating property:\n{e}")

    def _build_tab2(self, parent):
        frame = tk.Frame(parent, bg="#FFFFFF", padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Batch Prediction on Kaggle Test Data", font=("Segoe UI", 13, "bold"), bg="#FFFFFF", fg="#1E3A8A").pack(anchor="w")
        tk.Label(frame, text="Load a test CSV file and generate predictions ready for Kaggle submission.", bg="#FFFFFF", font=("Segoe UI", 9), fg="#6B7280").pack(anchor="w", pady=(0, 15))

        btn_bar = tk.Frame(frame, bg="#FFFFFF")
        btn_bar.pack(fill="x", pady=5)

        tk.Button(btn_bar, text="📂 Load Included test.csv", font=("Segoe UI", 9), command=self._load_default_test).pack(side="left", padx=5)
        tk.Button(btn_bar, text="📁 Choose Custom CSV...", font=("Segoe UI", 9), command=self._browse_custom_csv).pack(side="left", padx=5)
        self.batch_run_btn = tk.Button(btn_bar, text="🚀 Run Batch Predictions", bg="#2563EB", fg="#FFFFFF", font=("Segoe UI", 9, "bold"), command=self._run_batch)
        self.batch_run_btn.pack(side="left", padx=10)

        self.batch_status_lbl = tk.Label(frame, text="No dataset loaded.", bg="#FFFFFF", font=("Segoe UI", 9, "italic"), fg="#4B5563")
        self.batch_status_lbl.pack(anchor="w", pady=5)

        self.batch_df = None
        self.sub_df = None

        # Table for preview
        self.batch_tree = ttk.Treeview(frame, columns=("Id", "SalePrice"), show="headings", height=14)
        self.batch_tree.heading("Id", text="Property Id")
        self.batch_tree.heading("SalePrice", text="Predicted SalePrice ($)")
        self.batch_tree.column("Id", width=120)
        self.batch_tree.column("SalePrice", width=180, anchor="e")
        self.batch_tree.pack(fill="both", expand=True, pady=10)

        # Export button
        self.export_btn = tk.Button(frame, text="📥 Export Predictions to CSV", state="disabled", font=("Segoe UI", 9, "bold"), bg="#10B981", fg="#FFFFFF", command=self._export_batch)
        self.export_btn.pack(anchor="e", pady=5)

    def _load_default_test(self):
        p = find_data_file("test.csv")
        if p.exists():
            self.batch_df = pd.read_csv(p)
            self.batch_status_lbl.config(text=f"Loaded {p.name} ({len(self.batch_df)} rows). Click 'Run Batch Predictions' to process.")
        else:
            messagebox.showwarning("File Missing", "test.csv not found in data directory.")

    def _browse_custom_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.batch_df = pd.read_csv(filename)
            self.batch_status_lbl.config(text=f"Loaded {Path(filename).name} ({len(self.batch_df)} rows).")

    def _run_batch(self):
        if self.batch_df is None:
            messagebox.showinfo("Select Data", "Please load a test CSV first.")
            return

        def task():
            self.batch_status_lbl.config(text="Computing predictions with Ensemble Blend...")
            try:
                self.sub_df = predict_batch(self.batch_df, model_name=self.model_var.get())
                self.after(0, self._render_batch_results)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Batch Error", str(e)))

        threading.Thread(target=task, daemon=True).start()

    def _render_batch_results(self):
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

        preview_rows = self.sub_df.head(200)
        for _, row in preview_rows.iterrows():
            self.batch_tree.insert("", "end", values=(int(row["Id"]), f"${row['SalePrice']:,.2f}"))

        self.batch_status_lbl.config(text=f"Predicted {len(self.sub_df)} rows. Mean Price: ${self.sub_df['SalePrice'].mean():,.0f} | Median: ${self.sub_df['SalePrice'].median():,.0f}")
        self.export_btn.config(state="normal")

    def _export_batch(self):
        if self.sub_df is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="submission_ensemble.csv")
        if save_path:
            self.sub_df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Saved predictions to:\n{save_path}")

    def _build_tab3(self, parent):
        frame = tk.Frame(parent, bg="#FFFFFF", padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Machine Learning Benchmark Leaderboard", font=("Segoe UI", 13, "bold"), bg="#FFFFFF", fg="#1E3A8A").pack(anchor="w")
        tk.Label(frame, text="Empirical test performance on the Ames Housing competition benchmark.", bg="#FFFFFF", font=("Segoe UI", 9), fg="#6B7280").pack(anchor="w", pady=(0, 10))

        scores = [
            ("V2 Weighted Blend (CatBoost + XGBoost)", "0.12320", "0.12427", "🏆 Best Kaggle Score"),
            ("V3 Weighted Blend (SLSQP)", "0.12444", "0.12444", "🥈 2nd Best"),
            ("Gradient Boosting (Optuna Tuned)", "0.12291", "0.12600", "Top Single Model"),
            ("CatBoost Regressor", "0.12592", "0.12605", "Base Booster"),
            ("Stacking Regressor (Ridge Meta)", "0.12610", "0.12610", "Meta Ensemble"),
            ("XGBoost Regressor", "0.12728", "0.12750", "Base Booster"),
            ("Gradient Boosting (Baseline)", "0.12899", "0.12980", "Baseline"),
            ("Ridge Regression", "0.13026", "0.13026", "Linear Baseline"),
            ("LightGBM", "0.13108", "0.13200", "Tree Boosting"),
            ("Extra Trees Regressor", "0.13661", "0.13750", "Randomized Trees"),
            ("Random Forest Regressor", "0.13825", "0.13900", "Bagged Trees"),
        ]

        tree = ttk.Treeview(frame, columns=("Model", "CV_RMSE", "Kaggle_RMSE", "Rank"), show="headings", height=12)
        tree.heading("Model", text="Algorithm")
        tree.heading("CV_RMSE", text="CV RMSE (log)")
        tree.heading("Kaggle_RMSE", text="Kaggle RMSE (log)")
        tree.heading("Rank", text="Status / Rank")

        tree.column("Model", width=320)
        tree.column("CV_RMSE", width=120, anchor="center")
        tree.column("Kaggle_RMSE", width=130, anchor="center")
        tree.column("Rank", width=160)

        for row in scores:
            tree.insert("", "end", values=row)

        tree.pack(fill="x", pady=10)


if __name__ == "__main__":
    app = HousePriceApp()
    app.mainloop()
