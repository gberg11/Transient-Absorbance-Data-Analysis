#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class DataPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Plotter with Savitzky-Golay Filter")

        self.load_button = tk.Button(root, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=0, columnspan=2, pady=10)

        self.use_filter = tk.BooleanVar()
        self.filter_checkbox = tk.Checkbutton(root, text="Use Savitzky-Golay Filter", variable=self.use_filter)
        self.filter_checkbox.grid(row=1, column=0, columnspan=2)

        self.plot_raw_data = tk.BooleanVar()
        self.plot_raw_data_checkbox = tk.Checkbutton(root, text="Plot Raw Data", variable=self.plot_raw_data)
        self.plot_raw_data_checkbox.grid(row=2, column=0, columnspan=2)

        self.window_length_label = tk.Label(root, text="Window Length:")
        self.window_length_label.grid(row=3, column=0, sticky="E", padx=(10, 0), pady=5)
        self.window_length_entry = tk.Entry(root, width=10)
        self.window_length_entry.insert(0, "51")
        self.window_length_entry.grid(row=3, column=1, sticky="W", padx=(0, 10), pady=5)

        self.polyorder_label = tk.Label(root, text="Polynomial Order:")
        self.polyorder_label.grid(row=4, column=0, sticky="E", padx=(10, 0), pady=5)
        self.polyorder_entry = tk.Entry(root, width=10)
        self.polyorder_entry.insert(0, "3")
        self.polyorder_entry.grid(row=4, column=1, sticky="W", padx=(0, 10), pady=5)

        self.x_range_label = tk.Label(root, text="Wavelength Range:")
        self.x_range_label.grid(row=5, column=0, sticky="E", padx=(10, 0), pady=5)
        self.x_min_entry = tk.Entry(root, width=10)
        self.x_min_entry.insert(0, "400")
        self.x_min_entry.grid(row=5, column=1, sticky="W", padx=(0, 10), pady=5)
        self.x_max_entry = tk.Entry(root, width=10)
        self.x_max_entry.insert(0, "700")
        self.x_max_entry.grid(row=6, column=1, sticky="W", padx=(0, 10), pady=5)

        self.y_range_label = tk.Label(root, text="ΔA Range:")
        self.y_range_label.grid(row=7, column=0, sticky="E", padx=(10, 0), pady=5)
        self.y_min_entry = tk.Entry(root, width=10)
        self.y_min_entry.insert(0, "0")
        self.y_min_entry.grid(row=7, column=1, sticky="W", padx=(0, 10), pady=5)
        self.y_max_entry = tk.Entry(root, width=10)
        self.y_max_entry.insert(0, "10")
        self.y_max_entry.grid(row=8, column=1, sticky="W", padx=(0, 10), pady=5)

        self.plot_button = tk.Button(root, text="Plot Data", command=self.plot_data)
        self.plot_button.grid(row=9, column=0, columnspan=2, pady=10)

        self.save_button = tk.Button(root, text="Save Plot", command=self.save_plot)
        self.save_button.grid(row=10, column=0, columnspan=2, pady=10)

        self.data = None
        self.canvas = None
        self.ax = None
        self.toolbar = None

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}")

    def plot_data(self):
        if self.data is not None:
            try:
                window_length = int(self.window_length_entry.get())
                polyorder = int(self.polyorder_entry.get())
                x_min = float(self.x_min_entry.get())
                x_max = float(self.x_max_entry.get())
                y_min = float(self.y_min_entry.get())
                y_max = float(self.y_max_entry.get())

                if window_length % 2 == 0:
                    raise ValueError("Window length must be an odd number.")

                if self.canvas:
                    self.canvas.get_tk_widget().destroy()
                if self.toolbar:
                    self.toolbar.destroy()
                plt.close('all')  # Close all existing plots

                fig, self.ax = plt.subplots(figsize=(6.5, 3.25))
                colormap = plt.cm.magma
                num_traces = len(self.data.columns) // 2
                colors = colormap(np.linspace(0, 1, num_traces))

                for i, color in zip(range(0, len(self.data.columns), 2), colors):
                    wavelength_col = self.data.columns[i]
                    data_col = self.data.columns[i + 1]
                    delay_time = data_col.split(',')[1].strip()

                    if self.use_filter.get():
                        smoothed_data = savgol_filter(self.data[data_col], window_length=window_length, polyorder=polyorder)
                        self.ax.plot(self.data[wavelength_col], smoothed_data, label=f'{delay_time}', color=color, linewidth=1.5, clip_on=True)
                        if self.plot_raw_data.get():
                            self.ax.plot(self.data[wavelength_col], self.data[data_col], 'o', markersize=2, color=color, alpha=0.5)
                    else:
                        self.ax.plot(self.data[wavelength_col], self.data[data_col], label=f'{delay_time}', color=color, linewidth=1.5, clip_on=True)

                self.ax.set_xlabel('Wavelength (nm)')
                self.ax.set_ylabel('ΔA (mOD)')
                self.ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.3)

                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_min, y_max)

                plt.tight_layout()

                self.canvas = FigureCanvasTkAgg(fig, master=self.root)
                self.canvas.draw()
                self.canvas.get_tk_widget().grid(row=11, column=0, columnspan=2)

                self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
                self.toolbar.update()
                self.toolbar.grid(row=12, column=0, columnspan=2)

            except ValueError as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showwarning("Warning", "No data loaded. Please load a CSV file first.")

    def save_plot(self):
        if self.canvas:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")])
            if file_path:
                self.canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    root = tk.Tk()
    app = DataPlotterApp(root)
    root.mainloop()