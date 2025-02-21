import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.animation import FuncAnimation
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QLabel, QCheckBox, QTextEdit, QLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QSize, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap
import seaborn as sns
import pandas as pd
from scipy import stats
import sys
import logging
import io

# Configure logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom FlowLayout for dynamic button placement
class FlowLayout(QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.item_list = []

    def addItem(self, item):
        self.item_list.append(item)

    def count(self):
        return len(self.item_list)

    def itemAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)  # No expanding directions by default

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        try:
            height = self.do_layout(QRect(0, 0, width, 0), True)
            return height
        except Exception as e:
            logging.error(f"Error in heightForWidth: {e}")
            return 0

    def setGeometry(self, rect):
        try:
            super().setGeometry(rect)
            self.do_layout(rect, False)
        except Exception as e:
            logging.error(f"Error in setGeometry: {e}")

    def sizeHint(self):
        try:
            return self.minimumSize()
        except Exception as e:
            logging.error(f"Error in sizeHint: {e}")
            return QSize()

    def minimumSize(self):
        try:
            size = QSize()
            for item in self.item_list:
                size = size.expandedTo(item.minimumSize())
            return size
        except Exception as e:
            logging.error(f"Error in minimumSize: {e}")
            return QSize()

    def do_layout(self, rect, test_only):
        try:
            x = rect.x()
            y = rect.y()
            line_height = 0
            for item in self.item_list:
                wid = item.widget()
                space_x = self.spacing() if self.spacing() != -1 else wid.style().layoutSpacing(QSizePolicy.Policy.PushButton, QSizePolicy.Policy.PushButton, Qt.Orientation.Horizontal)
                space_y = self.spacing() if self.spacing() != -1 else wid.style().layoutSpacing(QSizePolicy.Policy.PushButton, QSizePolicy.Policy.PushButton, Qt.Orientation.Vertical)
                next_x = x + item.sizeHint().width() + space_x
                if next_x - space_x > rect.right() and line_height > 0:
                    x = rect.x()
                    y = y + line_height + space_y
                    next_x = x + item.sizeHint().width() + space_x
                    line_height = 0
                if not test_only:
                    item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
                x = next_x
                line_height = max(line_height, item.sizeHint().height())
            return y + line_height - rect.y()
        except Exception as e:
            logging.error(f"Error in do_layout: {e}")
            return 0

# Hamming distance
def hamming_distance(x, ref):
    """Calculate the Hamming distance between two binary strings."""
    return np.sum(np.array(list(x)) != np.array(list(ref)))

# Transition count (for 1D string from 2D grid)
def transition_count(x):
    """Count the number of bit transitions in a binary string."""
    return np.sum([1 for i in range(len(x)-1) if x[i] != x[i+1]])

# Combined complexity measure
def complexity_ht(x, alpha=0.5):
    """Compute the combined complexity using Hamming distance and transitions."""
    n = len(x)
    s0 = "0" * n
    s1 = "1" * n
    h0 = hamming_distance(x, s0) / n
    h1 = hamming_distance(x, s1) / n
    t = transition_count(x) / (n - 1)
    c0 = alpha * h0 + (1 - alpha) * t
    c1 = alpha * h1 + (1 - alpha) * t
    return min(c0, c1)

# Conway's Game of Life update
def game_of_life(grid):
    """Update the 2D grid according to Conway's Game of Life rules."""
    n, m = grid.shape
    new_grid = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            # Count live neighbors
            neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = (i + di) % n, (j + dj) % m
                    neighbors += grid[ni, nj]
            # Apply rules
            if grid[i, j] == 1:  # Live cell
                new_grid[i, j] = 1 if neighbors in [2, 3] else 0
            else:  # Dead cell
                new_grid[i, j] = 1 if neighbors == 3 else 0
    return new_grid

# Evolve Game of Life
def evolve_game_of_life(initial_grid, steps):
    """Evolve the Game of Life grid over the specified number of steps, storing states for visualization."""
    n, m = initial_grid.shape
    states = [initial_grid.copy()]  # Store NumPy arrays for visualization
    current = initial_grid.copy()
    for _ in range(steps):
        current = game_of_life(current)
        states.append(current.copy())  # Store each grid state
    return states

# Function to update the animation frame
def update(frame, states, ax, im, is_perturbed=False):
    """Update the heatmap for each frame in the animation."""
    ax.clear()
    sns.heatmap(states[frame], ax=ax, cmap='binary', cbar=False, xticklabels=False, yticklabels=False)
    ax.set_title(f"Conway's Game of Life - {'Perturbed' if is_perturbed else 'Original'} Step {frame}")
    return im,

# Save run to JSON (convert NumPy arrays to lists for JSON compatibility)
def save_run(states, complexities, filename):
    data = {
        'states': [state.tolist() for state in states],
        'complexities': complexities,
        'n': states[0].shape[0],
        'm': states[0].shape[1]
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

# Load run from JSON (convert lists back to NumPy arrays)
def load_run(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    states = [np.array(state) for state in data['states']]
    complexities = data['complexities']
    n, m = data['n'], data['m']
    return states, complexities, n, m

# Function to copy a matplotlib figure to clipboard
def copy_to_clipboard(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = QImage()
    img.loadFromData(buf.getvalue())
    pixmap = QPixmap.fromImage(img)
    QApplication.clipboard().setPixmap(pixmap)
    buf.close()
    print("Graph copied to clipboard!")

# Function to copy text to clipboard
def copy_text_to_clipboard(text):
    QApplication.clipboard().setText(text)
    print("Text copied to clipboard!")

class GOLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conway's Game of Life Entropy Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Tab widget for plots and visualization
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Tab 1: Original C_HT Plot
        self.tab1 = QWidget()
        self.tab_widget.addTab(self.tab1, "Original C_HT Plot")
        self.fig1, self.ax1 = plt.subplots(figsize=(8, 5))
        self.canvas1 = FigureCanvasQTAgg(self.fig1)
        layout1 = QVBoxLayout(self.tab1)
        layout1.addWidget(self.canvas1)
        self.copy_button1 = QPushButton("Copy to Clipboard", clicked=lambda: copy_to_clipboard(self.fig1))
        layout1.addWidget(self.copy_button1)

        # Tab 2: Perturbed C_HT Plot
        self.tab2 = QWidget()
        self.tab_widget.addTab(self.tab2, "Perturbed C_HT Plot")
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 5))
        self.canvas2 = FigureCanvasQTAgg(self.fig2)
        layout2 = QVBoxLayout(self.tab2)
        layout2.addWidget(self.canvas2)
        self.copy_button2 = QPushButton("Copy to Clipboard", clicked=lambda: copy_to_clipboard(self.fig2))
        layout2.addWidget(self.copy_button2)

        # Tab 3: Grid Visualization
        self.tab3 = QWidget()
        self.tab_widget.addTab(self.tab3, "Grid Visualization")
        self.fig3, self.ax3 = plt.subplots(figsize=(8, 8))
        self.canvas3 = FigureCanvasQTAgg(self.fig3)
        layout3 = QVBoxLayout(self.tab3)
        layout3.addWidget(self.canvas3)
        self.ani = None
        self.is_perturbed = False

        # Tab 4: Analysis Results
        self.tab4 = QWidget()
        self.tab_widget.addTab(self.tab4, "Analysis Results")
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        layout4 = QVBoxLayout(self.tab4)
        layout4.addWidget(self.analysis_text)
        self.copy_button3 = QPushButton("Copy to Clipboard", clicked=lambda: copy_text_to_clipboard(self.analysis_text.toPlainText()))
        layout4.addWidget(self.copy_button3)

        # Control panel with FlowLayout
        control_widget = QWidget()
        self.flow_layout = FlowLayout(control_widget)
        control_widget.setLayout(self.flow_layout)

        # Parameters
        self.n = 30  # Default grid width
        self.m = 30  # Default grid height
        self.steps = 1000  # Default steps
        self.alive_prob = 0.5  # Default probability of alive cells
        self.data_runs = []  # Store C_HT data for multiple runs
        self.original_states = None  # Initialize instance variables
        self.perturbed_states = None

        # Input fields
        self.flow_layout.addWidget(QLabel("Grid Width:"))
        self.n_input = QLineEdit(str(self.n))
        self.n_input.setFixedWidth(50)
        self.flow_layout.addWidget(self.n_input)

        self.flow_layout.addWidget(QLabel("Grid Height:"))
        self.m_input = QLineEdit(str(self.m))
        self.m_input.setFixedWidth(50)
        self.flow_layout.addWidget(self.m_input)

        self.flow_layout.addWidget(QLabel("Steps:"))
        self.steps_input = QLineEdit(str(self.steps))
        self.steps_input.setFixedWidth(50)
        self.flow_layout.addWidget(self.steps_input)

        self.flow_layout.addWidget(QLabel("Alive Prob:"))
        self.prob_input = QLineEdit(str(self.alive_prob))
        self.prob_input.setFixedWidth(50)
        self.flow_layout.addWidget(self.prob_input)

        # Buttons
        self.flow_layout.addWidget(QPushButton("Run Simulation", clicked=self.run_simulation))
        self.flow_layout.addWidget(QPushButton("New Run", clicked=self.new_run))
        self.flow_layout.addWidget(QPushButton("Restart", clicked=self.restart))
        self.flow_layout.addWidget(QPushButton("Toggle Visualization", clicked=self.toggle_visualization))
        self.flow_layout.addWidget(QPushButton("Collect Data (10 Runs)", clicked=self.collect_data))
        self.flow_layout.addWidget(QPushButton("Plot Aggregate Data", clicked=self.plot_aggregate))
        self.flow_layout.addWidget(QPushButton("Save Run", clicked=self.save_run))
        self.flow_layout.addWidget(QPushButton("Load Run", clicked=self.load_run))
        self.flow_layout.addWidget(QPushButton("Export Data as CSV", clicked=self.export_data))

        # Checkbox for perturbation
        self.perturb_check = QCheckBox("Apply Perturbation")
        self.perturb_check.setChecked(True)
        self.flow_layout.addWidget(self.perturb_check)

        layout.addWidget(control_widget)

        # Ensure clean exit on interruption
        self.root = self  # For cleanup
        self.root.destroyed.connect(self.cleanup)

    def cleanup(self):
        """Clean up animations and plots on window close or interruption."""
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
        plt.close('all')

    def run_simulation(self):
        try:
            # Get parameters from inputs
            n = int(self.n_input.text())
            m = int(self.m_input.text())
            steps = int(self.steps_input.text())
            alive_prob = float(self.prob_input.text())

            # Random initial state
            initial = np.random.choice([0, 1], size=(n, m), p=[1 - alive_prob, alive_prob])

            # Original evolution
            original_states = evolve_game_of_life(initial, steps)
            original_complexities = [complexity_ht(''.join(state.flatten().astype(str))) for state in original_states]

            # Perturbed evolution (if checked)
            perturbed_complexities = []
            if self.perturb_check.isChecked():
                perturbed_grid = original_states[-1].copy()
                for _ in range(5):  # Flip 5 random cells
                    i, j = np.random.randint(0, n), np.random.randint(0, m)
                    perturbed_grid[i, j] = 1 - perturbed_grid[i, j]
                perturbed_states = evolve_game_of_life(perturbed_grid, steps)
                perturbed_complexities = [complexity_ht(''.join(state.flatten().astype(str))) for state in perturbed_states]
            else:
                perturbed_complexities = original_complexities  # No perturbation
                perturbed_states = original_states  # No perturbation states

            # Store as instance attributes
            self.original_states = original_states
            self.perturbed_states = perturbed_states if self.perturb_check.isChecked() else original_states

            # Plot original C_HT
            self.ax1.clear()
            self.ax1.plot(original_complexities, label="Original C_HT", color='blue')
            self.ax1.set_xlabel("Time Step")
            self.ax1.set_ylabel("Complexity (C_HT)")
            self.ax1.set_title(f"Complexity Evolution in Conway's Game of Life (Original, {steps} Steps)")
            self.ax1.legend()
            self.ax1.grid(True)
            self.canvas1.draw()

            # Plot perturbed C_HT
            self.ax2.clear()
            self.ax2.plot(original_complexities, label="Original C_HT", color='blue')
            self.ax2.plot(perturbed_complexities, label="Perturbed C_HT" if self.perturb_check.isChecked() else "No Perturbation", color='red', linestyle='--')
            self.ax2.set_xlabel("Time Step")
            self.ax2.set_ylabel("Complexity (C_HT)")
            self.ax2.set_title(f"Complexity Evolution: Original vs. Perturbed ({steps} Steps)")
            self.ax2.legend()
            self.ax2.grid(True)
            self.canvas2.draw()

            # Direct analysis of C_HT and grid states
            # C_HT analysis
            mean_ch = np.mean(original_complexities)
            std_ch = np.std(original_complexities)
            trend = stats.linregress(np.arange(len(original_complexities)), original_complexities).slope
            stability = np.var(original_complexities[-100:]) if len(original_complexities) >= 100 else np.var(original_complexities)

            # Enhanced stability check: consider final 100 steps and pattern
            final_stability = stability < 0.005  # Tighter threshold for stability
            final_state = original_states[-1]
            live_cells = np.sum(final_state)
            is_empty = live_cells == 0
            is_sparse = live_cells < (n * m * 0.05)  # Less than 5% live cells

            # Prepare analysis text with refined interpretation
            analysis_text = (
                f"**C_HT Analysis (Original Run)**\n"
                f"Mean Complexity: {mean_ch:.3f}\n"
                f"Standard Deviation: {std_ch:.3f}\n"
                f"Trend (Slope): {trend:.3f} (Positive = Increasing, Negative = Decreasing)\n"
                f"Stability (Variance): {stability:.3f} (Lower = More Stable)\n"
                f"Final Stability: {'Stable' if final_stability else 'Unstable'}\n\n"
                f"**Grid State Analysis**\n"
                f"Final Live Cells: {live_cells}\n"
                f"Is Empty: {is_empty}\n"
                f"Is Sparse (<5% live): {is_sparse}\n"
                f"Pattern Interpretation: {'Empty grid' if is_empty else 'Sparse pattern (likely stable)' if is_sparse else 'Complex pattern'}\n"
                f"Overall Behavior: {'Stable low-complexity attractors' if final_stability or is_empty or is_sparse else 'Chaotic or unstable behavior'}"
            )
            self.analysis_text.setText(analysis_text)

            # Start visualization if not already running
            self.toggle_visualization()

        except ValueError as e:
            print(f"Error: Invalid input values - {e}")
        except Exception as e:
            logging.error(f"Error in run_simulation: {e}")
            print(f"Error: {e}")

    def new_run(self):
        """Generate a new random starting grid and run a simulation."""
        try:
            # Reset states
            self.original_states = None
            self.perturbed_states = None
            self.run_simulation()
        except Exception as e:
            logging.error(f"Error in new_run: {e}")
            print(f"Error in new run: {e}")

    def restart(self):
        """Reset all states and clear visualizations."""
        self.original_states = None
        self.perturbed_states = None
        self.data_runs = []
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.analysis_text.clear()
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
        print("Simulation restarted.")

    def toggle_visualization(self):
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
            self.ax3.clear()
            self.canvas3.draw()
            return

        if not self.original_states:
            print("Run a simulation first.")
            return

        states = self.perturbed_states if self.is_perturbed else self.original_states
        self.is_perturbed = not self.is_perturbed
        self.ax3.clear()
        im = self.ax3.imshow(states[0], cmap='binary')
        self.ani = FuncAnimation(self.fig3, update, frames=len(states), fargs=(states, self.ax3, im, self.is_perturbed), interval=50, blit=False)
        self.canvas3.draw()

    def collect_data(self):
        try:
            n = int(self.n_input.text())
            m = int(self.m_input.text())
            steps = int(self.steps_input.text())
            alive_prob = float(self.prob_input.text())
            num_runs = 10  # Default to 10 runs for data collection

            all_original_complexities = []
            all_perturbed_complexities = []

            for _ in range(num_runs):
                # Random initial state
                initial = np.random.choice([0, 1], size=(n, m), p=[1 - alive_prob, alive_prob])

                # Original evolution
                original_states = evolve_game_of_life(initial, steps)
                original_complexities = [complexity_ht(''.join(state.flatten().astype(str))) for state in original_states]

                # Perturbed evolution
                perturbed_grid = original_states[-1].copy()
                for _ in range(5):
                    i, j = np.random.randint(0, n), np.random.randint(0, m)
                    perturbed_grid[i, j] = 1 - perturbed_grid[i, j]
                perturbed_states = evolve_game_of_life(perturbed_grid, steps)
                perturbed_complexities = [complexity_ht(''.join(state.flatten().astype(str))) for state in perturbed_states]

                all_original_complexities.append(original_complexities)
                all_perturbed_complexities.append(perturbed_complexities)

            self.data_runs = {
                'original': np.array(all_original_complexities),
                'perturbed': np.array(all_perturbed_complexities)
            }
            print(f"Collected data from {num_runs} runs.")

            # Update analysis with aggregate data
            mean_original = np.mean(self.data_runs['original'], axis=0)
            std_original = np.std(self.data_runs['original'], axis=0)
            mean_perturbed = np.mean(self.data_runs['perturbed'], axis=0)
            std_perturbed = np.std(self.data_runs['perturbed'], axis=0)

            final_stability_original = np.var(mean_original[-100:]) < 0.005 if len(mean_original) >= 100 else np.var(mean_original) < 0.005
            final_stability_perturbed = np.var(mean_perturbed[-100:]) < 0.005 if len(mean_perturbed) >= 100 else np.var(mean_perturbed) < 0.005

            analysis_text = (
                f"**Aggregate C_HT Analysis (10 Runs)**\n"
                f"Original - Mean Complexity: {np.mean(mean_original):.3f}, Std Dev: {np.mean(std_original):.3f}\n"
                f"Perturbed - Mean Complexity: {np.mean(mean_perturbed):.3f}, Std Dev: {np.mean(std_perturbed):.3f}\n"
                f"Original Final Stability: {'Stable' if final_stability_original else 'Unstable'}\n"
                f"Perturbed Final Stability: {'Stable' if final_stability_perturbed else 'Unstable'}\n"
                f"Interpretation: {'Stable low-complexity attractors' if final_stability_original and final_stability_perturbed else 'Chaotic or unstable behavior'}"
            )
            self.analysis_text.setText(analysis_text)

        except ValueError as e:
            print(f"Error: Invalid input values - {e}")
        except Exception as e:
            logging.error(f"Error in collect_data: {e}")
            print(f"Error: {e}")

    def plot_aggregate(self):
        if not self.data_runs:
            print("No data collected. Run 'Collect Data' first.")
            return

        plt.figure(figsize=(12, 6))
        steps = len(self.data_runs['original'][0])
        time_steps = np.arange(steps)

        # Plot average and standard deviation for original
        mean_original = np.mean(self.data_runs['original'], axis=0)
        std_original = np.std(self.data_runs['original'], axis=0)
        plt.plot(time_steps, mean_original, label="Mean Original C_HT", color='blue')
        plt.fill_between(time_steps, mean_original - std_original, mean_original + std_original, alpha=0.2, color='blue')

        # Plot average and standard deviation for perturbed
        mean_perturbed = np.mean(self.data_runs['perturbed'], axis=0)
        std_perturbed = np.std(self.data_runs['perturbed'], axis=0)
        plt.plot(time_steps, mean_perturbed, label="Mean Perturbed C_HT", color='red', linestyle='--')
        plt.fill_between(time_steps, mean_perturbed - std_perturbed, mean_perturbed + std_perturbed, alpha=0.2, color='red')

        plt.xlabel("Time Step")
        plt.ylabel("Complexity (C_HT)")
        plt.title("Aggregate Complexity Evolution (10 Runs)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_run(self):
        if not self.original_states:
            print("Run a simulation first.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Run", "", "JSON Files (*.json)")
        if filename:
            save_run(self.original_states, self.original_complexities, filename)
            if self.perturb_check.isChecked() and self.perturbed_states is not None:
                perturbed_filename = filename.replace('.json', '_perturbed.json')
                save_run(self.perturbed_states, self.perturbed_complexities, perturbed_filename)
            print(f"Run saved to {filename}")

    def load_run(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Run", "", "JSON Files (*.json)")
        if filename:
            original_states, original_complexities, n, m = load_run(filename)
            self.n, self.m = n, m
            self.n_input.setText(str(n))
            self.m_input.setText(str(m))
            self.original_states = original_states
            self.original_complexities = original_complexities

            # Plot original C_HT
            self.ax1.clear()
            self.ax1.plot(original_complexities, label="Original C_HT", color='blue')
            self.ax1.set_xlabel("Time Step")
            self.ax1.set_ylabel("Complexity (C_HT)")
            self.ax1.set_title(f"Complexity Evolution in Conway's Game of Life (Original, {len(original_complexities)} Steps)")
            self.ax1.legend()
            self.ax1.grid(True)
            self.canvas1.draw()

            # Check for perturbed run
            perturbed_filename = filename.replace('.json', '_perturbed.json')
            if os.path.exists(perturbed_filename):
                perturbed_states, perturbed_complexities, _, _ = load_run(perturbed_filename)
                self.perturbed_states = perturbed_states
                self.perturbed_complexities = perturbed_complexities

                # Plot perturbed C_HT
                self.ax2.clear()
                self.ax2.plot(original_complexities, label="Original C_HT", color='blue')
                self.ax2.plot(perturbed_complexities, label="Perturbed C_HT", color='red', linestyle='--')
                self.ax2.set_xlabel("Time Step")
                self.ax2.set_ylabel("Complexity (C_HT)")
                self.ax2.set_title(f"Complexity Evolution: Original vs. Perturbed ({len(perturbed_complexities)} Steps)")
                self.ax2.legend()
                self.ax2.grid(True)
                self.canvas2.draw()
            else:
                self.ax2.clear()
                self.ax2.plot(original_complexities, label="Original C_HT", color='blue')
                self.ax2.set_xlabel("Time Step")
                self.ax2.set_ylabel("Complexity (C_HT)")
                self.ax2.set_title(f"Complexity Evolution: Original (No Perturbation, {len(original_complexities)} Steps)")
                self.ax2.legend()
                self.ax2.grid(True)
                self.canvas2.draw()

            # Update analysis for loaded run
            mean_ch = np.mean(original_complexities)
            std_ch = np.std(original_complexities)
            trend = stats.linregress(np.arange(len(original_complexities)), original_complexities).slope
            stability = np.var(original_complexities[-100:]) if len(original_complexities) >= 100 else np.var(original_complexities)

            final_state = original_states[-1]
            live_cells = np.sum(final_state)
            is_empty = live_cells == 0
            is_sparse = live_cells < (n * m * 0.05)

            analysis_text = (
                f"**C_HT Analysis (Loaded Original Run)**\n"
                f"Mean Complexity: {mean_ch:.3f}\n"
                f"Standard Deviation: {std_ch:.3f}\n"
                f"Trend (Slope): {trend:.3f} (Positive = Increasing, Negative = Decreasing)\n"
                f"Stability (Variance): {stability:.3f} (Lower = More Stable)\n"
                f"Final Stability: {'Stable' if stability < 0.005 else 'Unstable'}\n\n"
                f"**Grid State Analysis**\n"
                f"Final Live Cells: {live_cells}\n"
                f"Is Empty: {is_empty}\n"
                f"Is Sparse (<5% live): {is_sparse}\n"
                f"Pattern Interpretation: {'Empty grid' if is_empty else 'Sparse pattern (likely stable)' if is_sparse else 'Complex pattern'}\n"
                f"Overall Behavior: {'Stable low-complexity attractors' if stability < 0.005 or is_empty or is_sparse else 'Chaotic or unstable behavior'}"
            )
            self.analysis_text.setText(analysis_text)

            self.toggle_visualization()  # Start with original animation
            print(f"Run loaded from {filename}")

    def export_data(self):
        if not self.data_runs and not self.original_complexities:
            print("No data to export. Run a simulation or collect data first.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Export Data as CSV", "", "CSV Files (*.csv)")
        if filename:
            if self.data_runs:
                # Export aggregate data
                df = pd.DataFrame({
                    'Time Step': np.arange(len(self.data_runs['original'][0])),
                    'Mean Original C_HT': np.mean(self.data_runs['original'], axis=0),
                    'Std Original C_HT': np.std(self.data_runs['original'], axis=0),
                    'Mean Perturbed C_HT': np.mean(self.data_runs['perturbed'], axis=0),
                    'Std Perturbed C_HT': np.std(self.data_runs['perturbed'], axis=0)
                })
                df.to_csv(filename, index=False)
                print(f"Aggregate data exported to {filename}")
            else:
                # Export single run data
                df = pd.DataFrame({
                    'Time Step': np.arange(len(self.original_complexities)),
                    'Original C_HT': self.original_complexities,
                    'Perturbed C_HT': self.perturbed_complexities if self.perturb_check.isChecked() and self.perturbed_complexities else self.original_complexities
                })
                df.to_csv(filename, index=False)
                print(f"Single run data exported to {filename}")

if __name__ == '__main__':
    try:
        app = QApplication([])
        window = GOLApp()
        window.show()
        app.exec()
    except KeyboardInterrupt:
        print("Application interrupted. Cleaning up...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"An error occurred: {e}")
        sys.exit(1)
