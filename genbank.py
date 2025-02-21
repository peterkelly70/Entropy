import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from Bio import Entrez, SeqIO
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to fetch gene sequences from GenBank
def fetch_genbank_sequences(organism, max_records=100):
    Entrez.email = "your_email@example.com"  # Replace with your email
    query = f"{organism}[ORGN] AND gene"
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records)
    record = Entrez.read(handle)
    handle.close()
    
    ids = record.get("IdList", [])
    sequences = []
    
    if not ids:
        return sequences  # Return empty list if no sequences found
    
    for seq_id in ids:
        handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
        try:
            seq_record = SeqIO.read(handle, "fasta-pearson")  # Use fasta-pearson format
            sequences.append(str(seq_record.seq))
        except ValueError:
            continue  # Skip any invalid sequences
        finally:
            handle.close()
    
    return sequences

# Function to encode DNA sequence as binary
def encode_dna_to_binary(sequence):
    mapping = {"A": "00", "T": "01", "C": "10", "G": "11"}
    return "".join([mapping[base] for base in sequence if base in mapping])

# Function to calculate Hamming complexity
def hamming_complexity(sequence):
    binary_seq = encode_dna_to_binary(sequence)
    max_ordered_seq = "0" * len(binary_seq)  # All-zero reference sequence
    hamming_distance = sum(b1 != b2 for b1, b2 in zip(binary_seq, max_ordered_seq))
    return hamming_distance

# Function to copy text to clipboard
def copy_to_clipboard(text, root):
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    messagebox.showinfo("Copied", "Text copied to clipboard!")

# Function to run analysis and display results in separate tabs
def run_analysis(organism, notebook, root):
    sequences = fetch_genbank_sequences(organism, max_records=100)
    
    if not sequences:
        messagebox.showerror("Error", f"No sequences found for {organism}. Try another organism.")
        return
    
    data = {"Sequence": sequences, "Hamming Complexity": []}
    
    for seq in sequences:
        complexity = hamming_complexity(seq)
        data["Hamming Complexity"].append(complexity)
    
    df = pd.DataFrame(data)
    
    # Create separate tabs for histogram, boxplot, and statistics
    hist_tab = ttk.Frame(notebook)
    boxplot_tab = ttk.Frame(notebook)
    stats_tab = ttk.Frame(notebook)
    
    notebook.add(hist_tab, text=f"{organism} Histogram")
    notebook.add(boxplot_tab, text=f"{organism} Boxplot")
    notebook.add(stats_tab, text=f"{organism} Stats")
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["Hamming Complexity"], bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel("Hamming Complexity")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Hamming Complexity in {organism} DNA Sequences")
    
    canvas = FigureCanvasTkAgg(fig, master=hist_tab)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    # Copy button for histogram
    copy_hist_btn = tk.Button(hist_tab, text="Copy Data", command=lambda: copy_to_clipboard(df.to_string(), root))
    copy_hist_btn.pack()
    
    # Create boxplot
    fig_box, ax_box = plt.subplots(figsize=(6, 4))
    ax_box.boxplot(df["Hamming Complexity"], vert=False)
    ax_box.set_xlabel("Hamming Complexity")
    ax_box.set_title("Boxplot of Complexity")
    canvas_box = FigureCanvasTkAgg(fig_box, master=boxplot_tab)
    canvas_box.draw()
    canvas_box.get_tk_widget().pack()
    
    # Copy button for boxplot data
    copy_box_btn = tk.Button(boxplot_tab, text="Copy Data", command=lambda: copy_to_clipboard(df.to_string(), root))
    copy_box_btn.pack()
    
    # Create statistics tab
    summary = df["Hamming Complexity"].describe()
    summary_text = summary.to_string()
    text_widget = tk.Text(stats_tab, height=20, width=80)
    text_widget.pack()
    text_widget.insert(tk.END, "Statistical Summary:\n")
    text_widget.insert(tk.END, summary_text + "\n\n")
    text_widget.insert(tk.END, df.to_string())
    
    # Copy button for statistics
    copy_stats_btn = tk.Button(stats_tab, text="Copy Stats", command=lambda: copy_to_clipboard(summary_text, root))
    copy_stats_btn.pack()

# GUI for organism selection with tabbed results
def main():
    root = tk.Tk()
    root.title("GenBank Hamming Complexity Analyzer")
    
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")
    
    control_frame = ttk.Frame(root)
    control_frame.pack()
    
    tk.Label(control_frame, text="Select Organism:").pack()
    organism_var = tk.StringVar()
    organism_dropdown = ttk.Combobox(control_frame, textvariable=organism_var)
    organism_dropdown['values'] = ("Homo sapiens", "Mus musculus", "Drosophila melanogaster", "Escherichia coli")
    organism_dropdown.pack()
    organism_dropdown.current(0)
    
    def start_analysis():
        organism = organism_var.get()
        run_analysis(organism, notebook, root)
    
    tk.Button(control_frame, text="Analyze", command=start_analysis).pack()
    root.mainloop()

if __name__ == "__main__":
    main()
