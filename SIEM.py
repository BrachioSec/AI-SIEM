import os
import json
import argparse
import logging
import joblib
import numpy as np
# Set Agg backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from urllib.parse import urlparse
import csv
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import queue

VERSION = "2.0.1"
DATA_DIR = Path("./siem_data")
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
DEFAULT_MODEL = MODELS_DIR / "siem_model.joblib"

# Create directories if needed
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "siem_tool.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIEM-Tool")

class BERTEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              max_length=256, padding="max_length").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    @torch.no_grad()
    def embed_batch(self, texts):
        return np.vstack([self.embed(t) for t in texts])

class PhishingDetector:
    def __init__(self):
        self.cache_file = CACHE_DIR / "url_cache.csv"
        self.cache = self.load_cache()

    def load_cache(self):
        cache = {}
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                for row in csv.reader(f):
                    if len(row) == 2:
                        cache[row[0]] = row[1] == "True"
        return cache

    def save_cache(self):
        with open(self.cache_file, "w") as f:
            writer = csv.writer(f)
            for url, flagged in self.cache.items():
                writer.writerow([url, flagged])

    def is_phishing(self, url: str) -> bool:
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if normalized in self.cache:
            return self.cache[normalized]
        # In a real implementation, you would call an API here
        flagged = "phish" in normalized or "malicious" in normalized
        self.cache[normalized] = flagged
        self.save_cache()
        return flagged

    def extract_urls(self, text: str):
        return [w for w in text.split() if w.startswith(("http://", "https://"))]

    def detect_in_text(self, text: str):
        urls = self.extract_urls(text)
        results = []
        for u in urls:
            results.append({"url": u, "is_phishing": self.is_phishing(u)})
        return {
            "text": text, 
            "urls": results, 
            "has_phishing": any(r["is_phishing"] for r in results)
        }

class SIEMDetector:
    def __init__(self, model_path=None):
        self.embedder = BERTEmbedder()
        self.supervised_model = None
        self.unsupervised_model = None
        self.phishing_detector = PhishingDetector()
        self.model_path = model_path or DEFAULT_MODEL

    def train(self, jsonl_file: Path, progress_callback=None):
        if progress_callback:
            progress_callback(10, "Loading training data...")
        
        # Load training data
        logs, labels = [], []
        try:
            with open(jsonl_file, "r") as f:
                total_lines = sum(1 for _ in f)
                f.seek(0)
                
                for i, line in enumerate(f):
                    entry = json.loads(line)
                    logs.append(entry["log"])
                    labels.append(entry["label"])
                    
                    if progress_callback and i % 100 == 0:
                        progress = 10 + int(70 * i / total_lines)
                        progress_callback(progress, f"Loading data: {i}/{total_lines}")
                        
            if progress_callback:
                progress_callback(80, "Generating embeddings...")
        except Exception as e:
            return False, f"Error loading training data: {e}"

        # Generate embeddings
        try:
            embeddings = self.embedder.embed_batch(logs)
        except Exception as e:
            return False, f"Error generating embeddings: {e}"

        # Train supervised model
        if progress_callback:
            progress_callback(85, "Training classifier...")
        try:
            self.supervised_model = RandomForestClassifier(
                n_estimators=150, 
                random_state=42, 
                class_weight="balanced", 
                n_jobs=-1
            )
            self.supervised_model.fit(embeddings, labels)
        except Exception as e:
            return False, f"Error training classifier: {e}"

        # Train unsupervised model
        if progress_callback:
            progress_callback(90, "Training anomaly detector...")
        try:
            self.unsupervised_model = IsolationForest(
                n_estimators=100, 
                contamination=0.05, 
                random_state=42, 
                n_jobs=-1
            )
            self.unsupervised_model.fit(embeddings)
        except Exception as e:
            return False, f"Error training anomaly detector: {e}"

        # Save model
        if progress_callback:
            progress_callback(95, "Saving model...")
        try:
            model_dict = {
                "supervised": self.supervised_model, 
                "unsupervised": self.unsupervised_model
            }
            joblib.dump(model_dict, self.model_path)
            return True, f"Training complete! Model saved to {self.model_path}"
        except Exception as e:
            return False, f"Error saving model: {e}"

    def load_model(self):
        if not self.model_path.exists():
            return False, f"Model not found: {self.model_path}"
            
        try:
            models = joblib.load(self.model_path)
            self.supervised_model = models["supervised"]
            self.unsupervised_model = models["unsupervised"]
            return True, f"Loaded model: {self.model_path.name}"
        except Exception as e:
            return False, f"Error loading model: {e}"

    def analyze_log(self, log_entry):
        try:
            log_text = log_entry.get("message", log_entry.get("log", str(log_entry)))
            embedding = self.embedder.embed(log_text)

            supervised_pred = int(self.supervised_model.predict([embedding])[0])
            supervised_prob = float(self.supervised_model.predict_proba([embedding])[0][1])

            anomaly_score = float(self.unsupervised_model.decision_function([embedding])[0])
            is_anomalous = anomaly_score < 0

            phishing_result = self.phishing_detector.detect_in_text(log_text)

            return {
                "log_entry": log_entry,
                "log_text": log_text,
                "supervised_prediction": supervised_pred,
                "supervised_probability": supervised_prob,
                "anomaly_score": anomaly_score,
                "is_anomalous": is_anomalous,
                "phishing_detection": phishing_result,
                "is_threat": supervised_pred == 1 or is_anomalous or phishing_result["has_phishing"]
            }
        except Exception as e:
            logger.error(f"Error analyzing log: {e}")
            return None

    def detect(self, json_file: Path, progress_callback=None):
        success, msg = self.load_model()
        if not success:
            return False, msg
            
        # Load log file
        try:
            with open(json_file, "r") as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = [logs]
        except Exception as e:
            return False, f"Error loading log file: {e}"

        # Process logs
        results = []
        threat_count = 0
        phishing_count = 0
        
        for i, entry in enumerate(logs):
            if progress_callback:
                progress = int(100 * i / len(logs))
                progress_callback(progress, f"Analyzing log {i+1}/{len(logs)}")
                
            result = self.analyze_log(entry)
            if result is None:
                continue
                
            results.append(result)
            if result["is_threat"]:
                threat_count += 1
            if result["phishing_detection"]["has_phishing"]:
                phishing_count += 1

        # Generate report
        report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_file = REPORTS_DIR / f"{report_name}.json"
        try:
            with open(report_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            return False, f"Error saving report: {e}"
            
        # Return data for visualization instead of generating it here
        return True, {
            "summary": self.get_summary(results, json_file.name),
            "results": results,
            "report_file": report_file,
            "report_name": report_name
        }
    
    def get_summary(self, results, filename):
        threat_count = sum(1 for r in results if r["is_threat"])
        phishing_count = sum(1 for r in results if r["phishing_detection"]["has_phishing"])
        
        summary = f"Analysis of {filename} complete!\n\n"
        summary += f"Model used: {self.model_path.name}\n"
        summary += f"Total logs processed: {len(results)}\n"
        summary += f"Potential threats detected: {threat_count}\n"
        summary += f"Phishing URLs found: {phishing_count}\n"
        
        if results:
            anomaly_rate = sum(1 for r in results if r['is_anomalous']) / len(results)
            summary += f"Anomaly rate: {anomaly_rate:.2%}\n"
        
        if threat_count > 0:
            summary += "\nTop threats:\n"
            threats = sorted(
                [r for r in results if r['is_threat']], 
                key=lambda x: x['supervised_probability'], 
                reverse=True
            )[:3]
            
            for i, threat in enumerate(threats, 1):
                summary += f"\n{i}. [Risk: {threat['supervised_probability']:.2%}]\n"
                summary += f"   {threat['log_text'][:100]}{'...' if len(threat['log_text']) > 100 else ''}\n"
                if threat['phishing_detection']['has_phishing']:
                    summary += "   🚩 Contains phishing URL\n"
        
        summary += f"\nReport saved to: {REPORTS_DIR}/"
        return summary

    def generate_visualization(self, results, report_name):
        if not results:
            return None
            
        normal = sum(1 for r in results if not r["is_threat"])
        anomaly = sum(1 for r in results if r["is_anomalous"])
        phishing = sum(1 for r in results if r["phishing_detection"]["has_phishing"])
        supervised = sum(1 for r in results if r["supervised_prediction"] == 1)

        # Create plot safely in main thread
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ["Normal", "Anomalies", "Phishing", "Known Threats"]
        counts = [normal, anomaly, phishing, supervised]
        colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
        bars = ax.bar(categories, counts, color=colors)
        ax.set_title("Security Event Distribution")
        ax.set_ylabel("Count")

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                height + 0.5, 
                str(int(height)), 
                ha='center'
            )

        # Add watermark
        fig.text(0.5, 0.01, f"Generated by SIEM Tool v{VERSION}", ha="center", fontsize=10, alpha=0.7)

        plot_file = REPORTS_DIR / f"{report_name}_plot.png"
        try:
            fig.savefig(plot_file, bbox_inches="tight")
            plt.close(fig)  # Important: close the figure to free memory
            return plot_file
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            return None

class SIEMToolGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"SIEM Security Analyzer v{VERSION}")
        self.geometry("900x700")
        self.configure(bg="#f0f0f0")
        self.detector = SIEMDetector()
        self.current_model = DEFAULT_MODEL
        self.create_widgets()
        self.plot_queue = queue.Queue()
        
        # Center the window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')
        
        # Start periodic check for plot generation
        self.after(100, self.process_plot_queue)

    def create_widgets(self):
        # Create style
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TButton", font=("Arial", 10), padding=6)
        style.configure("Header.TLabel", font=("Arial", 16, "bold"), background="#f0f0f0")
        style.configure("Title.TLabel", font=("Arial", 24, "bold"), background="#3f51b5", foreground="white")
        
        # Header frame
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(header_frame, text="SIEM SECURITY ANALYZER", style="Title.TLabel")
        title_label.pack(fill=tk.X, padx=10, pady=10)
        
        # Version and author
        info_label = ttk.Label(header_frame, text=f"Version {VERSION} | Made by BRACİHOSEC", 
                              font=("Arial", 9), background="#f0f0f0")
        info_label.pack(pady=(0, 10))
        
        # Main content frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Actions
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        action_frame = ttk.LabelFrame(left_frame, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Action buttons
        ttk.Button(action_frame, text="Train New Model", command=self.train_model, 
                  style="Accent.TButton").pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="Detect Threats", command=self.detect_threats, 
                  style="Accent.TButton").pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="Select Model", command=self.select_model).pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="View Reports", command=self.view_reports).pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="Exit", command=self.destroy).pack(fill=tk.X, pady=5)
        
        # Model info
        model_frame = ttk.LabelFrame(left_frame, text="Current Model", padding=10)
        model_frame.pack(fill=tk.X)
        
        self.model_label = ttk.Label(model_frame, text="No model loaded", wraplength=200)
        self.model_label.pack(fill=tk.X)
        self.update_model_label()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Right panel - Console
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding=10)
        console_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=20)
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.configure(state=tk.DISABLED, font=("Consolas", 10))
        
        # Configure custom styles
        style.configure("Accent.TButton", background="#4CAF50", foreground="white")
        style.map("Accent.TButton", background=[("active", "#45a049")])
        
    def update_model_label(self):
        if self.current_model.exists():
            self.model_label.config(text=self.current_model.name)
        else:
            self.model_label.config(text="No model loaded")
        
    def log_message(self, message):
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)
        
    def set_status(self, message):
        self.status_var.set(message)
        
    def train_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Training Data",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        # Create progress dialog
        progress_dialog = tk.Toplevel(self)
        progress_dialog.title("Training Model")
        progress_dialog.geometry("400x150")
        progress_dialog.transient(self)
        progress_dialog.grab_set()
        
        # Center the dialog
        progress_dialog.update_idletasks()
        width = progress_dialog.winfo_width()
        height = progress_dialog.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        progress_dialog.geometry(f'+{x}+{y}')
        
        ttk.Label(progress_dialog, text="Training model...").pack(pady=10)
        
        progress_var = tk.IntVar()
        progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=5)
        
        status_var = tk.StringVar(value="Starting training...")
        ttk.Label(progress_dialog, textvariable=status_var).pack(pady=5)
        
        def update_progress(progress, message):
            progress_var.set(progress)
            status_var.set(message)
            progress_dialog.update()
            
        def training_thread():
            try:
                self.detector.model_path = self.current_model
                success, message = self.detector.train(Path(file_path), update_progress)
                
                if success:
                    self.log_message(f"Training successful!\n{message}")
                    self.update_model_label()
                    messagebox.showinfo("Training Complete", "Model trained successfully!")
                else:
                    self.log_message(f"Training failed: {message}")
                    messagebox.showerror("Training Error", message)
            except Exception as e:
                self.log_message(f"Unexpected error: {str(e)}")
                messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            finally:
                progress_dialog.destroy()
                
        threading.Thread(target=training_thread, daemon=True).start()
        
    def detect_threats(self):
        file_path = filedialog.askopenfilename(
            title="Select Log File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        # Create progress dialog
        progress_dialog = tk.Toplevel(self)
        progress_dialog.title("Analyzing Logs")
        progress_dialog.geometry("400x150")
        progress_dialog.transient(self)
        progress_dialog.grab_set()
        
        # Center the dialog
        progress_dialog.update_idletasks()
        width = progress_dialog.winfo_width()
        height = progress_dialog.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        progress_dialog.geometry(f'+{x}+{y}')
        
        ttk.Label(progress_dialog, text="Analyzing logs...").pack(pady=10)
        
        progress_var = tk.IntVar()
        progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=5)
        
        status_var = tk.StringVar(value="Starting analysis...")
        ttk.Label(progress_dialog, textvariable=status_var).pack(pady=5)
        
        def update_progress(progress, message):
            progress_var.set(progress)
            status_var.set(message)
            progress_dialog.update()
            
        def detection_thread():
            try:
                self.detector.model_path = self.current_model
                success, result = self.detector.detect(Path(file_path), update_progress)
                
                if success:
                    # Add visualization task to queue to be processed in main thread
                    self.plot_queue.put((result, file_path))
                    self.log_message(f"Detection complete!\n{result['summary']}")
                    messagebox.showinfo("Analysis Complete", "Threat detection completed successfully!")
                else:
                    self.log_message(f"Detection failed: {result}")
                    messagebox.showerror("Detection Error", result)
            except Exception as e:
                self.log_message(f"Unexpected error: {str(e)}")
                messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            finally:
                progress_dialog.destroy()
                
        threading.Thread(target=detection_thread, daemon=True).start()
        
    def process_plot_queue(self):
        """Process visualization tasks from queue in main thread"""
        try:
            while not self.plot_queue.empty():
                result, file_path = self.plot_queue.get_nowait()
                
                # Generate visualization in main thread
                plot_file = self.detector.generate_visualization(
                    result["results"], 
                    result["report_name"]
                )
                
                if plot_file:
                    self.log_message(f"Visualization saved to: {plot_file}")
                else:
                    self.log_message("Failed to generate visualization")
        except queue.Empty:
            pass
        
        # Check again after 100ms
        self.after(100, self.process_plot_queue)
        
    def select_model(self):
        models = list(MODELS_DIR.glob("*.joblib"))
        if not models:
            messagebox.showinfo("No Models", "No trained models found. Please train a model first.")
            return
            
        # Create selection dialog
        model_dialog = tk.Toplevel(self)
        model_dialog.title("Select Model")
        model_dialog.geometry("500x300")
        model_dialog.transient(self)
        model_dialog.grab_set()
        
        # Center the dialog
        model_dialog.update_idletasks()
        width = model_dialog.winfo_width()
        height = model_dialog.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        model_dialog.geometry(f'+{x}+{y}')
        
        ttk.Label(model_dialog, text="Select a model to use for detection:").pack(pady=10)
        
        # Create listbox with models
        list_frame = ttk.Frame(model_dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        model_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial", 11))
        model_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=model_list.yview)
        
        for model in models:
            model_list.insert(tk.END, model.name)
            
        # Set default selection to current model if available
        if self.current_model in models:
            idx = models.index(self.current_model)
            model_list.select_set(idx)
            model_list.see(idx)
        
        # Button frame
        button_frame = ttk.Frame(model_dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def select_and_close():
            selection = model_list.curselection()
            if selection:
                self.current_model = models[selection[0]]
                self.update_model_label()
                self.log_message(f"Selected model: {self.current_model.name}")
                model_dialog.destroy()
        
        ttk.Button(button_frame, text="Select", command=select_and_close).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Cancel", command=model_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
    def view_reports(self):
        reports = list(REPORTS_DIR.glob("report_*.json"))
        if not reports:
            messagebox.showinfo("No Reports", "No analysis reports found.")
            return
            
        # Create report viewer
        report_dialog = tk.Toplevel(self)
        report_dialog.title("View Reports")
        report_dialog.geometry("800x600")
        report_dialog.transient(self)
        
        # Center the dialog
        report_dialog.update_idletasks()
        width = report_dialog.winfo_width()
        height = report_dialog.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        report_dialog.geometry(f'+{x}+{y}')
        
        # Header
        header_frame = ttk.Frame(report_dialog)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header_frame, text="Analysis Reports", font=("Arial", 14, "bold")).pack(side=tk.LEFT)
        
        # Report list
        list_frame = ttk.LabelFrame(report_dialog, text="Available Reports")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        report_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial", 11))
        report_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=report_list.yview)
        
        for report in sorted(reports, reverse=True):
            report_list.insert(tk.END, report.name)
            
        # Report viewer
        view_frame = ttk.LabelFrame(report_dialog, text="Report Content")
        view_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        report_viewer = scrolledtext.ScrolledText(view_frame, wrap=tk.WORD)
        report_viewer.pack(fill=tk.BOTH, expand=True)
        report_viewer.configure(state=tk.DISABLED, font=("Consolas", 10))
        
        def show_report(event=None):
            selection = report_list.curselection()
            if not selection:
                return
                
            report_file = reports[selection[0]]
            try:
                with open(report_file, "r") as f:
                    content = json.load(f)
                
                # Format the report content
                formatted = f"Report: {report_file.name}\n\n"
                formatted += f"Total entries: {len(content)}\n"
                
                threats = sum(1 for r in content if r['is_threat'])
                anomalies = sum(1 for r in content if r['is_anomalous'])
                phishing = sum(1 for r in content if r['phishing_detection']['has_phishing'])
                
                formatted += f"Threats detected: {threats}\n"
                formatted += f"Anomalies detected: {anomalies}\n"
                formatted += f"Phishing URLs found: {phishing}\n\n"
                
                # Show top threats
                if threats > 0:
                    top_threats = sorted(
                        [r for r in content if r['is_threat']], 
                        key=lambda x: x['supervised_probability'], 
                        reverse=True
                    )[:5]
                    
                    formatted += "Top Threats:\n"
                    for i, threat in enumerate(top_threats, 1):
                        formatted += f"\n{i}. [Risk: {threat['supervised_probability']:.2%}]\n"
                        formatted += f"   Log: {threat['log_text'][:150]}{'...' if len(threat['log_text']) > 150 else ''}\n"
                        if threat['phishing_detection']['has_phishing']:
                            formatted += "   🚩 Contains phishing URL\n"
                
                report_viewer.configure(state=tk.NORMAL)
                report_viewer.delete(1.0, tk.END)
                report_viewer.insert(tk.END, formatted)
                report_viewer.configure(state=tk.DISABLED)
            except Exception as e:
                report_viewer.configure(state=tk.NORMAL)
                report_viewer.delete(1.0, tk.END)
                report_viewer.insert(tk.END, f"Error loading report: {str(e)}")
                report_viewer.configure(state=tk.DISABLED)
        
        report_list.bind("<<ListboxSelect>>", show_report)
        
        # Buttons
        button_frame = ttk.Frame(report_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Close", command=report_dialog.destroy).pack(side=tk.RIGHT)
        
        # Select first report by default
        if reports:
            report_list.selection_set(0)
            report_list.event_generate("<<ListboxSelect>>")

def command_line_mode():
    parser = argparse.ArgumentParser(description=f"SIEM Tool v{VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model from JSONL")
    train_parser.add_argument("file", type=Path, help="Training data JSONL file")
    train_parser.add_argument("-o", "--output", type=Path, help="Output model file", default=DEFAULT_MODEL)
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect threats from JSON logs")
    detect_parser.add_argument("file", type=Path, help="Log file JSON")
    detect_parser.add_argument("-m", "--model", type=Path, help="Model file to use", default=DEFAULT_MODEL)
    
    # Info command
    subparsers.add_parser("info", help="Show system info")

    args = parser.parse_args()
    detector = SIEMDetector()

    if args.command == "train":
        detector.model_path = args.output
        success, message = detector.train(args.file)
        print(message)
        if not success:
            sys.exit(1)
    
    elif args.command == "detect":
        detector.model_path = args.model
        success, result = detector.detect(args.file)
        if success:
            print(result["summary"])
            # Generate visualization for CLI
            plot_file = detector.generate_visualization(result["results"], result["report_name"])
            if plot_file:
                print(f"Visualization saved to: {plot_file}")
        else:
            print(result)
            sys.exit(1)
    
    elif args.command == "info":
        print(f"SIEM Tool v{VERSION}")
        print(f"Models directory: {MODELS_DIR}")
        print(f"Reports directory: {REPORTS_DIR}")
        print(f"Logs directory: {LOGS_DIR}")
        print(f"Cache directory: {CACHE_DIR}")
        
        print("\nAvailable models:")
        models = list(MODELS_DIR.glob("*.joblib"))
        for model in models:
            print(f"- {model.name}")
            
        print("\nAvailable reports:")
        reports = list(REPORTS_DIR.glob("report_*.json"))
        for report in reports:
            print(f"- {report.name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command_line_mode()
    else:
        app = SIEMToolGUI()
        app.mainloop()
