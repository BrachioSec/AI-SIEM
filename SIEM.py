import os
import json
import argparse
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from urllib.parse import urlparse
import hashlib
import csv
import sys

VERSION = "1.2.0"
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

def print_banner():
    print(f"""
    ███████╗██╗███████╗███╗   ███╗
    ██╔════╝██║██╔════╝████╗ ████║
    ███████╗██║█████╗  ██╔████╔██║
    ╚════██║██║██╔══╝  ██║╚██╔╝██║
    ███████║██║███████╗██║ ╚═╝ ██║
    ╚══════╝╚═╝╚══════╝╚═╝     ╚═╝
    Made by BRACİHOSEC
    Version {VERSION}
    """)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ========== NEW FILE SELECTION FUNCTIONS ==========
def list_files_in_dir(directory: Path, extension: str):
    """List files in directory with specific extension"""
    return [f for f in directory.iterdir() if f.is_file() and f.suffix == extension]

def choose_file_from_dir(directory: Path, extension: str, prompt: str):
    """Interactive file selection from directory"""
    files = list_files_in_dir(directory, extension)
    if not files:
        print(f"{directory} theres no {extension} file.")
        return None
    print(prompt)
    for i, f in enumerate(files, 1):
        print(f"{i}. {f.name}")
    while True:
        try:
            choice = int(input(f"\nSelect one (1-{len(files)}): "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f"write number{len(files)}.")
        except ValueError:
            print("Write an number accordingly")
# ========== END OF NEW FUNCTIONS ==========

def find_local_files(extensions):
    """Find files with given extensions in current directory"""
    files = []
    for ext in extensions:
        files.extend(list(Path.cwd().glob(f"*.{ext}")))
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

def select_file(prompt, file_types, allow_custom=True):
    """Let user select a file from local directory or enter custom path"""
    # Find matching files
    local_files = find_local_files(file_types)
    
    if local_files:
        print(f"\n{prompt}")
        print("Found these files in current directory:")
        for i, file in enumerate(local_files, 1):
            print(f"{i}. {file.name} (modified: {datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})")
        
        if allow_custom:
            print(f"{len(local_files)+1}. Enter custom path")
        
        choice = input("\nSelect file or enter custom path: ").strip()
        
        try:
            # Try to convert to integer selection
            choice_num = int(choice)
            if 1 <= choice_num <= len(local_files):
                return local_files[choice_num-1]
            if allow_custom and choice_num == len(local_files)+1:
                return get_custom_path(file_types)
        except ValueError:
            # If not a number, treat as custom path
            custom_path = Path(choice)
            if custom_path.exists() and custom_path.suffix[1:] in file_types:
                return custom_path
            print(f"File not found or wrong type: {choice}")
    
    # If no files found or custom path needed
    if allow_custom:
        return get_custom_path(file_types)
    
    print("No matching files found in current directory")
    return None

def get_custom_path(file_types):
    """Get custom file path from user with validation"""
    while True:
        path = input("Enter full path to file: ").strip()
        if not path:
            continue
            
        path = Path(path)
        if not path.exists():
            print(f"File not found: {path}")
            continue
            
        if path.suffix[1:] not in file_types:
            print(f"File must be one of these types: {', '.join(file_types)}")
            continue
            
        return path

def get_user_choice(prompt, options):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        choice = input("\nEnter your choice: ").strip()
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return choice_num
        print(f"Please enter a number between 1 and {len(options)}")

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
    def __init__(self):
        self.embedder = BERTEmbedder()
        self.supervised_model = None
        self.unsupervised_model = None
        self.phishing_detector = PhishingDetector()

    def train(self, jsonl_file: Path):
        print(f"\n🚀 Training model with {jsonl_file.name}")
        
        # Load training data
        logs, labels = [], []
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    logs.append(entry["log"])
                    labels.append(entry["label"])
            print(f"Loaded {len(logs)} training examples")
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return

        # Generate embeddings
        print("🔧 Generating embeddings...")
        try:
            embeddings = self.embedder.embed_batch(logs)
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return

        # Train supervised model
        print("🎓 Training classifier...")
        try:
            self.supervised_model = RandomForestClassifier(
                n_estimators=150, 
                random_state=42, 
                class_weight="balanced", 
                n_jobs=-1
            )
            self.supervised_model.fit(embeddings, labels)
        except Exception as e:
            print(f"❌ Error training classifier: {e}")
            return

        # Train unsupervised model
        print("🔍 Training anomaly detector...")
        try:
            self.unsupervised_model = IsolationForest(
                n_estimators=100, 
                contamination=0.05, 
                random_state=42, 
                n_jobs=-1
            )
            self.unsupervised_model.fit(embeddings)
        except Exception as e:
            print(f"❌ Error training anomaly detector: {e}")
            return

        # Save model
        try:
            joblib.dump({
                "supervised": self.supervised_model, 
                "unsupervised": self.unsupervised_model
            }, DEFAULT_MODEL)
            print(f"✅ Training complete! Model saved to {DEFAULT_MODEL}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def load_model(self):
        if not DEFAULT_MODEL.exists():
            print("❌ No trained model found. Please train a model first.")
            return False
        try:
            models = joblib.load(DEFAULT_MODEL)
            self.supervised_model = models["supervised"]
            self.unsupervised_model = models["unsupervised"]
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

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
            print(f"⚠️ Error analyzing log: {e}")
            return None

    def detect(self, json_file: Path):
        if not self.load_model():
            return
            
        print(f"\n🔍 Analyzing {json_file.name}")

        # Load log file
        try:
            with open(json_file, "r") as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = [logs]
            print(f"Found {len(logs)} log entries to analyze")
        except Exception as e:
            print(f"❌ Error loading log file: {e}")
            return

        # Process logs
        results = []
        threat_count = 0
        phishing_count = 0
        
        print("\nProgress: [", end="", flush=True)
        for i, entry in enumerate(logs):
            result = self.analyze_log(entry)
            if result is None:
                continue
                
            results.append(result)
            if result["is_threat"]:
                threat_count += 1
            if result["phishing_detection"]["has_phishing"]:
                phishing_count += 1
            
            # Update progress every 2%
            if i % max(1, len(logs)//50) == 0:
                print("#", end="", flush=True)
        print("] 100%")

        # Generate report
        report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_file = REPORTS_DIR / f"{report_name}.json"
        try:
            with open(report_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return
            
        # Generate visualization
        plot_file = self.generate_visualization(results, report_name)

        # Show summary
        print(f"\n📊 Analysis Results")
        print("=" * 50)
        print(f"Total logs processed: {len(results)}")
        print(f"Potential threats detected: {threat_count}")
        print(f"Phishing URLs found: {phishing_count}")
        if results:
            anomaly_rate = sum(1 for r in results if r['is_anomalous']) / len(results)
            print(f"Anomaly rate: {anomaly_rate:.2%}")
        else:
            print("No results to analyze")
            return

        if threat_count > 0:
            print("\n🔔 Top threats:")
            threats = sorted(
                [r for r in results if r['is_threat']], 
                key=lambda x: x['supervised_probability'], 
                reverse=True
            )[:3]
            
            for i, threat in enumerate(threats, 1):
                print(f"\n{i}. [Risk: {threat['supervised_probability']:.2%}]")
                print(f"   {threat['log_text'][:100]}{'...' if len(threat['log_text']) > 100 else ''}")
                if threat['phishing_detection']['has_phishing']:
                    print("   🚩 Contains phishing URL")

        print("\n" + "=" * 50)
        print(f"📂 Report saved to: {report_file}")
        print(f"📈 Visualization saved to: {plot_file}")
        print("=" * 50)
        
        return results

    def generate_visualization(self, results, report_name):
        if not results:
            return None
            
        normal = sum(1 for r in results if not r["is_threat"])
        anomaly = sum(1 for r in results if r["is_anomalous"])
        phishing = sum(1 for r in results if r["phishing_detection"]["has_phishing"])
        supervised = sum(1 for r in results if r["supervised_prediction"] == 1)

        plt.figure(figsize=(10, 6))
        categories = ["Normal", "Anomalies", "Phishing", "Known Threats"]
        counts = [normal, anomaly, phishing, supervised]
        colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
        bars = plt.bar(categories, counts, color=colors)
        plt.title("Security Event Distribution")
        plt.ylabel("Count")

        # Add count labels
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.5, 
                str(int(bar.get_height())), 
                ha='center'
            )

        # Add watermark
        plt.figtext(0.5, 0.01, "Generated by SIEM Tool", ha="center", fontsize=10, alpha=0.7)

        plot_file = REPORTS_DIR / f"{report_name}_plot.png"
        try:
            plt.savefig(plot_file, bbox_inches="tight")
            plt.close()
            return plot_file
        except Exception as e:
            print(f"❌ Error saving visualization: {e}")
            return None

# ========== NEW INTERACTIVE MODE ==========
def interactive_mode():
    clear_screen()
    print_banner()
    cwd = Path.cwd()
    while True:
        choice = get_user_choice("What would you like to do?", ["Train a new model", "Detect threats in logs", "Exit"])
        if choice == 1:
            clear_screen()
            print("=" * 50)
            print("🚀 MODEL TRAINING")
            print("=" * 50)
            print("Please select a JSONL file containing training data")
            train_file = choose_file_from_dir(cwd, ".jsonl", "Available training files:")
            if train_file:
                SIEMDetector().train(train_file)
            else:
                print("No .jsonl files found.")
            input("\nPress Enter to continue...")
            clear_screen()
            print_banner()
        elif choice == 2:
            clear_screen()
            print("=" * 50)
            print("🔍 THREAT DETECTION")
            print("=" * 50)
            print("Please select a JSON log file")
            detect_file = choose_file_from_dir(cwd, ".json", "Available log files:")
            if detect_file:
                SIEMDetector().detect(detect_file)
            else:
                print("No .json files found.")
            input("\nPress Enter to continue...")
            clear_screen()
            print_banner()
        else:
            print("👋 See ya!")
            sys.exit(0)
# ========== END OF NEW INTERACTIVE MODE ==========

def command_line_mode():
    parser = argparse.ArgumentParser(description=f"SIEM Tool v{VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model from JSONL")
    train_parser.add_argument("file", type=Path, help="Training data JSONL file", nargs='?', default=None)
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect threats from JSON logs")
    detect_parser.add_argument("file", type=Path, help="Log file JSON", nargs='?', default=None)
    
    # Info command
    subparsers.add_parser("info", help="Show system info")

    args = parser.parse_args()
    detector = SIEMDetector()

    if args.command == "train":
        file = args.file or select_file("Select training file:", ["jsonl"], allow_custom=True)
        if file:
            detector.train(file)
        else:
            print("No training file selected")
    
    elif args.command == "detect":
        file = args.file or select_file("Select log file:", ["json"], allow_custom=True)
        if file:
            detector.detect(file)
        else:
            print("No log file selected")
    
    elif args.command == "info":
        print(f"SIEM Tool v{VERSION}")
        print(f"Models directory: {MODELS_DIR}")
        print(f"Reports directory: {REPORTS_DIR}")
        print(f"Logs directory: {LOGS_DIR}")
        print(f"Cache directory: {CACHE_DIR}")
        print("\nCurrent directory files:")
        files = find_local_files(["json", "jsonl"])
        for file in files:
            print(f"- {file.name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command_line_mode()
    else:
        interactive_mode()
