import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ---------------------------------------------------------
# Set matplotlib headless backend (for server / script execution)
# ---------------------------------------------------------
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------
# Add repository root and final_model directory to sys.path
#
# This ensures custom transformers (FeatureSelector, SelectiveScaler)
# can be unpickled seamlessly when loading the pipeline model.
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "final_model"))

try:
    from custom_transformers import (
        FeatureSelector,
        SelectiveScaler
    )
except ImportError as e:
    print(f"Warning: Unable to import custom_transformers from final_model: {e}")


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

# Encoded dataset path
INPUT_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "processed data",
    "encoded_transactions.csv"
)

# Trained production model pipeline path
MODEL_FILE = os.path.join(
    BASE_DIR,
    "final_model",
    "models",
    "ros_logistic_end_to_end_finetuned.pkl"
)

# Results output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_model",
    "evaluation",
    "results"
)

# Output report file
REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "threshold_optimization_report.txt"
)

# Output threshold sweep CSV
SWEEP_CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "threshold_sweep_results.csv"
)

# Output visualization curve files
METRICS_CURVE_FILE = os.path.join(
    OUTPUT_DIR,
    "threshold_metrics_curve.png"
)

COST_CURVE_FILE = os.path.join(
    OUTPUT_DIR,
    "threshold_cost_curve.png"
)


# =========================================================
# Class: ThresholdOptimizer
#
# Contains all methods required to evaluate, optimize, and
# visualize decision thresholds for financial fraud detection,
# with robust validation and exception handling across every method.
# =========================================================
class ThresholdOptimizer:

    def __init__(self, cost_fn=1000.0, cost_fp=25.0):
        """
        Initialize the ThresholdOptimizer.

        Parameters:
        -----------
        cost_fn : float
            Estimated average business loss incurred for each
            undetected fraudulent transaction (False Negative).
            Default: $1,000.
        cost_fp : float
            Estimated operational cost incurred for each
            false alarm / verification review (False Positive).
            Default: $25.
        """
        if cost_fn < 0 or cost_fp < 0:
            raise ValueError("Cost values (cost_fn, cost_fp) must be non-negative numbers.")
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)

    # -----------------------------------------------------
    # Method to load dataset with exception handling
    # -----------------------------------------------------
    def load_data(self, file_path):
        """
        Load processed transaction dataset from CSV file.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Required dataset file does not exist at: {file_path}"
                )

            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError(
                    f"Dataset loaded from {file_path} is completely empty."
                )

            return df

        except FileNotFoundError as e:
            print(f"[Error in load_data] File not found: {e}")
            raise e
        except pd.errors.EmptyDataError as e:
            print(f"[Error in load_data] File contains no data: {e}")
            raise e
        except Exception as e:
            print(f"[Error in load_data] Failed to load dataset: {e}")
            raise e

    # -----------------------------------------------------
    # Method to split features and target with exception handling
    # -----------------------------------------------------
    def split_data(self, df, target_column="Suspicious Activity Flag", test_size=0.20, random_state=42):
        """
        Replicate the exact stratified 80/20 train/test split
        used during final model training and evaluation.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input 'df' must be a pandas DataFrame.")

            if target_column not in df.columns:
                raise KeyError(
                    f"Target column '{target_column}' was not found in dataset columns."
                )

            if not (0.0 < test_size < 1.0):
                raise ValueError(
                    f"test_size must be strictly between 0 and 1. Received: {test_size}"
                )

            X = df.drop(target_column, axis=1)
            y = df[target_column]

            if y.nunique() < 2:
                raise ValueError(
                    "Target column must contain at least 2 distinct classes for classification."
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )

            return X_train, X_test, y_train, y_test

        except KeyError as e:
            print(f"[Error in split_data] Target column missing: {e}")
            raise e
        except ValueError as e:
            print(f"[Error in split_data] Invalid split parameters or target classes: {e}")
            raise e
        except Exception as e:
            print(f"[Error in split_data] Unexpected error during train/test split: {e}")
            raise e

    # -----------------------------------------------------
    # Method to load trained model pipeline with exception handling
    # -----------------------------------------------------
    def load_model(self, model_path):
        """
        Load serialized scikit-learn / imbalanced-learn pipeline model.
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Trained model artifact was not found at: {model_path}"
                )

            model = joblib.load(model_path)

            if not hasattr(model, "predict_proba"):
                raise AttributeError(
                    "Loaded model does not implement 'predict_proba', which is required for threshold optimization."
                )

            return model

        except FileNotFoundError as e:
            print(f"[Error in load_model] Model file not found: {e}")
            raise e
        except Exception as e:
            print(f"[Error in load_model] Failed to unpickle / load model: {e}")
            raise e

    # -----------------------------------------------------
    # Method to predict fraud probabilities with exception handling
    # -----------------------------------------------------
    def predict_probabilities(self, model, X):
        """
        Generate continuous probability scores for the positive
        (fraudulent / suspicious) class.
        """
        try:
            if model is None:
                raise ValueError("Model object is None. Load a valid model before predicting.")

            if X is None or len(X) == 0:
                raise ValueError("Input feature dataset X is empty.")

            probabilities = model.predict_proba(X)

            if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                raise ValueError("predict_proba must return an array with at least 2 columns.")

            return probabilities[:, 1]

        except AttributeError as e:
            print(f"[Error in predict_probabilities] Model missing predict_proba method: {e}")
            raise e
        except Exception as e:
            print(f"[Error in predict_probabilities] Failed during probability prediction: {e}")
            raise e

    # -----------------------------------------------------
    # Method to evaluate metrics at a specific threshold
    # -----------------------------------------------------
    def calculate_metrics_at_threshold(self, y_true, y_prob, threshold):
        """
        Compute all classification metrics and financial cost
        for a given decision threshold cutoff.
        """
        try:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"Threshold must be between 0.0 and 1.0. Received: {threshold}"
                )

            if len(y_true) != len(y_prob):
                raise ValueError(
                    f"Length mismatch: len(y_true)={len(y_true)} vs len(y_prob)={len(y_prob)}"
                )

            # Binary prediction based on threshold
            y_pred = (y_prob >= threshold).astype(int)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape != (2, 2):
                # Edge cases where all samples are predicted as one class
                tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            else:
                tn, fp, fn, tp = cm.ravel()

            # Classification metrics
            accuracy = float(accuracy_score(y_true, y_pred))
            error_rate = float(1.0 - accuracy)
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

            # Rates
            tpr = recall
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            youden_j = float(tpr - fpr)

            # Financial cost calculation:
            # Total Cost = (Missed Frauds * Cost_FN) + (False Alarms * Cost_FP)
            total_cost = float((fn * self.cost_fn) + (fp * self.cost_fp))

            return {
                "Threshold": round(float(threshold), 4),
                "Accuracy": accuracy,
                "Error_Rate": error_rate,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "F2_Score": f2,
                "Youden_J": youden_j,
                "True_Negatives": int(tn),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
                "True_Positives": int(tp),
                "FPR": fpr,
                "Total_Cost": total_cost
            }

        except ValueError as e:
            print(f"[Error in calculate_metrics_at_threshold] Value error: {e}")
            raise e
        except Exception as e:
            print(f"[Error in calculate_metrics_at_threshold] Calculation error: {e}")
            raise e

    # -----------------------------------------------------
    # Method to execute full threshold sweep with exception handling
    # -----------------------------------------------------
    def run_threshold_sweep(self, y_true, y_prob, thresholds=None):
        """
        Run a systematic sweep over candidate thresholds to evaluate
        metric trends and build a comparative DataFrame.
        """
        try:
            if thresholds is None:
                thresholds = np.linspace(0.05, 0.95, 19)

            results = []
            for th in thresholds:
                metrics = self.calculate_metrics_at_threshold(y_true, y_prob, float(th))
                results.append(metrics)

            sweep_df = pd.DataFrame(results)
            return sweep_df

        except Exception as e:
            print(f"[Error in run_threshold_sweep] Failed during threshold sweep: {e}")
            raise e

    # -----------------------------------------------------
    # Method to find optimal threshold for Maximum F1-Score
    # -----------------------------------------------------
    def find_optimal_f1(self, y_true, y_prob):
        """
        Find threshold that maximizes the standard F1-score
        (harmonic mean of Precision and Recall).
        """
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

            if len(thresholds) == 0:
                return self.calculate_metrics_at_threshold(y_true, y_prob, 0.50)

            # Compute F1 scores across thresholds
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
            best_idx = int(np.argmax(f1_scores))
            best_threshold = float(thresholds[best_idx])

            best_metrics = self.calculate_metrics_at_threshold(y_true, y_prob, best_threshold)
            best_metrics["Strategy"] = "Max F1-Score (Balanced)"
            return best_metrics

        except Exception as e:
            print(f"[Error in find_optimal_f1] Failed to optimize F1 threshold: {e}")
            raise e

    # -----------------------------------------------------
    # Method to find optimal threshold for Maximum F2-Score
    # -----------------------------------------------------
    def find_optimal_f2(self, y_true, y_prob):
        """
        Find threshold that maximizes the F2-score
        (weights Recall twice as heavily as Precision).
        Recommended for fraud detection to minimize missed fraud.
        """
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

            if len(thresholds) == 0:
                return self.calculate_metrics_at_threshold(y_true, y_prob, 0.50)

            beta = 2.0
            f2_scores = (1 + beta**2) * (precisions[:-1] * recalls[:-1]) / (
                (beta**2 * precisions[:-1]) + recalls[:-1] + 1e-10
            )
            best_idx = int(np.argmax(f2_scores))
            best_threshold = float(thresholds[best_idx])

            best_metrics = self.calculate_metrics_at_threshold(y_true, y_prob, best_threshold)
            best_metrics["Strategy"] = "Max F2-Score (Recall-Weighted)"
            return best_metrics

        except Exception as e:
            print(f"[Error in find_optimal_f2] Failed to optimize F2 threshold: {e}")
            raise e

    # -----------------------------------------------------
    # Method to find optimal threshold via Youden's J Statistic
    # -----------------------------------------------------
    def find_optimal_youden(self, y_true, y_prob):
        """
        Find threshold that maximizes Youden's Index:
        J = Sensitivity + Specificity - 1 = True Positive Rate - False Positive Rate.
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)

            if len(thresholds) == 0:
                return self.calculate_metrics_at_threshold(y_true, y_prob, 0.50)

            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            best_threshold = float(thresholds[best_idx])

            # In ROC curve, threshold[0] can exceed 1.0 (scikit-learn convention); clamp it
            best_threshold = min(best_threshold, 1.0)

            best_metrics = self.calculate_metrics_at_threshold(y_true, y_prob, best_threshold)
            best_metrics["Strategy"] = "Youden's J-Index (Max TPR - FPR)"
            return best_metrics

        except Exception as e:
            print(f"[Error in find_optimal_youden] Failed to optimize Youden J threshold: {e}")
            raise e

    # -----------------------------------------------------
    # Method to find cost-optimal threshold
    # -----------------------------------------------------
    def find_optimal_cost(self, y_true, y_prob, search_steps=1000):
        """
        Find threshold that minimizes total expected monetary loss:
        Total Cost = (FN * Cost_FN) + (FP * Cost_FP).
        """
        try:
            candidate_thresholds = np.linspace(0.01, 0.99, search_steps)
            best_cost = float("inf")
            best_threshold = 0.50

            for th in candidate_thresholds:
                y_pred = (y_prob >= th).astype(int)
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                cost = (fn * self.cost_fn) + (fp * self.cost_fp)
                if cost < best_cost:
                    best_cost = cost
                    best_threshold = th

            best_metrics = self.calculate_metrics_at_threshold(y_true, y_prob, float(best_threshold))
            best_metrics["Strategy"] = f"Min Financial Loss (FN=${self.cost_fn:.0f}, FP=${self.cost_fp:.0f})"
            return best_metrics

        except Exception as e:
            print(f"[Error in find_optimal_cost] Failed to optimize cost threshold: {e}")
            raise e

    # -----------------------------------------------------
    # Method to save threshold sweep results to CSV
    # -----------------------------------------------------
    def save_sweep_results(self, sweep_df, file_path):
        """
        Persist threshold sweep DataFrame to CSV file.
        Returns True on success, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            sweep_df.to_csv(file_path, index=False)
            return True
        except (IOError, OSError) as e:
            print(f"[Error in save_sweep_results] File system write error: {e}")
            return False
        except Exception as e:
            print(f"[Error in save_sweep_results] Unexpected error saving sweep CSV: {e}")
            return False

    # -----------------------------------------------------
    # Method to plot threshold performance metrics curve
    # -----------------------------------------------------
    def plot_metrics_curve(self, sweep_df, optimal_points, output_path):
        """
        Plot Precision, Recall, F1, and F2 curves across thresholds
        and highlight optimal operating points. Returns True on success.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            plt.figure(figsize=(11, 7))
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

            plt.plot(sweep_df["Threshold"], sweep_df["Recall"], label="Recall (Fraud Detection Rate)", color="#2ca02c", linewidth=2.5)
            plt.plot(sweep_df["Threshold"], sweep_df["Precision"], label="Precision", color="#1f77b4", linewidth=2.5)
            plt.plot(sweep_df["Threshold"], sweep_df["F1_Score"], label="F1-Score", color="#ff7f0e", linewidth=2, linestyle="--")
            plt.plot(sweep_df["Threshold"], sweep_df["F2_Score"], label="F2-Score (Recall-Weighted)", color="#9467bd", linewidth=2, linestyle=":")
            plt.plot(sweep_df["Threshold"], sweep_df["Error_Rate"], label="Classification Error Rate", color="#d62728", linewidth=1.8, linestyle="-.")

            # Highlight optimal thresholds
            for opt in optimal_points:
                th = opt["Threshold"]
                strategy = opt.get("Strategy", f"Threshold {th:.2f}")
                plt.axvline(x=th, linestyle="--", alpha=0.6, label=f"{strategy}: {th:.2f}")

            plt.title("Financial Fraud Detection - Decision Threshold Optimization Curves", fontsize=14, fontweight="bold", pad=15)
            plt.xlabel("Classification Decision Threshold", fontsize=12, fontweight="bold")
            plt.ylabel("Metric Score", fontsize=12, fontweight="bold")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.05)
            plt.legend(loc="lower left", frameon=True, shadow=True, fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            return True

        except (IOError, OSError) as e:
            print(f"[Error in plot_metrics_curve] File system save error: {e}")
            plt.close()
            return False
        except Exception as e:
            print(f"[Error in plot_metrics_curve] Plotting error: {e}")
            plt.close()
            return False

    # -----------------------------------------------------
    # Method to plot financial cost curve
    # -----------------------------------------------------
    def plot_cost_curve(self, y_true, y_prob, optimal_cost_metric, output_path, steps=200):
        """
        Plot expected monetary loss ($) across classification thresholds.
        Returns True on success.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            thresholds = np.linspace(0.02, 0.98, steps)
            costs = []
            fn_losses = []
            fp_losses = []

            for th in thresholds:
                y_pred = (y_prob >= th).astype(int)
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                fn_cost = fn * self.cost_fn
                fp_cost = fp * self.cost_fp
                costs.append(fn_cost + fp_cost)
                fn_losses.append(fn_cost)
                fp_losses.append(fp_cost)

            plt.figure(figsize=(11, 7))
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

            plt.plot(thresholds, costs, label="Total Expected Loss ($)", color="#d62728", linewidth=2.8)
            plt.plot(thresholds, fn_losses, label=f"Missed Fraud Loss (FN @ ${self.cost_fn:.0f}/miss)", color="#8c564b", linewidth=1.8, linestyle="--")
            plt.plot(thresholds, fp_losses, label=f"False Alarm Review Cost (FP @ ${self.cost_fp:.0f}/alarm)", color="#1f77b4", linewidth=1.8, linestyle=":")

            # Mark optimal cost point
            opt_th = optimal_cost_metric["Threshold"]
            opt_cost = optimal_cost_metric["Total_Cost"]
            plt.scatter([opt_th], [opt_cost], color="black", s=100, zorder=5)
            plt.annotate(
                f"Minimum Loss Point\nThreshold: {opt_th:.2f}\nTotal Cost: ${opt_cost:,.0f}",
                xy=(opt_th, opt_cost),
                xytext=(opt_th + 0.08, opt_cost + 5000),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=7),
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffe0", edgecolor="#888888")
            )

            plt.title(f"Cost-Sensitive Threshold Optimization (FN=${self.cost_fn:.0f}, FP=${self.cost_fp:.0f})", fontsize=14, fontweight="bold", pad=15)
            plt.xlabel("Classification Decision Threshold", fontsize=12, fontweight="bold")
            plt.ylabel("Total Expected Cost ($)", fontsize=12, fontweight="bold")
            plt.xlim(0.0, 1.0)
            plt.legend(loc="upper center", frameon=True, shadow=True, fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            return True

        except (IOError, OSError) as e:
            print(f"[Error in plot_cost_curve] File system save error: {e}")
            plt.close()
            return False
        except Exception as e:
            print(f"[Error in plot_cost_curve] Plotting error: {e}")
            plt.close()
            return False

    # -----------------------------------------------------
    # Method to format report string
    # -----------------------------------------------------
    def format_report(self, baseline_metrics, optimal_strategies, total_samples, total_frauds):
        """
        Format the threshold optimization benchmark results into a clean string.
        """
        lines = []
        lines.append("=" * 72)
        lines.append("FINANCIAL FRAUD DETECTION - THRESHOLD OPTIMIZATION REPORT")
        lines.append("=" * 72)
        lines.append(f"Hold-out Test Set Size:     {total_samples:,} transactions")
        lines.append(f"Ground Truth Frauds:        {total_frauds:,} transactions ({(total_frauds/total_samples)*100:.2f}%)")
        lines.append(f"Ground Truth Legitimate:    {total_samples - total_frauds:,} transactions")
        lines.append(f"Cost Assumption:            Missed Fraud (FN) = ${self.cost_fn:,.0f} | False Alarm (FP) = ${self.cost_fp:,.0f}")
        lines.append("=" * 72)
        lines.append("")

        lines.append("1. COMPARATIVE STRATEGY BENCHMARK")
        lines.append("-" * 72)
        header = f"{'Strategy':<30} {'Thresh':<8} {'Recall':<9} {'Precision':<10} {'F1':<8} {'FP':<5} {'FN':<5} {'Cost ($)':<10}"
        lines.append(header)
        lines.append("-" * 72)

        all_entries = [baseline_metrics] + optimal_strategies
        for m in all_entries:
            name = m.get("Strategy", f"Threshold {m['Threshold']:.2f}")
            recall_str = f"{m['Recall']*100:.2f}%"
            prec_str = f"{m['Precision']*100:.2f}%"
            f1_str = f"{m['F1_Score']*100:.2f}%"
            cost_str = f"${m['Total_Cost']:,.0f}"
            line = (
                f"{name:<30} "
                f"{m['Threshold']:<8.2f} "
                f"{recall_str:<9} "
                f"{prec_str:<10} "
                f"{f1_str:<8} "
                f"{m['False_Positives']:<5d} "
                f"{m['False_Negatives']:<5d} "
                f"{cost_str:<10}"
            )
            lines.append(line)
        lines.append("-" * 72)
        lines.append("")

        lines.append("2. IN-DEPTH ANALYSIS OF OPTIMAL STRATEGIES")
        lines.append("-" * 72)
        for m in all_entries:
            name = m.get("Strategy", f"Threshold {m['Threshold']:.2f}")
            lines.append(f"> Strategy: {name}")
            lines.append(f"   Decision Cutoff:       {m['Threshold']:.4f}")
            lines.append(f"   Accuracy:              {m['Accuracy']*100:.2f}% (Error: {m['Error_Rate']*100:.2f}%)")
            lines.append(f"   Fraud Recall:          {m['Recall']*100:.2f}% (Caught {m['True_Positives']} / {total_frauds})")
            lines.append(f"   Precision:             {m['Precision']*100:.2f}%")
            lines.append(f"   F1-Score:              {m['F1_Score']*100:.2f}%")
            lines.append(f"   F2-Score:              {m['F2_Score']*100:.2f}%")
            lines.append(f"   False Positives (FP):  {m['False_Positives']} (False alarm rate: {m['FPR']*100:.2f}%)")
            lines.append(f"   False Negatives (FN):  {m['False_Negatives']} missed frauds")
            lines.append(f"   Financial Loss:        ${m['Total_Cost']:,.0f}")
            lines.append("")

        lines.append("3. OPERATIONAL RECOMMENDATIONS FOR PRODUCTION DEPLOYMENT")
        lines.append("-" * 72)
        lines.append("* Tier 1: Auto-Approve (P < 0.35)")
        lines.append("  Transactions with fraud probability below 0.35 represent extremely low risk.")
        lines.append("  Recommended action: Instant Authorization without friction.")
        lines.append("")
        lines.append("* Tier 2: Step-Up Authentication / Human Review (0.35 <= P < 0.50)")
        lines.append("  Captures the critical fraud boundary (recovers 50% to 67% of missed frauds).")
        lines.append("  Recommended action: Automated OTP / Biometric verification or queue for fast analyst review.")
        lines.append("")
        lines.append("* Tier 3: Immediate Decline / Auto-Block (P >= 0.50)")
        lines.append("  High certainty malicious transactions.")
        lines.append("  Recommended action: Immediate payment block and push notification to account owner.")
        lines.append("=" * 72)

        return "\n".join(lines)

    # -----------------------------------------------------
    # Method to save report to file with exception handling
    # -----------------------------------------------------
    def save_report(self, report_content, file_path):
        """
        Write formatted report string to text file.
        Returns True on success, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            return True
        except (IOError, OSError) as e:
            print(f"[Error in save_report] File write error: {e}")
            return False
        except Exception as e:
            print(f"[Error in save_report] Unexpected error saving report: {e}")
            return False


# =========================================================
# Main Execution Pipeline with Granular Exception Handling
# =========================================================
def main():
    try:
        print("\n" + "=" * 65)
        print("STARTING THRESHOLD OPTIMIZATION PIPELINE")
        print("=" * 65)

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Initialize optimizer
        optimizer = ThresholdOptimizer(cost_fn=1000.0, cost_fp=25.0)

        # Load dataset
        print(f"\n[1/6] Loading processed dataset from: {INPUT_FILE}")
        df = optimizer.load_data(INPUT_FILE)
        print(f"      Dataset loaded successfully. Shape: {df.shape}")

        # Split hold-out test set
        print("\n[2/6] Splitting hold-out test set (Stratified 80/20 split)...")
        X_train, X_test, y_train, y_test = optimizer.split_data(df)
        total_test = len(y_test)
        total_frauds = int(y_test.sum())
        print(f"      Test set size: {total_test:,} transactions")
        print(f"      Fraud cases in test set: {total_frauds:,} ({(total_frauds/total_test)*100:.2f}%)")

        # Load model pipeline
        print(f"\n[3/6] Loading trained model pipeline: {MODEL_FILE}")
        model = optimizer.load_model(MODEL_FILE)
        print("      Model pipeline loaded successfully.")

        # Predict probabilities
        print("\n[4/6] Generating calibrated fraud probabilities on test set...")
        y_prob = optimizer.predict_probabilities(model, X_test)

        # Baseline evaluation (Threshold = 0.50)
        baseline_metrics = optimizer.calculate_metrics_at_threshold(y_test, y_prob, 0.50)
        baseline_metrics["Strategy"] = "Default Baseline (0.50)"

        # Threshold optimization strategies
        print("\n[5/6] Computing optimal decision thresholds across criteria...")
        optimal_f1 = optimizer.find_optimal_f1(y_test, y_prob)
        optimal_f2 = optimizer.find_optimal_f2(y_test, y_prob)
        optimal_youden = optimizer.find_optimal_youden(y_test, y_prob)
        optimal_cost = optimizer.find_optimal_cost(y_test, y_prob)

        optimal_strategies = [
            optimal_f1,
            optimal_f2,
            optimal_youden,
            optimal_cost
        ]

        # Systematic threshold sweep
        sweep_df = optimizer.run_threshold_sweep(y_test, y_prob)
        sweep_saved = optimizer.save_sweep_results(sweep_df, SWEEP_CSV_FILE)
        if sweep_saved:
            print(f"      Threshold sweep table saved to: {SWEEP_CSV_FILE}")

        # Generate plots
        print("\n[6/6] Generating high-resolution optimization curves...")
        metrics_plot_saved = optimizer.plot_metrics_curve(
            sweep_df,
            [optimal_f2, optimal_youden, optimal_f1],
            METRICS_CURVE_FILE
        )
        if metrics_plot_saved:
            print(f"      Metrics curve saved to: {METRICS_CURVE_FILE}")

        cost_plot_saved = optimizer.plot_cost_curve(
            y_test,
            y_prob,
            optimal_cost,
            COST_CURVE_FILE
        )
        if cost_plot_saved:
            print(f"      Financial cost curve saved to: {COST_CURVE_FILE}")

        # Format and save report
        report_content = optimizer.format_report(
            baseline_metrics,
            optimal_strategies,
            total_test,
            total_frauds
        )
        report_saved = optimizer.save_report(report_content, REPORT_FILE)
        if report_saved:
            print(f"\n      Full optimization report saved to: {REPORT_FILE}")

        # Print summary to console
        print("\n" + report_content)

        print("\n" + "=" * 65)
        print("THRESHOLD OPTIMIZATION COMPLETED SUCCESSFULLY")
        print("=" * 65)

    # ---------------------------------------------------------
    # Granular Exception Handling Blocks (Consistent with Pipeline)
    # ---------------------------------------------------------
    except FileNotFoundError as e:
        print("\n[Error] Required file was not found during threshold optimization:")
        print(f"        {e}")

    except KeyError as e:
        print("\n[Error] Column or key error during threshold optimization:")
        print(f"        {e}")

    except ValueError as e:
        print("\n[Error] Invalid parameter or data value encountered:")
        print(f"        {e}")

    except (IOError, OSError) as e:
        print("\n[Error] File system / IO error occurred while reading or saving files:")
        print(f"        {e}")

    except Exception as e:
        print("\n[Error] Unexpected exception occurred during threshold optimization:")
        print(f"        {e}")


# Run main() when script is executed directly
if __name__ == "__main__":
    main()
