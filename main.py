import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class MeterReading:
    timestamp: pd.Timestamp
    kwh: float


class Building:
    def __init__(self, name: str):
        self.name = name
        self.meter_readings: List[MeterReading] = []

    def add_reading(self, timestamp: pd.Timestamp, kwh: float) -> None:
        """Add a single meter reading."""
        self.meter_readings.append(MeterReading(timestamp=timestamp, kwh=kwh))

    def calculate_total_consumption(self) -> float:
        """Total energy consumed by this building (kWh)."""
        return sum(r.kwh for r in self.meter_readings)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert readings to a DataFrame (for extra analysis if needed)."""
        if not self.meter_readings:
            return pd.DataFrame(columns=["timestamp", "kwh", "building"])
        return pd.DataFrame(
            {
                "timestamp": [r.timestamp for r in self.meter_readings],
                "kwh": [r.kwh for r in self.meter_readings],
                "building": [self.name] * len(self.meter_readings),
            }
        )

    def generate_report(self) -> str:
        """Simple text summary for this building."""
        total = self.calculate_total_consumption()
        return f"Building {self.name}: total consumption = {total:.2f} kWh"


class BuildingManager:
    def __init__(self):
        self.buildings: Dict[str, Building] = {}

    def get_or_create_building(self, name: str) -> Building:
        if name not in self.buildings:
            self.buildings[name] = Building(name)
        return self.buildings[name]

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load readings from a combined DataFrame with columns:
        ['timestamp', 'kwh', 'building']
        """
        for _, row in df.iterrows():
            bld_name = row["building"]
            ts = row["timestamp"]
            kwh = row["kwh"]
            building = self.get_or_create_building(bld_name)
            building.add_reading(ts, kwh)

    def total_campus_consumption(self) -> float:
        return sum(b.calculate_total_consumption() for b in self.buildings.values())

    def building_reports(self) -> List[str]:
        return [b.generate_report() for b in self.buildings.values()]


def infer_building_name_from_path(path: Path) -> str:
    """
    Infer building name from filename.
    e.g. 'library_jan.csv' -> 'library'
         'HostelA.csv'     -> 'HostelA'
    """
    stem = path.stem  
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def load_and_merge_data(data_dir: str) -> pd.DataFrame:
    """
    Read all CSV files from data_dir, merge into one DataFrame,
    and handle basic errors.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_rows = []
    missing_files_log = []
    corrupt_rows_log = []

    for csv_file in data_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, on_bad_lines="skip")

            if "timestamp" not in df.columns or "kwh" not in df.columns:
                print(f"[WARNING] File {csv_file.name} missing 'timestamp' or 'kwh' column. Skipping.")
                continue

            
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "kwh"])

            if "building" not in df.columns:
                building_name = infer_building_name_from_path(csv_file)
                df["building"] = building_name

            all_rows.append(df)

        except FileNotFoundError:
            missing_files_log.append(csv_file.name)
        except Exception as e:
            corrupt_rows_log.append((csv_file.name, str(e)))
            print(f"[ERROR] Problem reading {csv_file.name}: {e}")

    if not all_rows:
        raise ValueError("No valid CSV files found in data directory.")

    df_combined = pd.concat(all_rows, ignore_index=True)

    df_combined = df_combined.sort_values(by=["building", "timestamp"]).reset_index(drop=True)

    print("=== Data Ingestion Summary ===")
    print(f"Total rows loaded: {len(df_combined)}")
    if missing_files_log:
        print("Missing files:")
        for f in missing_files_log:
            print(" -", f)
    if corrupt_rows_log:
        print("Files with issues:")
        for f, msg in corrupt_rows_log:
            print(f" - {f}: {msg}")

    return df_combined


def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily total kWh per building.
    Returns DataFrame with columns: building, date, total_kwh
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    daily = (
        df.groupby(["building", "date"])["kwh"]
        .sum()
        .reset_index()
        .rename(columns={"kwh": "total_kwh"})
    )
    return daily


def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly total kWh per building using resample.
    Assumes timestamp is datetime.
    Returns DataFrame with: building, week_start, total_kwh
    """
    df = df.copy()
    df = df.set_index("timestamp")
    weekly_list = []

    for building, group in df.groupby("building"):
        weekly = (
            group["kwh"]
            .resample("W")
            .sum()
            .reset_index()
            .rename(columns={"kwh": "total_kwh", "timestamp": "week_start"})
        )
        weekly["building"] = building
        weekly_list.append(weekly)

    weekly_df = pd.concat(weekly_list, ignore_index=True)
    return weekly_df


def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics per building: mean, min, max, total.
    """
    summary = (
        df.groupby("building")["kwh"]
        .agg(["mean", "min", "max", "sum"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_kwh",
                "min": "min_kwh",
                "max": "max_kwh",
                "sum": "total_kwh",
            }
        )
    )
    return summary

def create_dashboard(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    building_summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a multi-plot dashboard.png:
    1. Line chart: daily totals over time for each building
    2. Bar chart: average weekly usage per building
    3. Scatter: timestamp vs kwh (peak points)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("Campus Energy-Use Dashboard", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    for building in daily_df["building"].unique():
        sub = daily_df[daily_df["building"] == building]
        ax1.plot(sub["date"], sub["total_kwh"], marker="o", label=building)
    ax1.set_title("Daily Energy Consumption")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total kWh")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    
    ax2 = axes[1]
    weekly_avg = (
        weekly_df.groupby("building")["total_kwh"]
        .mean()
        .reset_index()
        .rename(columns={"total_kwh": "avg_weekly_kwh"})
    )
    ax2.bar(weekly_avg["building"], weekly_avg["avg_weekly_kwh"])
    ax2.set_title("Average Weekly Energy Usage per Building")
    ax2.set_xlabel("Building")
    ax2.set_ylabel("Avg Weekly kWh")
    ax2.grid(True, axis="y", alpha=0.3)

    
    ax3 = axes[2]
    
    N = 50
    
    
    for building in daily_df["building"].unique():
        sub = daily_df[daily_df["building"] == building]
    
        sub_peaks = sub.nlargest(N, "total_kwh")
        ax3.scatter(sub_peaks["date"], sub_peaks["total_kwh"], label=building, alpha=0.7)
    ax3.set_title("Peak Daily Consumption Points")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Daily kWh")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Dashboard saved to {output_path}")

def generate_text_summary(
    df: pd.DataFrame,
    building_summary_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    output_txt_path: str,
) -> None:
    """
    Generate summary.txt with:
    - Total campus consumption
    - Highest-consuming building
    - Peak load time
    - Basic daily/weekly trends
    """
    total_campus = df["kwh"].sum()

    
    top_building_row = building_summary_df.sort_values("total_kwh", ascending=False).iloc[0]
    top_building = top_building_row["building"]
    top_building_kwh = top_building_row["total_kwh"]

    
    peak_row = df.loc[df["kwh"].idxmax()]
    peak_time = peak_row["timestamp"]
    peak_kwh = peak_row["kwh"]
    peak_building = peak_row["building"]

    
    daily_totals_overall = (
        daily_df.groupby("date")["total_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"total_kwh": "campus_daily_kwh"})
    )
    min_daily = daily_totals_overall["campus_daily_kwh"].min()
    max_daily = daily_totals_overall["campus_daily_kwh"].max()

    lines = []
    lines.append("Campus Energy Consumption Summary")
    lines.append("================================")
    lines.append(f"Total campus consumption: {total_campus:.2f} kWh")
    lines.append("")
    lines.append(f"Highest-consuming building: {top_building} ({top_building_kwh:.2f} kWh)")
    lines.append("")
    lines.append("Peak Load Details:")
    lines.append(f" - Building : {peak_building}")
    lines.append(f" - Time     : {peak_time}")
    lines.append(f" - kWh      : {peak_kwh:.2f}")
    lines.append("")
    lines.append("Daily Trend Insights:")
    lines.append(f" - Lowest daily campus total : {min_daily:.2f} kWh")
    lines.append(f" - Highest daily campus total: {max_daily:.2f} kWh")
    lines.append("")
    lines.append("Observations:")
    lines.append(" - You can add your own observations here based on the charts.")
    lines.append(" - For example, identify buildings with consistently high load or unusual peaks.")
    lines.append(" - Suggest potential energy-saving opportunities (like shifting load, improving insulation, etc.).")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Summary report saved to {output_txt_path}")


def save_outputs(
    df_clean: pd.DataFrame,
    building_summary_df: pd.DataFrame,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cleaned_csv = output_path / "cleaned_energy_data.csv"
    summary_csv = output_path / "building_summary.csv"

    df_clean.to_csv(cleaned_csv, index=False)
    building_summary_df.to_csv(summary_csv, index=False)

    print(f"[INFO] Cleaned data saved to {cleaned_csv}")
    print(f"[INFO] Building summary saved to {summary_csv}")


def main(
    data_dir: str = "data",
    output_dir: str = "output",
) -> None:

    df = load_and_merge_data(data_dir)

    daily_totals = calculate_daily_totals(df)
    weekly_aggregates = calculate_weekly_aggregates(df)
    building_summary_df = building_wise_summary(df)

    manager = BuildingManager()
    manager.load_from_dataframe(df)
    print("=== OOP Building Reports ===")
    for report in manager.building_reports():
        print(report)
    print(f"Total campus consumption (via manager): {manager.total_campus_consumption():.2f} kWh")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dashboard_png = output_path / "dashboard.png"
    create_dashboard(
        daily_df=daily_totals,
        weekly_df=weekly_aggregates,
        building_summary_df=building_summary_df,
        output_path=str(dashboard_png),
    )

    save_outputs(df_clean=df, building_summary_df=building_summary_df, output_dir=output_dir)

    summary_txt_path = output_path / "summary.txt"
    generate_text_summary(
        df=df,
        building_summary_df=building_summary_df,
        daily_df=daily_totals,
        output_txt_path=str(summary_txt_path),
    )


if __name__ == "__main__":
    main()
