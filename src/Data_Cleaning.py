from pathlib import Path

import numpy as np
import pandas as pd


def resolve_data_path(file_path):
    """Resolve project-relative paths regardless of the current working directory."""
    path = Path(file_path)
    if path.is_absolute():
        return path

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    candidates = [script_dir / path, project_root / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (project_root / path).resolve()


def load_and_clean_data(file_path="Data_set/train.csv"):
    """Load and clean the retail sales dataset."""
    dataset_path = resolve_data_path(file_path)

    if not dataset_path.exists():
        print(f"File not found at '{dataset_path}'. Generating mock dataset for demonstration...")
        df = generate_mock_data()
    else:
        df = pd.read_csv(dataset_path)
        print(f"Successfully loaded data from {dataset_path}")

    initial_rows = len(df)
    rename_map = {
        "Order Date": "OrderDate",
        "Order ID": "OrderID",
        "Sub-Category": "SubCategory",
        "Segment": "CustomerSegment",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})
    df = df.drop_duplicates()

    if "OrderDate" not in df.columns or "Sales" not in df.columns:
        raise ValueError("Input data must contain an order date column and a Sales column.")

    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True, errors="coerce")
    for col in ["Sales", "Quantity", "UnitPrice", "Profit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["OrderDate", "Sales"])

    for col in ["Category", "SubCategory", "Region", "CustomerSegment"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.title()

    print(f"Data cleaning complete. Processed {len(df)} rows (removed {initial_rows - len(df)} invalid/duplicate rows).")
    return df


def feature_engineering(df):
    """Engineer time and revenue features for analysis."""
    df = df.copy()
    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["MonthName"] = df["OrderDate"].dt.strftime("%B")
    df["DayOfWeek"] = df["OrderDate"].dt.day_name()
    df["Quarter"] = df["OrderDate"].dt.quarter

    if "Sales" in df.columns:
        df["Calculated_Revenue"] = df["Sales"]
    elif {"Quantity", "UnitPrice"}.issubset(df.columns):
        df["Calculated_Revenue"] = df["Quantity"] * df["UnitPrice"]
    else:
        raise ValueError("Input data must contain Sales or both Quantity and UnitPrice columns.")

    if "Profit" in df.columns:
        df["Profit_Margin_%"] = np.where(
            df["Calculated_Revenue"] > 0,
            (df["Profit"] / df["Calculated_Revenue"]) * 100,
            0.0,
        )

    return df


def aggregate_business_metrics(df):
    """Aggregate monthly and category-level business KPIs."""
    print("\n" + "=" * 40)
    print("EXECUTIVE RETAIL SALES SUMMARY")
    print("=" * 40)

    monthly_aggregations = {"Total_Revenue": ("Calculated_Revenue", "sum")}
    if "Profit" in df.columns:
        monthly_aggregations["Total_Profit"] = ("Profit", "sum")
    if "OrderID" in df.columns:
        monthly_aggregations["Total_Orders"] = ("OrderID", "nunique")

    monthly_summary = (
        df.groupby(["Year", "Month", "MonthName"])
        .agg(**monthly_aggregations)
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    if "Total_Orders" in monthly_summary.columns:
        monthly_summary["Average_Order_Value"] = (
            monthly_summary["Total_Revenue"] / monthly_summary["Total_Orders"]
        )

    print("\n--- Monthly Performance Snapshot ---")
    print(monthly_summary.head(6).to_string(index=False))

    category_aggregations = {"Revenue": ("Calculated_Revenue", "sum")}
    if "Profit" in df.columns:
        category_aggregations["Profit"] = ("Profit", "sum")
    if "Quantity" in df.columns:
        category_aggregations["Total_Units_Sold"] = ("Quantity", "sum")

    category_summary = (
        df.groupby("Category")
        .agg(**category_aggregations)
        .sort_values(by="Revenue", ascending=False)
        .reset_index()
    )

    print("\n--- Sales by Product Category ---")
    print(category_summary.to_string(index=False))

    return monthly_summary, category_summary


def generate_mock_data():
    """Generate a realistic sample DataFrame for testing or demo runs."""
    rng = np.random.default_rng(42)
    n_rows = 5000
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", periods=n_rows)

    data = {
        "OrderID": [f"ORD-{i:05d}" for i in range(n_rows)],
        "OrderDate": rng.choice(dates, n_rows),
        "Category": rng.choice(["Furniture", "Office Supplies", "Technology"], n_rows),
        "SubCategory": rng.choice(["Chairs", "Binders", "Phones", "Tables", "Paper"], n_rows),
        "Region": rng.choice(["North", "South", "East", "West"], n_rows),
        "CustomerSegment": rng.choice(["Consumer", "Corporate", "Home Office"], n_rows),
        "Quantity": rng.integers(1, 10, n_rows),
        "UnitPrice": np.round(rng.uniform(15, 600, n_rows), 2),
        "Profit": np.round(rng.uniform(-40, 200, n_rows), 2),
    }
    df = pd.DataFrame(data)
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    return df


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "Data_set" / "train.csv"
    output_path = project_root / "Data_set" / "cleaned_retail_sales.csv"

    cleaned_df = load_and_clean_data(str(input_path))
    enriched_df = feature_engineering(cleaned_df)
    aggregate_business_metrics(enriched_df)
    enriched_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    main()