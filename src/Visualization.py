from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


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


def load_cleaned_data(file_path="Data_set/cleaned_retail_sales.csv"):
    """Loads the cleaned retail dataset and parses dates properly."""
    dataset_path = resolve_data_path(file_path)
    df = pd.read_csv(dataset_path)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    return df


def generate_visualizations(df):
    """Generate a 4-panel dashboard for retail sales analysis."""
    fig = plt.figure(figsize=(20, 14))

    ax1 = plt.subplot(2, 2, 1)
    monthly_trend = df.groupby(["Year", "Month", "MonthName"])["Calculated_Revenue"].sum().reset_index()
    monthly_trend = monthly_trend.sort_values(["Year", "Month"])
    monthly_trend["Period"] = monthly_trend["MonthName"].str.slice(0, 3) + " " + monthly_trend["Year"].astype(str)

    sns.lineplot(
        data=monthly_trend,
        x="Period",
        y="Calculated_Revenue",
        hue="Year",
        marker="o",
        linewidth=2.5,
        ax=ax1,
    )
    ax1.set_title("Monthly Revenue Trend Analysis", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Period", fontsize=10)
    ax1.set_ylabel("Total Revenue ($)", fontsize=10)
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    plt.sca(ax1)
    plt.xticks(rotation=45)

    ax2 = plt.subplot(2, 2, 2)
    cat_summary = df.groupby("Category")["Calculated_Revenue"].sum().sort_values(ascending=False).reset_index()

    sns.barplot(
        data=cat_summary,
        x="Category",
        y="Calculated_Revenue",
        hue="Category",
        palette="crest",
        dodge=False,
        ax=ax2,
        legend=False,
    )
    ax2.set_title("Total Revenue by Product Category", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Category", fontsize=10)
    ax2.set_ylabel("Revenue ($)", fontsize=10)
    ax2.yaxis.set_major_formatter("${x:,.0f}")

    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(
            f"${height:,.0f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            xytext=(0, 3),
            textcoords="offset points",
        )

    ax3 = plt.subplot(2, 2, 3)
    if "Region" in df.columns:
        if "Profit" in df.columns:
            region_summary = df.groupby("Region")[["Calculated_Revenue", "Profit"]].sum().reset_index()
            region_melted = region_summary.melt(
                id_vars="Region",
                value_vars=["Calculated_Revenue", "Profit"],
                var_name="Metric",
                value_name="Amount",
            )
            region_melted["Metric"] = region_melted["Metric"].map({"Calculated_Revenue": "Revenue", "Profit": "Profit"})
            palette = "Set2"
        else:
            region_summary = df.groupby("Region")["Calculated_Revenue"].sum().reset_index()
            region_melted = region_summary.rename(columns={"Calculated_Revenue": "Amount"})
            region_melted["Metric"] = "Revenue"
            palette = ["#4c72b0"]

        sns.barplot(
            data=region_melted,
            x="Region",
            y="Amount",
            hue="Metric",
            palette=palette,
            ax=ax3,
        )
        ax3.set_title("Regional Sales Breakdown", fontsize=13, fontweight="bold", pad=12)
        ax3.set_xlabel("Region", fontsize=10)
        ax3.set_ylabel("Amount ($)", fontsize=10)
        ax3.yaxis.set_major_formatter("${x:,.0f}")

    ax4 = plt.subplot(2, 2, 4)
    numeric_cols = ["Quantity", "UnitPrice", "Sales", "Calculated_Revenue", "Profit", "Profit_Margin_%"]
    valid_numeric = [col for col in numeric_cols if col in df.columns]

    if len(valid_numeric) > 1:
        corr = df[valid_numeric].corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax4,
        )
        ax4.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    else:
        ax4.text(0.5, 0.5, "Insufficient numeric features for correlation", horizontalalignment="center", verticalalignment="center")
        ax4.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent.parent / "retail_analytics_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard successfully generated and saved to '{output_path}'")
    plt.show()


if __name__ == "__main__":
    cleaned_df = load_cleaned_data("Data_set/cleaned_retail_sales.csv")
    generate_visualizations(cleaned_df)