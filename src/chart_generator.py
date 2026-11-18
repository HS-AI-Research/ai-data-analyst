
import os
import re
import matplotlib

# Use a non-interactive backend for saving charts
matplotlib.use("Agg")

import matplotlib.pyplot as plt


CHART_DIR = "outputs/charts"


def generate_trend_chart(
    evidence: dict,
    metric: str = "revenue",
    language: str = "en"
) -> str:
    """
    Generate and save a time-based trend chart.

    Supported metrics:
        - revenue
        - quantity
        - orders

    Supported languages:
        - en
        - fa
    """

    results = evidence.get("results", [])

    if not results:
        raise ValueError("No trend data available for chart generation.")

    translations = {
        "en": {
            "revenue": "Revenue ($)",
            "quantity": "Quantity Sold",
            "orders": "Number of Orders",
            "period": "Period",
            "trend": "Trend",
        },
        "fa": {
            "revenue": "درآمد فروش (دلار)",
            "quantity": "تعداد کالاهای فروخته‌شده",
            "orders": "تعداد سفارش‌ها",
            "period": "دوره زمانی",
            "trend": "روند",
        }
    }

    language = language.lower()

    if language not in translations:
        language = "en"

    labels = translations[language]

    if metric not in ("revenue", "quantity", "orders"):
        raise ValueError(f"Unsupported metric: {metric}")

    periods = [item["period"] for item in results]
    values = [item.get(metric, 0) for item in results]

    os.makedirs(CHART_DIR, exist_ok=True)

    granularity = evidence.get("granularity", "period")

    filename = f"{metric}_{granularity}_trend_{language}.png"
    filepath = os.path.join(CHART_DIR, filename)

    plt.figure(figsize=(11, 6))

    plt.plot(
        periods,
        values,
        marker="o",
        linewidth=2
    )

    plt.title(
        f"{labels[metric]} - {granularity.capitalize()} {labels['trend']}"
        if language == "en"
        else f"{labels['trend']} {granularity} - {labels[metric]}"
    )

    plt.xlabel(labels["period"])
    plt.ylabel(labels[metric])

    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

    return filepath
