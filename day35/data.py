"""
Day 35 Data Module

Handles:
1. Olist dataset loading
2. Data preparation
3. Synthetic fallback data generation
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"


# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================

def generate_demo_data(
    seed=42,
    n_customers=1000,
    n_orders=5000,
    n_products=300
):

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Customers
    # --------------------------------------------------------

    customers = pd.DataFrame({

        "customer_id": [
            f"CUST_{i:05d}"
            for i in range(n_customers)
        ]

    })

    # --------------------------------------------------------
    # Products
    # --------------------------------------------------------

    categories = [
        "beauty",
        "electronics",
        "fashion",
        "home",
        "sports",
        "books",
        "automotive",
        "health"
    ]

    products = pd.DataFrame({

        "product_id": [
            f"PROD_{i:04d}"
            for i in range(n_products)
        ],

        "product_category_name":
        rng.choice(
            categories,
            n_products
        )
    })

    # --------------------------------------------------------
    # Orders
    # --------------------------------------------------------

    customer_ids = rng.choice(
        customers["customer_id"],
        n_orders
    )

    product_ids = rng.choice(
        products["product_id"],
        n_orders
    )

    dates = pd.date_range(
        start="2017-01-01",
        end="2018-08-01",
        periods=n_orders
    )

    prices = np.maximum(
        rng.lognormal(
            mean=3.5,
            sigma=0.7,
            size=n_orders
        ),
        5
    )

    freight = np.maximum(
        rng.normal(
            loc=15,
            scale=7,
            size=n_orders
        ),
        2
    )

    orders = pd.DataFrame({

        "order_id": [
            f"ORD_{i:06d}"
            for i in range(n_orders)
        ],

        "customer_id": customer_ids,

        "product_id": product_ids,

        "order_date": dates,

        "price": prices,

        "freight_value": freight

    })

    return orders


# ============================================================
# OLIST LOADER
# ============================================================

def load_olist_data():

    customers_file = (
        DATA_DIR /
        "olist_customers_dataset.csv"
    )

    orders_file = (
        DATA_DIR /
        "olist_orders_dataset.csv"
    )

    items_file = (
        DATA_DIR /
        "olist_order_items_dataset.csv"
    )

    products_file = (
        DATA_DIR /
        "olist_products_dataset.csv"
    )

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required = [
        customers_file,
        orders_file,
        items_file
    ]

    if not all(
        file.exists()
        for file in required
    ):

        return None

    # --------------------------------------------------------
    # Load CSV files
    # --------------------------------------------------------

    customers = pd.read_csv(
        customers_file
    )

    orders = pd.read_csv(
        orders_file
    )

    items = pd.read_csv(
        items_file
    )

    # --------------------------------------------------------
    # Load products if available
    # --------------------------------------------------------

    products = None

    if products_file.exists():

        products = pd.read_csv(
            products_file
        )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    orders[
        "order_purchase_timestamp"
    ] = pd.to_datetime(
        orders[
            "order_purchase_timestamp"
        ],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Merge customers
    # --------------------------------------------------------

    merged = orders.merge(
        customers,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # Merge order items
    # --------------------------------------------------------

    item_columns = [
        column
        for column in [
            "order_id",
            "product_id",
            "price",
            "freight_value"
        ]
        if column in items.columns
    ]

    merged = merged.merge(
        items[item_columns],
        on="order_id",
        how="left"
    )

    # --------------------------------------------------------
    # Merge products
    # --------------------------------------------------------

    if products is not None:

        product_columns = [
            column
            for column in [
                "product_id",
                "product_category_name"
            ]
            if column in products.columns
        ]

        if product_columns:

            merged = merged.merge(
                products[
                    product_columns
                ],
                on="product_id",
                how="left"
            )

    # --------------------------------------------------------
    # Standardize columns
    # --------------------------------------------------------

    merged["order_date"] = pd.to_datetime(
        merged[
            "order_purchase_timestamp"
        ],
        errors="coerce"
    )

    merged["price"] = pd.to_numeric(
        merged["price"],
        errors="coerce"
    ).fillna(0)

    merged["freight_value"] = pd.to_numeric(
        merged["freight_value"],
        errors="coerce"
    ).fillna(0)

    return merged


# ============================================================
# MAIN DATA FUNCTION
# ============================================================

def load_data():

    data = load_olist_data()

    if data is not None:

        return (
            data,
            "Olist Brazilian E-Commerce Dataset"
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    demo_data = generate_demo_data()

    return (
        demo_data,
        "Synthetic Demonstration Dataset"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    df, source = load_data()

    print("=" * 60)

    print(
        "Data Source:",
        source
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )

    print("=" * 60)

    print(
        df.head()
    )