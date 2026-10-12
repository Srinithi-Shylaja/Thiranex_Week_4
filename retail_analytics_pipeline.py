import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display & visual preferences
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)

# ============================================================
# STEP 1: DATA INGESTION & SYNTHETIC DATA CREATION
# ============================================================
def load_data(filepath='online_retail.csv'):
    try:
        df = pd.read_csv(filepath, parse_dates=['InvoiceDate'])
        print(f"Loaded '{filepath}' with {df.shape[0]} rows.")
    except FileNotFoundError:
        print("File not found. Generating synthetic e-commerce transaction dataset...")
        np.random.seed(42)
        n_rows = 1000
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')
        categories = ['Electronics', 'Home & Kitchen', 'Apparel', 'Books', 'Beauty & Health']
        
        df = pd.DataFrame({
            'InvoiceNo': np.random.randint(100000, 105000, size=n_rows),
            'CustomerID': np.random.randint(1001, 1100, size=n_rows),
            'InvoiceDate': np.random.choice(dates, size=n_rows),
            'Category': np.random.choice(categories, size=n_rows, p=[0.25, 0.25, 0.20, 0.15, 0.15]),
            'Quantity': np.random.randint(1, 10, size=n_rows),
            'UnitPrice': np.round(np.random.uniform(5.0, 150.0, size=n_rows), 2)
        })
        df.to_csv('online_retail.csv', index=False)
        print("Sample dataset created and saved as 'online_retail.csv'.")
    return df

# ============================================================
# STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================
df = load_data()

# Filter out non-positive quantities or prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

# Calculate total sales revenue
df['Total_Sales'] = df['Quantity'] * df['UnitPrice']

# Extract temporal features for seasonality analysis
df['YearMonth'] = pd.to_datetime(df['InvoiceDate']).dt.to_period('M')
df['Month'] = pd.to_datetime(df['InvoiceDate']).dt.month_name()
df['DayOfWeek'] = pd.to_datetime(df['InvoiceDate']).dt.day_name()

print("\n=== CLEANED DATA SAMPLE ===")
print(df[['InvoiceNo', 'CustomerID', 'Category', 'Quantity', 'UnitPrice', 'Total_Sales', 'YearMonth']].head())

# ============================================================
# STEP 3: MONTHLY SALES PERFORMANCE & SEASONALITY ANALYSIS
# ============================================================
monthly_sales = df.groupby('YearMonth')['Total_Sales'].sum().reset_index()
monthly_sales['YearMonth_Str'] = monthly_sales['YearMonth'].astype(str)

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['YearMonth_Str'], monthly_sales['Total_Sales'], marker='o', color='teal', linewidth=2.5)
plt.title('Monthly Sales Revenue Trend (2023)', fontsize=14, fontweight='bold')
plt.xlabel('Year-Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('monthly_sales_trend.png', dpi=150)
plt.close()

# ============================================================
# STEP 4: CATEGORY REVENUE SHARE ANALYSIS
# ============================================================
category_sales = df.groupby('Category')['Total_Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(9, 5))
category_sales.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Total Revenue by Product Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('category_revenue_share.png', dpi=150)
plt.close()

# ============================================================
# STEP 5: CUSTOMER SPENDING SEGMENTATION (RFM QUARTILES)
# ============================================================
customer_rfm = df.groupby('CustomerID').agg(
    Total_Spent=('Total_Sales', 'sum'),
    Order_Count=('InvoiceNo', 'nunique'),
    Avg_Order_Value=('Total_Sales', 'mean')
).reset_index()

# Categorize spending tiers into Quartiles
spending_labels = ['Low Spender', 'Mid Spender', 'High Spender', 'VIP Spender']
customer_rfm['Customer_Segment'] = pd.qcut(customer_rfm['Total_Spent'], q=4, labels=spending_labels)

print("\n=== CUSTOMER SEGMENTATION SUMMARY ===")
segment_summary = customer_rfm.groupby('Customer_Segment', observed=False).agg(
    Customer_Count=('CustomerID', 'count'),
    Avg_Total_Spent=('Total_Spent', 'mean'),
    Avg_Orders=('Order_Count', 'mean')
)
print(segment_summary)

plt.figure(figsize=(8, 5))
sns.countplot(x='Customer_Segment', hue='Customer_Segment', data=customer_rfm, palette='viridis', legend=False)
plt.title('Customer Count by Spending Tier', fontsize=14, fontweight='bold')
plt.xlabel('Customer Segment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('customer_segmentation.png', dpi=150)
plt.close()

print("\nPipeline execution complete! Saved figures: 'monthly_sales_trend.png', 'category_revenue_share.png', 'customer_segmentation.png'.")
