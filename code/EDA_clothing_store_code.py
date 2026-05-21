import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)

# --- 1. DATA LOADING AND CLEANING ---

# Keep original file location as requested
data = pd.read_csv(
    "C:/Users/krzys/OneDrive/Pulpit/DM Project/Dane_clothing.txt"
)

new_column_names = {
    "HHKEY": "Customer_ID",
    "ZIP_CODE": "Zip_Code",
    "FRE": "Visit_Frequency",
    "MON": "Total_Net_Spend",
    "AVRG": "Avg_Spend_Per_Visit",
    "DAYS": "Days_In_Database",
    "FREDAYS": "Days_Between_Purchases",
    "LTFREDAY": "Avg_Days_Between_Purchases",
    "PERCRET": "Return_Percentage",
    "HI": "Purchase_Diversity_Index",
    "CLASSES": "Product_Category_Count",
    "STYLES": "Items_Purchased_Count",
    "STORES": "Stores_Visited_Count",
    "OMONSPEND": "Spend_Last_Month",
    "TMONSPEND": "Spend_Last_3_Months",
    "SMONSPEND": "Spend_Last_6_Months",
    "PREVPD": "Spend_Year_Ago",
    "AMSPEND": "Spend_Franchise_A",
    "PSSPEND": "Spend_Franchise_PS",
    "CCSPEND": "Spend_Franchise_CC",
    "AXSPEND": "Spend_Franchise_AX",
    "PSWEATERS": "Pct_Sweaters",
    "PKNIT_TOPS": "Pct_Knit_Tops",
    "PKNIT_DRES": "Pct_Knit_Dresses",
    "PBLOUSES": "Pct_Blouses",
    "PJACKETS": "Pct_Jackets",
    "PCAR_PNTS": "Pct_Dress_Pants",
    "PCAS_PNTS": "Pct_Casual_Pants",
    "PSHIRTS": "Pct_Shirts",
    "PDRESSES": "Pct_Dresses",
    "PSUITS": "Pct_Suits",
    "POUTERWEAR": "Pct_Outerwear",
    "PJEWELRY": "Pct_Jewelry",
    "PFASHION": "Pct_Fashion",
    "PLEGWEAR": "Pct_Legwear_Socks",
    "PCOLLSPND": "Pct_Collectibles",
    "GMP": "Gross_Margin_Percentage",
    "PROMOS": "Promotions_Received_Count",
    "COUPONS": "Coupons_Used_Count",
    "MARKDOWN": "Markdown_Purchase_Percentage",
    "MAILED": "Promos_Mailed_Last_Year",
    "RESPONDED": "Promos_Responded_Last_Year",
    "RESPONSERATE": "Response_Rate_Percentage",
    "CLUSTYPE": "Lifestyle_Cluster_Type",
    "CC_CARD": "Flag_Credit_Card",
    "VALPHON": "Flag_Valid_Phone",
    "WEB": "Flag_Online_Buyer",
    "REC": "TO_DROP_1",
    "PC_CALC20": "TO_DROP_2",
    "STORELOY": "TO_DROP_3",
    "RESP": "TARGET_Promo_Response",
}

data = data.rename(columns=new_column_names)

data = data.drop(columns=["TO_DROP_1", "TO_DROP_2", "TO_DROP_3"])

data["Flag_Valid_Phone"] = (
    data["Flag_Valid_Phone"].map({"Y": 1, "N": 0}).astype(int)
)


# --- 2. MISSING VALUES ANALYSIS ---

na_counts = data.isna().sum().reset_index()
na_counts.columns = ["Column", "Missing_Count"]

half = len(na_counts) // 2

left = na_counts.iloc[:half].reset_index(drop=True)
right = na_counts.iloc[half:].reset_index(drop=True)

summary_table = pd.concat([left, right], axis=1)
summary_table.style.hide(axis="index")

data.info()


# --- 3. VARIABLE SEGREGATION ---

identifiers = [
    "Customer_ID",
    "Zip_Code",
    "Lifestyle_Cluster_Type",
    "TARGET_Promo_Response",
]

percentage_columns = [col for col in data.columns if col.startswith("Pct_")]

continuous_columns = [
    col
    for col in data.columns
    if not col.startswith("Pct_")
    and not col.startswith("Flag_")
    and col not in identifiers
    and data[col].dtype in ["int64", "float64"]
]

data[continuous_columns].describe().round(2).T


# --- 4. EXPLORATORY DATA ANALYSIS (EDA) ---

to_exclude = [
    "Spend_Franchise_A",
    "Spend_Franchise_PS",
    "Spend_Franchise_AX",
    "Spend_Last_3_Months",
    "Spend_Last_Month",
    "Spend_Year_Ago",
    "Coupons_Used_Count",
    "Promos_Responded_Last_Year",
    "Response_Rate_Percentage",
    "Return_Percentage",
    "Markdown_Purchase_Percentage",
    "Spend_Last_6_Months",
]

hist_columns = [col for col in continuous_columns if col not in to_exclude]

n_cols = 2
n_rows = math.ceil(len(hist_columns) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(hist_columns):
    lower_bound = data[col].quantile(0.01)
    upper_bound = data[col].quantile(0.95)

    plot_data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    sns.histplot(data=plot_data, x=col, kde=True, bins=30, ax=axes[i])

    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


box_columns = [col for col in continuous_columns if col not in to_exclude]

n_cols = 3
n_rows = math.ceil(len(box_columns) / n_cols)

# Creating a grid of plots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2.5))
axes = axes.flatten()

for i, col in enumerate(box_columns):
    sns.boxplot(x=data[col], ax=axes[i])
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# --- 5. CORRELATION ANALYSIS ---

data["Lifestyle_Cluster_Type"] = (
    data.groupby("Lifestyle_Cluster_Type")["TARGET_Promo_Response"]
    .transform("mean")
    .round(4)
)

numeric_data = data.select_dtypes(include="number")
target = "TARGET_Promo_Response"

top_features_corr = (
    numeric_data.corr()[target].abs().sort_values(ascending=False).head(11).index
)
plt.figure(figsize=(4, 3))
sns.heatmap(numeric_data[top_features_corr].corr(), cmap="coolwarm", center=0)
plt.show()


# --- 6. MODEL PREPARATION AND FEATURE SELECTION ---

model_data = data.drop(columns=["Customer_ID", "Zip_Code"])
X = model_data.drop("TARGET_Promo_Response", axis=1)
y = model_data["TARGET_Promo_Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(X_train, y_train)

importances = pd.Series(rf_selector.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10).index.tolist()

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]


# --- 7. HYPERPARAMETER TUNING ---

rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}

xgb_params = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
)
grid_rf.fit(X_train_top, y_train)

grid_xgb = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    xgb_params,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
)
grid_xgb.fit(X_train_top, y_train)


# --- 8. MODEL EVALUATION ---

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), X_train_s, X_test_s),
    "LDA": (LinearDiscriminantAnalysis(), X_train_s, X_test_s),
    "Random Forest": (RandomForestClassifier(random_state=42), X_train, X_test),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        X_train,
        X_test,
    ),
    "RF (Top 10 + Tuned)": (grid_rf.best_estimator_, X_train_top, X_test_top),
    "XGBoost (Top 10 + Tuned)": (
        grid_xgb.best_estimator_,
        X_train_top,
        X_test_top,
    ),
}

results = []

for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)

    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.3).astype(int)

    results.append(
        [
            name,
            roc_auc_score(y_test, prob),
            accuracy_score(y_test, pred),
            precision_score(y_test, pred),
            recall_score(y_test, pred),
            f1_score(y_test, pred),
        ]
    )

pd.DataFrame(
    results,
    columns=["Model", "ROC AUC", "Accuracy", "Precision", "Recall", "F1"],
).sort_values("ROC AUC", ascending=False).style.hide(axis="index")
