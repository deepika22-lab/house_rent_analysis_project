from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay
)

# Place House_Rent_Dataset.csv in the data/ folder.
DATA_PATH = Path(__file__).resolve().parent / "data" / "House_Rent_Dataset.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Create a 'data' folder and place House_Rent_Dataset.csv inside it."
    )

raw = pd.read_csv(DATA_PATH)
raw.columns = raw.columns.str.strip().str.replace(" ", "_")

print("Original shape:", raw.shape)
raw.info()
print("Missing values:", raw.isna().sum().sum())
print("Duplicate rows:", raw.duplicated().sum())
print(raw[["Rent", "Size", "BHK", "Bathroom"]].describe())
print("Rent skewness:", raw["Rent"].skew())

df = raw.drop_duplicates().copy()

LEVELS = {"Ground": 0, "Upper Basement": -1, "Lower Basement": -2}


def parse_floor(value):
    parts = str(value).split(" out of ")
    floor_text = parts[0].strip()
    floor_no = LEVELS.get(floor_text, pd.to_numeric(floor_text, errors="coerce"))
    total_floors = (
        pd.to_numeric(parts[1], errors="coerce") if len(parts) > 1 else np.nan
    )
    return pd.Series([floor_no, total_floors])


df[["Floor_No", "Total_Floors"]] = df["Floor"].apply(parse_floor)
df["Total_Floors"] = df["Total_Floors"].fillna(df["Floor_No"])
df["Total_Floors"] = np.maximum(df["Total_Floors"], df["Floor_No"])
df["Floor_No"] = df["Floor_No"].fillna(df["Floor_No"].median())

df = df.drop(columns=["Floor", "Area_Locality", "Posted_On"])
df = df[
    (df["Area_Type"] != "Built Area")
    & (df["Point_of_Contact"] != "Contact Builder")
]
df = df[(df["Size"] >= 100) & (df["Size"] <= 4000)]

df["Log_Rent"] = np.log(df["Rent"])
q1, q3 = df["Log_Rent"].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df = df[
    df["Log_Rent"].between(lower_bound, upper_bound)
].reset_index(drop=True)

print("Cleaned shape:", df.shape)

city_median = df.groupby("City")["Rent"].transform("median")
df["High_Rent"] = (df["Rent"] > city_median).astype(int)

# Visualize rent distributions.
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(raw["Rent"], bins=60, ax=axes[0])
axes[0].set_title("Raw Rent")
sns.histplot(df["Rent"], bins=50, ax=axes[1])
axes[1].set_title("Rent After Outlier Filtering")
sns.histplot(df["Log_Rent"], bins=50, kde=True, ax=axes[2])
axes[2].set_title("Log-Transformed Rent")
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x="City", y="Rent")
plt.yscale("log")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x="Size", y="Rent", hue="City", alpha=0.5)
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x="BHK", y="Rent")
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x="Furnishing_Status", y="Rent")
plt.tight_layout()
plt.show()

num_cols = ["BHK", "Size", "Bathroom", "Floor_No", "Total_Floors"]
cat_cols = [
    "City", "Furnishing_Status", "Tenant_Preferred",
    "Area_Type", "Point_of_Contact"
]

sns.heatmap(
    df[num_cols + ["Rent", "Log_Rent"]].corr(),
    annot=True, fmt=".2f", cmap="coolwarm"
)
plt.tight_layout()
plt.show()

for column in num_cols:
    pearson_r, pearson_p = stats.pearsonr(df[column], df["Log_Rent"])
    spearman_rho, spearman_p = stats.spearmanr(df[column], df["Log_Rent"])
    print(
        f"{column:13s} Pearson r={pearson_r:.3f} (p={pearson_p:.1e}) | "
        f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.1e})"
    )

groups = [group["Log_Rent"].values for _, group in df.groupby("City")]
print("ANOVA:", stats.f_oneway(*groups))
print("Kruskal-Wallis:", stats.kruskal(*groups))

furnished = df.loc[
    df["Furnishing_Status"] == "Furnished", "Log_Rent"
]
unfurnished = df.loc[
    df["Furnishing_Status"] == "Unfurnished", "Log_Rent"
]
print("Welch's t-test:", stats.ttest_ind(furnished, unfurnished, equal_var=False))

for column in ["Furnishing_Status", "Tenant_Preferred", "Area_Type", "Point_of_Contact"]:
    contingency_table = pd.crosstab(df[column], df["High_Rent"])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)
    print(f"{column}: chi2={chi2:.1f}, p={p_value:.3g}, dof={dof}")

X = df[num_cols + cat_cols]
y = df["High_Rent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def make_preprocessor():
    numeric_pipeline = Pipeline([("scale", StandardScaler())])
    categorical_pipeline = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )
    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])


def make_pipeline(model):
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

knn_search = GridSearchCV(
    make_pipeline(KNeighborsClassifier()),
    {"model__n_neighbors": range(1, 32, 2)},
    cv=cv, scoring="f1", n_jobs=-1
).fit(X_train, y_train)

tree_search = GridSearchCV(
    make_pipeline(DecisionTreeClassifier(random_state=42)),
    {
        "model__max_depth": [2, 3, 4, 5, 6, 8, 10, 12],
        "model__min_samples_leaf": [1, 5, 10, 20],
    },
    cv=cv, scoring="f1", n_jobs=-1
).fit(X_train, y_train)

svm_search = GridSearchCV(
    make_pipeline(SVC(probability=True, random_state=42)),
    {"model__C": [0.5, 1, 5, 10], "model__gamma": ["scale", 0.05]},
    cv=cv, scoring="f1", n_jobs=-1
).fit(X_train, y_train)

models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000)
    ).fit(X_train, y_train),
    "KNN": knn_search.best_estimator_,
    "Naive Bayes": make_pipeline(GaussianNB()).fit(X_train, y_train),
    "Decision Tree": tree_search.best_estimator_,
    "SVM": svm_search.best_estimator_,
}

results = []
for name, model in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "AUC": roc_auc_score(y_test, probabilities),
        "LogLoss": log_loss(y_test, probabilities),
    })

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title(name)
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\nModel comparison:")
print(results_df.round(3).to_string(index=False))
print("\nBest parameters:")
print("KNN:", knn_search.best_params_)
print("Decision Tree:", tree_search.best_params_)
print("SVM:", svm_search.best_params_)
