import pandas as pd
import matplotlib.pyplot as plt

wti = pd.read_csv('wti.csv')

wti = wti.rename(columns={
    'observation_date': 'Date',
    'DCOILWTICO': 'WTI'})

wti['Date'] = pd.to_datetime(wti['Date'])

wti_weekly = (
    wti.set_index('Date')
    .resample('W-Fri')
    .last()
    .reset_index()
)

wti_weekly['WTI_return'] = wti_weekly['WTI'].pct_change() * 100
print(wti_weekly.head(10))


inventories = pd.read_csv("inventories.csv", skiprows = 4)

inventories = inventories.rename(columns = {
    inventories.columns[0]: 'Date',
    inventories.columns[1]: 'Inventories'
})


inventories['Date'] = pd.to_datetime(inventories['Date'])
inventories = inventories.sort_values('Date')

inventories = inventories[inventories["Date"] >= "2018-01-01"]

inventories["Inventory_change"] = inventories['Inventories'].diff()
inventories["Inventory_change_mb"] = inventories["Inventory_change"]/1000

inventories = inventories.reset_index(drop=True)


print(inventories.head())
print(inventories.columns)


data = pd.merge(
    wti_weekly,
    inventories,
    on = "Date",
    how = "inner"
)

print(data.head())

# plt.scatter(
#     data["Inventory_change_mb"],
#     data["WTI_return"]
# )

# plt.xlabel("Weekly Inventory Change (million barrels)")
# plt.ylabel("Weekly WTI Return (%)")
# plt.title("Crude Oil Inventory Changes vs WTI Returns")

# plt.axhline(0)
# plt.axvline(0)

# plt.show()

production = pd.read_excel(
    "production.xls",
    sheet_name="Data 1",
    header=2
)

production = production.rename(columns={
    production.columns[0]: "Date",
    production.columns[1]: "Production"
})

production["Date"] = pd.to_datetime(production["Date"])

production = production.sort_values("Date")

production = production[
    production["Date"] >= "2018-01-01"
]

production = production.reset_index(drop=True)

print(production.head())

data = pd.merge(
    data,
    production,
    on="Date"
)

refinery = pd.read_excel(
    "refinery_inputs.xls",
    sheet_name="Data 1",
    header=2
)

refinery = refinery[[
    "Date",
    "Weekly U.S. Refiner Net Input of Crude Oil  (Thousand Barrels per Day)"
]]

refinery = refinery.rename(columns={
    "Weekly U.S. Refiner Net Input of Crude Oil  (Thousand Barrels per Day)": "Refinery_inputs"
})

refinery["Date"] = pd.to_datetime(refinery["Date"])

refinery = refinery.sort_values("Date")

refinery = refinery[
    refinery["Date"] >= "2018-01-01"
]

refinery = refinery.reset_index(drop=True)

print(refinery.head())

data = pd.merge(
    data,
    refinery,
    on="Date"
)

print(data.head())

imports = pd.read_excel(
    "imports.xls",
    sheet_name="Data 1",
    header=2
)
imports = imports.iloc[:, [0, 1]]

imports = imports.rename(columns={
    imports.columns[0]: "Date",
    imports.columns[1]: "Imports"
})

imports["Date"] = pd.to_datetime(imports["Date"])

imports = imports.sort_values("Date")

imports = imports[
    imports["Date"] >= "2018-01-01"
]

imports = imports.reset_index(drop=True)

data = pd.merge(
    data,
    imports,
    on="Date"
)

exports = pd.read_excel(
    "exports.xls",
    sheet_name="Data 2",
    header=2
)

exports = exports[[
    "Date",
    "Weekly U.S. Exports of Crude Oil  (Thousand Barrels per Day)"
]]

exports = exports.rename(columns={
    "Weekly U.S. Exports of Crude Oil  (Thousand Barrels per Day)": "Exports"
})

exports["Date"] = pd.to_datetime(exports["Date"])

exports = exports.sort_values("Date")

exports = exports[
    exports["Date"] >= "2018-01-01"
]

exports = exports.reset_index(drop=True)

data = pd.merge(
    data,
    exports,
    on="Date"
)

print(data.head())

data["Balance_kbd"] = (
    data["Production"]
    + data["Imports"]
    - data["Refinery_inputs"]
    - data["Exports"]
)

data["Implied_inventory_change_mb"] = (
    data["Balance_kbd"] * 7 / 1000
)

data["Balance_residual_mb"] = (
    data["Inventory_change_mb"]
    - data["Implied_inventory_change_mb"]
)

print(
    data[[
        "Date",
        "Production",
        "Imports",
        "Refinery_inputs",
        "Exports",
        "Implied_inventory_change_mb",
        "Inventory_change_mb",
        "Balance_residual_mb"
    ]].head(10)
)

# Convert the unexplained weekly balance difference
# back into thousand barrels per day
data["Unexplained_balance_kbd"] = (
    data["Balance_residual_mb"] * 1000 / 7
)

# Compare the simple model with the actual inventory change
comparison = data[
    [
        "Date",
        "Implied_inventory_change_mb",
        "Inventory_change_mb",
        "Balance_residual_mb",
        "Unexplained_balance_kbd"
    ]
].dropna()

print(comparison.head(10))

# Some basic model-validation statistics
mae = comparison["Balance_residual_mb"].abs().mean()

correlation = comparison[
    ["Implied_inventory_change_mb", "Inventory_change_mb"]
].corr().iloc[0, 1]


data["Balance_4w"] = data["Balance_kbd"].rolling(4).mean()

data["Inventory_change_4w"] = (
    data["Inventory_change_mb"].rolling(4).mean()
)

data["WTI_forward_4w"] = (
    data["WTI"].shift(-4) / data["WTI"] - 1
) * 100

analysis = data.dropna(
    subset=["Balance_4w", "WTI_forward_4w"]
).copy()

analysis["Regime"] = pd.qcut(
    analysis["Balance_4w"],
    q=3,
    labels=["Tight", "Neutral", "Loose"]
)

print(
    analysis[
        ["Date", "Balance_4w", "Regime", "WTI_forward_4w"]
    ].head(15)
)

regime_summary = analysis.groupby(
    "Regime",
    observed=True
)["WTI_forward_4w"].agg(
    ["count", "mean", "median"]
)

positive_pct = analysis.groupby(
    "Regime",
    observed=True
)["WTI_forward_4w"].apply(
    lambda x: (x > 0).mean() * 100
)

regime_summary["positive_return_pct"] = positive_pct

print(regime_summary)


# Make sure everything is in date order
data = data.sort_values("Date").reset_index(drop=True)

# Week number: 1 to 52/53
data["Week_of_year"] = (
    data["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Average inventory level for this same week
# over the previous 5 years
data["Seasonal_inventory_mean"] = (
    data.groupby("Week_of_year")["Inventories"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)

# Difference from seasonal norm, in million barrels
data["Inventory_vs_seasonal_mb"] = (
    data["Inventories"]
    - data["Seasonal_inventory_mean"]
) / 1000

print(
    data[[
        "Date",
        "Inventories",
        "Seasonal_inventory_mean",
        "Inventory_vs_seasonal_mb"
    ]].tail(20)
)

seasonal = data.dropna(
    subset=["Inventory_vs_seasonal_mb"]
)


data["Seasonal_position_change_4w"] = (
    data["Inventory_vs_seasonal_mb"]
    - data["Inventory_vs_seasonal_mb"].shift(4)
)

latest = data.dropna(
    subset=[
        "Balance_4w",
        "Inventory_vs_seasonal_mb",
        "Seasonal_position_change_4w"
    ]
).iloc[-1]

# Historical percentile of the 4-week crude balance
data["Balance_4w_percentile"] = (
    data["Balance_4w"].rank(pct=True) * 100
)

# Historical percentile of seasonal inventory position
valid_seasonal = data["Inventory_vs_seasonal_mb"].notna()

data.loc[
    valid_seasonal,
    "Seasonal_inventory_percentile"
] = (
    data.loc[
        valid_seasonal,
        "Inventory_vs_seasonal_mb"
    ].rank(pct=True) * 100
)

# Net crude imports
data["Net_imports"] = (
    data["Imports"] - data["Exports"]
)

# 4-week averages of the main balance components
data["Production_4w"] = (
    data["Production"].rolling(4).mean()
)

data["Refinery_inputs_4w"] = (
    data["Refinery_inputs"].rolling(4).mean()
)

data["Net_imports_4w"] = (
    data["Net_imports"].rolling(4).mean()
)

data["Production_change_4w"] = (
    data["Production_4w"]
    - data["Production_4w"].shift(4)
)

data["Refinery_change_4w"] = (
    data["Refinery_inputs_4w"]
    - data["Refinery_inputs_4w"].shift(4)
)

data["Net_imports_change_4w"] = (
    data["Net_imports_4w"]
    - data["Net_imports_4w"].shift(4)
)

latest = data.dropna(
    subset=[
        "Balance_4w_percentile",
        "Seasonal_inventory_percentile"
    ]
).iloc[-1]

print("\nHISTORICAL CONTEXT")
print("------------------")

print(
    "4-week balance percentile:",
    round(latest["Balance_4w_percentile"], 1),
    "%"
)

print(
    "Seasonal inventory percentile:",
    round(latest["Seasonal_inventory_percentile"], 1),
    "%"
)

print("\nCURRENT U.S. CRUDE MARKET SNAPSHOT")
print("-----------------------------------")

print("Date:", latest["Date"].date())

print(
    "WTI:",
    round(latest["WTI"], 2),
    "USD/barrel"
)

print(
    "Commercial inventories:",
    round(latest["Inventories"] / 1000, 1),
    "million barrels"
)

print(
    "Inventories vs 5-year seasonal average:",
    round(latest["Inventory_vs_seasonal_mb"], 1),
    "million barrels"
)

print(
    "4-week change in seasonal position:",
    round(latest["Seasonal_position_change_4w"], 1),
    "million barrels"
)

print(
    "4-week average crude balance:",
    round(latest["Balance_4w"], 0),
    "thousand barrels/day"
)

if latest["Inventory_vs_seasonal_mb"] < 0:
    inventory_status = "Below seasonal average"
else:
    inventory_status = "Above seasonal average"

if latest["Seasonal_position_change_4w"] < 0:
    inventory_momentum = "Tightening"
else:
    inventory_momentum = "Loosening"

print("Inventory position:", inventory_status)
print("4-week inventory momentum:", inventory_momentum)

latest = data.dropna(
    subset=[
        "Balance_4w",
        "Inventory_vs_seasonal_mb",
        "Seasonal_position_change_4w",
        "Production_change_4w",
        "Refinery_change_4w",
        "Net_imports_change_4w"
    ]
).iloc[-1]

print("\nBALANCE DRIVERS — CHANGE VS 4 WEEKS AGO")
print("---------------------------------------")

print(
    "Production:",
    round(latest["Production_change_4w"], 0),
    "kbd"
)

print(
    "Refinery inputs:",
    round(latest["Refinery_change_4w"], 0),
    "kbd"
)

print(
    "Net imports:",
    round(latest["Net_imports_change_4w"], 0),
    "kbd"
)

data["Imports_4w"] = data["Imports"].rolling(4).mean()
data["Exports_4w"] = data["Exports"].rolling(4).mean()

data["Imports_change_4w"] = (
    data["Imports_4w"]
    - data["Imports_4w"].shift(4)
)

data["Exports_change_4w"] = (
    data["Exports_4w"]
    - data["Exports_4w"].shift(4)
)

latest = data.dropna(
    subset=[
        "Imports_change_4w",
        "Exports_change_4w"
    ]
).iloc[-1]

print(
    "Imports:",
    round(latest["Imports_change_4w"], 0),
    "kbd"
)

print(
    "Exports:",
    round(latest["Exports_change_4w"], 0),
    "kbd"
)
#plots

driver_names = [
    "Production",
    "Imports",
    "Refinery Inputs",
    "Exports"
]

driver_contributions = [
    latest["Production_change_4w"],
    latest["Imports_change_4w"],
    -latest["Refinery_change_4w"],
    -latest["Exports_change_4w"]
]

print("Mean absolute error:", mae, "million barrels")
print("Correlation:", correlation)

comparison["Correct_direction"] = (
    (
        (comparison["Implied_inventory_change_mb"] > 0)
        & (comparison["Inventory_change_mb"] > 0)
    )
    |
    (
        (comparison["Implied_inventory_change_mb"] < 0)
        & (comparison["Inventory_change_mb"] < 0)
    )
)

direction_accuracy = (
    comparison["Correct_direction"].mean() * 100
)

print(
    "Build/draw directional accuracy:",
    round(direction_accuracy, 1),
    "%"
)

r_squared = correlation ** 2
print(round(r_squared, 3))

# Was the actual week a build or a draw?
actual_build = comparison["Inventory_change_mb"] > 0

# Percentage of weeks that were builds/draws
build_pct = actual_build.mean() * 100
draw_pct = 100 - build_pct

# Dumb model: always predict whichever outcome occurs most often
naive_accuracy = max(build_pct, draw_pct)

print(
    "Actual build weeks:",
    round(build_pct, 1),
    "%"
)

print(
    "Actual draw weeks:",
    round(draw_pct, 1),
    "%"
)

print(
    "Naive baseline accuracy:",
    round(naive_accuracy, 1),
    "%"
)

print(
    "Model improvement over baseline:",
    round(direction_accuracy - naive_accuracy, 1),
    "percent"
)

actual_build = comparison["Inventory_change_mb"] > 0
model_build = comparison["Implied_inventory_change_mb"] > 0

build_accuracy = (
    (model_build[actual_build] == True).mean() * 100
)

draw_accuracy = (
    (model_build[~actual_build] == False).mean() * 100
)

print(
    "Accuracy when stocks actually build:",
    round(build_accuracy, 1),
    "%"
)

print(
    "Accuracy when stocks actually draw:",
    round(draw_accuracy, 1),
    "%"
)
comparison["Actual_direction"] = comparison[
    "Inventory_change_mb"
].apply(
    lambda x: "Build" if x > 0 else "Draw"
)

comparison["Model_direction"] = comparison[
    "Implied_inventory_change_mb"
].apply(
    lambda x: "Build" if x > 0 else "Draw"
)

direction_table = pd.crosstab(
    comparison["Actual_direction"],
    comparison["Model_direction"]
)

print(direction_table)

plt.figure(figsize=(8, 5))

plt.bar(
    driver_names,
    driver_contributions
)

plt.axhline(0)

plt.ylabel("Contribution to Balance Change (kbd)")
plt.title("Drivers of Change in U.S. Crude Balance — Last 4 Weeks")

plt.xticks(rotation=20)

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(
    seasonal["Date"],
    seasonal["Inventory_vs_seasonal_mb"]
)

plt.axhline(0)

plt.xlabel("Date")
plt.ylabel("Inventory vs Seasonal Average (million barrels)")
plt.title(
    "U.S. Commercial Crude Inventories vs 5-Year Seasonal Average"
)

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(
    comparison["Date"],
    comparison["Inventory_change_mb"],
    label="Actual inventory change"
)

plt.plot(
    comparison["Date"],
    comparison["Implied_inventory_change_mb"],
    label="Simple balance implied change"
)

plt.xlabel("Date")
plt.ylabel("Weekly inventory change (million barrels)")
plt.title("Actual vs Implied U.S. Crude Inventory Changes")
plt.legend()

plt.show()

# plt.plot(wti_weekly['Date'], wti_weekly['WTI'])
# plt.title('WTI Crude Oil Prices Over Time')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.show()
