import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def simulate_imported_drugs(N=10000):
    """
    Simulates N imported drug shipments. Any shipment that is seized, stolen,
    or arrives later than 7 days is counted in final_missing.
    """

    # 1) Random parameters
    p_counterfeit = np.random.uniform(0.1, 0.2)

    p_seized_export_counterfeit = np.random.uniform(0.0001, 0.001)
    p_seized_export_legit = np.random.beta(3, 10000)

    p_seized_import_counterfeit = np.random.uniform(0.001, 0.005)
    p_seized_import_legit = np.random.beta(3, 10000)

    p_warehouse_theft = np.random.uniform(0.0000014, 0.00001)
    p_last_mile_theft = np.random.uniform(0.0001, 0.001)

    base_mean = 4.74
    base_var = 0.785
    base_std = np.sqrt(base_var)

    # 2) Initialize counters
    legit_total = 0
    counterfeit_total = 0

    final_legit_on_time = 0
    final_counterfeit_on_time = 0

    total_transit_time = 0.0
    arrived_shipments = 0

    # Unified counter for anything “not on time”
    final_missing = 0

    # 3) Simulate each shipment
    for _ in range(N):
        # A) Counterfeit or legit?
        is_counterfeit = (np.random.random() < p_counterfeit)
        if is_counterfeit:
            counterfeit_total += 1
        else:
            legit_total += 1

        # B) Export Customs Seizure → count as missing and skip
        if is_counterfeit:
            if np.random.random() < p_seized_export_counterfeit:
                final_missing += 1
                continue
        else:
            if np.random.random() < p_seized_export_legit:
                final_missing += 1
                continue

        # C) Base Transport Time
        transit_time = np.random.normal(base_mean, base_std)

        # D) Import Customs Seizure → count as missing and skip
        if is_counterfeit:
            if np.random.random() < p_seized_import_counterfeit:
                final_missing += 1
                continue
        else:
            if np.random.random() < p_seized_import_legit:
                final_missing += 1
                continue

        # E) Extra Stops + Warehouse Theft
        lambda_stops = 3
        num_stops = max(1, np.random.poisson(lambda_stops))

        stolen = False
        for _stop in range(num_stops):
            extra_time = np.random.normal(0.5, np.sqrt(0.6))
            transit_time += extra_time

            # If stolen in warehouse, mark missing and break out
            if np.random.random() < p_warehouse_theft:
                final_missing += 1
                stolen = True
                break

        # If stolen, skip last-mile theft & arrival
        if stolen:
            continue

        # F) Last-Mile Theft
        if np.random.random() < p_last_mile_theft:
            final_missing += 1
            continue

        # Shipment arrives
        arrived_shipments += 1
        total_transit_time += transit_time

        # G) Check if on-time (< 7 days). Otherwise, mark as missing.
        if transit_time < 7:
            if is_counterfeit:
                final_counterfeit_on_time += 1
            else:
                final_legit_on_time += 1
        else:
            final_missing += 1  # late arrival => counted as missing

    # 4) Compile results
    avg_transit_time = (total_transit_time / arrived_shipments) if arrived_shipments > 0 else 0.0

    results = {
        "final_legit_on_time": final_legit_on_time,
        "final_counterfeit_on_time": final_counterfeit_on_time,
        "final_missing": final_missing,
        "final_legit_total": legit_total,
        "final_counterfeit_total": counterfeit_total,
        "total_surviving": final_legit_on_time + final_counterfeit_on_time,
        "avg_transit_time": avg_transit_time,

        # Probabilities
        "p_counterfeit": p_counterfeit,
        "p_seized_export_counterfeit": p_seized_export_counterfeit,
        "p_seized_export_legit": p_seized_export_legit,
        "p_seized_import_counterfeit": p_seized_import_counterfeit,
        "p_seized_import_legit": p_seized_import_legit,
        "p_warehouse_theft": p_warehouse_theft,
        "p_last_mile_theft": p_last_mile_theft
    }
    return results


def simulate_local_drugs(N=10000):
    """
    Similar approach for local shipments: any stolen or late shipment
    is marked in final_missing.
    """
    p_counterfeit_local = np.random.uniform(0.1, 0.25)
    p_warehouse_theft_local = np.random.uniform(0.0000014, 0.00001)
    p_last_mile_theft_local = np.random.uniform(0.0001, 0.001)

    final_legit_on_time = 0
    final_counterfeit_on_time = 0

    legit_total = 0
    counterfeit_total = 0

    total_transit_time = 0.0
    arrived_shipments = 0

    final_missing = 0

    for _ in range(N):
        # A) Is it counterfeit?
        is_counterfeit = (np.random.random() < p_counterfeit_local)
        if is_counterfeit:
            counterfeit_total += 1
        else:
            legit_total += 1

        # B) For local: no export/import
        transit_time = 0.0
        lambda_stops = 1
        num_stops = max(1, np.random.poisson(lambda_stops))

        stolen = False
        for _stop in range(num_stops):
            extra_time = np.random.normal(0.5, np.sqrt(0.6))
            transit_time += extra_time

            # Warehouse theft => mark missing and break
            if np.random.random() < p_warehouse_theft_local:
                final_missing += 1
                stolen = True
                break

        if stolen:
            continue

        # Last-mile theft => mark missing
        if np.random.random() < p_last_mile_theft_local:
            final_missing += 1
            continue

        # Shipment arrives
        arrived_shipments += 1
        total_transit_time += transit_time

        # Check if on-time
        if transit_time < 7:
            if is_counterfeit:
                final_counterfeit_on_time += 1
            else:
                final_legit_on_time += 1
        else:
            final_missing += 1  # late => missing

    # Results
    avg_transit_time = (total_transit_time / arrived_shipments) if arrived_shipments > 0 else 0.0

    results = {
        "final_legit_on_time": final_legit_on_time,
        "final_counterfeit_on_time": final_counterfeit_on_time,
        "final_missing": final_missing,
        "final_legit_total": legit_total,
        "final_counterfeit_total": counterfeit_total,
        "total_surviving": final_legit_on_time + final_counterfeit_on_time,
        "avg_transit_time": avg_transit_time,

        "p_counterfeit_local": p_counterfeit_local,
        "p_warehouse_theft_local": p_warehouse_theft_local,
        "p_last_mile_theft_local": p_last_mile_theft_local
    }
    return results


def main_simulation(num_sims=1000, N=10000):
    """
    Runs multiple Monte Carlo simulations. Only final_missing is tracked
    as "not on time" (late, stolen, or seized).
    """
    imported_data = []
    local_data = []

    for _ in range(num_sims):
        imported_res = simulate_imported_drugs(N)
        local_res = simulate_local_drugs(N)

        # fraction of shipments that ended up missing
        missing_fraction_imported = imported_res["final_missing"] / N
        missing_fraction_local = local_res["final_missing"] / N

        # Collect data for imported
        imported_data.append({
            # X features
            "p_counterfeit": imported_res["p_counterfeit"],
            "p_seized_export_counterfeit": imported_res["p_seized_export_counterfeit"],
            "p_seized_export_legit": imported_res["p_seized_export_legit"],
            "p_seized_import_counterfeit": imported_res["p_seized_import_counterfeit"],
            "p_seized_import_legit": imported_res["p_seized_import_legit"],
            "p_warehouse_theft": imported_res["p_warehouse_theft"],
            "p_last_mile_theft": imported_res["p_last_mile_theft"],
            "avg_transit_time": imported_res["avg_transit_time"],

            # Y metric
            "missing_frac": missing_fraction_imported,
        })

        # Collect data for local
        local_data.append({
            "p_counterfeit_local": local_res["p_counterfeit_local"],
            "p_warehouse_theft_local": local_res["p_warehouse_theft_local"],
            "p_last_mile_theft_local": local_res["p_last_mile_theft_local"],
            "avg_transit_time_local": local_res["avg_transit_time"],

            "missing_frac": missing_fraction_local,
        })

    # Build DataFrames
    imported_df = pd.DataFrame(imported_data)
    local_df = pd.DataFrame(local_data)
    return imported_df, local_df


def regression_analysis_imported(imported_df):
    """ Regress missing_frac on all probabilities & avg_transit_time for imported shipments. """
    X = imported_df[[
        "p_counterfeit",
        "p_seized_export_counterfeit",
        "p_seized_export_legit",
        "p_seized_import_counterfeit",
        "p_seized_import_legit",
        "p_warehouse_theft",
        "p_last_mile_theft",
        "avg_transit_time"
    ]]
    y = imported_df["missing_frac"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)

    print("\n=== Regression for Imported Missing Fraction ===")
    for name, coef in zip(X.columns, model.coef_):
        print(f"  {name}: {coef:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R^2: {model.score(X_scaled, y):.4f}")


def regression_analysis_local(local_df):
    """ Regress missing_frac on local probabilities & avg_transit_time. """
    X = local_df[[
        "p_counterfeit_local",
        "p_warehouse_theft_local",
        "p_last_mile_theft_local",
        "avg_transit_time_local"
    ]]
    y = local_df["missing_frac"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)

    print("\n=== Regression for Local Missing Fraction ===")
    for name, coef in zip(X.columns, model.coef_):
        print(f"  {name}: {coef:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R^2: {model.score(X_scaled, y):.4f}")


if __name__ == "__main__":
    np.random.seed(42)

    imported_df, local_df = main_simulation(num_sims=1000, N=10000)

    regression_analysis_imported(imported_df)
    regression_analysis_local(local_df)