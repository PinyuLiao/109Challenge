import numpy as np

def simulate_imported_batch(N=10000):
    """
    Simulates a batch of imported pharmaceutical shipments using batch probabilities
    and Poisson distributions for rare events.

    Returns:
      - Total legit shipments that survived and were on time
      - Total counterfeit shipments that survived and were on time
      - Total legit shipments (before losses)
      - Total counterfeit shipments (before losses)
    """

    # --- 1. Generate Counterfeit vs. Legit ---
    p_counterfeit = np.random.uniform(0.1, 0.2)
    counterfeit_import = np.random.binomial(N, p_counterfeit)
    legit_import = N - counterfeit_import

    # --- 2. Export Customs Seizure (Using Poisson for rare event modeling) ---
    p_seized_export_counterfeit = np.random.uniform(0.0001, 0.001)
    p_seized_export_legit = np.random.beta(3, 10000)

    lambda_export_counterfeit = counterfeit_import * p_seized_export_counterfeit
    lambda_export_legit = legit_import * p_seized_export_legit

    seized_export_counterfeit = np.random.poisson(lambda_export_counterfeit)
    seized_export_legit = np.random.poisson(lambda_export_legit)

    remaining_counterfeit_export = max(0, counterfeit_import - seized_export_counterfeit)
    remaining_legit_export = max(0, legit_import - seized_export_legit)

    # --- 3. Import Customs Seizure (Using Poisson) ---
    p_seized_import_counterfeit = np.random.uniform(0.001, 0.005)
    p_seized_import_legit = np.random.beta(3, 10000)

    lambda_import_counterfeit = remaining_counterfeit_export * p_seized_import_counterfeit
    lambda_import_legit = remaining_legit_export * p_seized_import_legit

    seized_import_counterfeit = np.random.poisson(lambda_import_counterfeit)
    seized_import_legit = np.random.poisson(lambda_import_legit)

    remaining_counterfeit_import = max(0, remaining_counterfeit_export - seized_import_counterfeit)
    remaining_legit_import = max(0, remaining_legit_export - seized_import_legit)

    # --- 4. Transit Time ---
    transit_times = np.random.normal(4.74, np.sqrt(0.785), N)
    on_time_mask = transit_times < 7
    frac_on_time = np.sum(on_time_mask) / N

    legit_on_time_import = int(remaining_legit_import * frac_on_time)
    counterf_on_time_import = int(remaining_counterfeit_import * frac_on_time)

    # --- 5. Warehouse Theft (Using Poisson for rare event modeling) ---
    p_warehouse_theft = np.random.uniform(0.0000014, 0.00001)
    lambda_warehouse_theft_legit = legit_on_time_import * p_warehouse_theft
    lambda_warehouse_theft_counterfeit = counterf_on_time_import * p_warehouse_theft

    theft_legit_warehouse = np.random.poisson(lambda_warehouse_theft_legit)
    theft_counterfeit_warehouse = np.random.poisson(lambda_warehouse_theft_counterfeit)

    remaining_legit_warehouse = max(0, legit_on_time_import - theft_legit_warehouse)
    remaining_counterfeit_warehouse = max(0, counterf_on_time_import - theft_counterfeit_warehouse)

    # --- 6. Last-Mile Theft (Using Poisson) ---
    p_last_mile_theft = np.random.uniform(0.0001, 0.001)
    lambda_last_mile_theft_legit = remaining_legit_warehouse * p_last_mile_theft
    lambda_last_mile_theft_counterfeit = remaining_counterfeit_warehouse * p_last_mile_theft

    theft_legit_last_mile = np.random.poisson(lambda_last_mile_theft_legit)
    theft_counterfeit_last_mile = np.random.poisson(lambda_last_mile_theft_counterfeit)

    final_legit = max(0, remaining_legit_warehouse - theft_legit_last_mile)
    final_counterfeit = max(0, remaining_counterfeit_warehouse - theft_counterfeit_last_mile)

    return final_legit, final_counterfeit, legit_import, counterfeit_import


def batch_simulation(num_sims=1000, N=10000):
    """
    Runs multiple batch simulations for imported pharmaceuticals.

    Returns a summary of the results.
    """

    imported_legit_arrivals = []
    imported_counterfeit_arrivals = []
    imported_legit_total = []
    imported_counterfeit_total = []

    for _ in range(num_sims):
        imp_legit, imp_counter, imp_legit_tot, imp_counter_tot = simulate_imported_batch(N)

        imported_legit_arrivals.append(imp_legit / imp_legit_tot if imp_legit_tot > 0 else 0)
        imported_counterfeit_arrivals.append(imp_counter / imp_counter_tot if imp_counter_tot > 0 else 0)

        imported_legit_total.append(imp_legit_tot)
        imported_counterfeit_total.append(imp_counter_tot)

    results_summary = {
        "imported_mean_legit_fraction": np.mean(imported_legit_total) / N,
        "imported_mean_counterfeit_fraction": np.mean(imported_counterfeit_total) / N,
        "imported_legit_on_time": np.mean(imported_legit_arrivals),
        "imported_counterfeit_on_time": np.mean(imported_counterfeit_arrivals),
    }

    return results_summary


if __name__ == "__main__":
    np.random.seed(42)

    summary = batch_simulation(num_sims=1000, N=10000)

    print("=== Batch Simulation Results ===\n")
    print(f"Imported  : Mean Legit Fraction       = {summary['imported_mean_legit_fraction']:.4f}")
    print(f"Imported  : Mean Counterfeit Fraction = {summary['imported_mean_counterfeit_fraction']:.4f}")
    print(f"Imported  : Legit On-Time Arrival     = {summary['imported_legit_on_time']:.4f}")
    print(f"Imported  : Counterfeit On-Time Arrival = {summary['imported_counterfeit_on_time']:.4f}")