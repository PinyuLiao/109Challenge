import numpy as np

def simulate_imported_drugs(N=10000):
    """
    Simulates N imported drug shipments with the following sequence:
      1) Is shipment counterfeit or legit?
      2) Export customs seizure.
      3) Base transport time ~ Normal(4.74, 0.785).
      4) Import customs seizure.
      5) 1–5 stops, each adding Normal(1, 0.6) days + warehouse theft check.
      6) Last-mile theft check.
      7) If not seized/stolen, check if total time < 7 days => on-time.

    Returns a dict with:
      - final_legit_on_time
      - final_counterfeit_on_time
      - final_legit_total
      - final_counterfeit_total
      - total_surviving (both legit + counterfeit that arrived on time)
    """

    # === 1) Draw random probabilities used in this simulation run ===

    # Probability a shipment is counterfeit: uniform(0.1, 0.2)
    p_counterfeit = np.random.uniform(0.1, 0.2)

    # Export customs seizure probabilities
    p_seized_export_counterfeit = np.random.uniform(0.0001, 0.001)
    p_seized_export_legit = np.random.beta(3, 10000)  # Very small for legit

    # Import customs seizure (0.1–0.5% => 0.001–0.005 for counterfeit)
    p_seized_import_counterfeit = np.random.uniform(0.001, 0.005)
    p_seized_import_legit = np.random.beta(3, 10000)  # small for legit

    # Warehouse theft probability (applies at each stop)
    p_warehouse_theft = np.random.uniform(0.0000014, 0.00001)

    # Last-mile theft
    p_last_mile_theft = np.random.uniform(0.0001, 0.001)

    # Base transport time distribution
    base_mean = 4.74
    base_var = 0.785
    base_std = np.sqrt(base_var)

    # === 2) Initialize counters ===

    legit_total = 0
    counterfeit_total = 0
    final_legit_on_time = 0
    final_counterfeit_on_time = 0

    # === 3) Simulate each shipment individually ===
    for _ in range(N):

        # --- A) Is this shipment counterfeit or legit? ---
        is_counterfeit = (np.random.random() < p_counterfeit)
        if is_counterfeit:
            counterfeit_total += 1
        else:
            legit_total += 1

        # --- B) Export Customs Seizure ---
        if is_counterfeit:
            if np.random.random() < p_seized_export_counterfeit:
                # Seized at export => does not proceed
                continue
        else:
            if np.random.random() < p_seized_export_legit:
                continue

        # --- C) Base Transport Time ---
        transit_time = np.random.normal(base_mean, base_std)

        # --- D) Import Customs Seizure ---
        if is_counterfeit:
            if np.random.random() < p_seized_import_counterfeit:
                continue
        else:
            if np.random.random() < p_seized_import_legit:
                continue

        # --- E) Extra Stops: 1–5 stops, each with time + theft ---
        lambda_stops = 3  # Adjust based on real-world averages
        num_stops = np.random.poisson(lambda_stops)

        # Ensure at least 1 stop (since Poisson can give 0)
        num_stops = max(1, num_stops)
        stolen = False
        for _stop in range(num_stops):
            # Add random time for this stop
            extra_time = np.random.normal(0.5, np.sqrt(0.6))
            transit_time += extra_time

            # Warehouse theft at this stop?
            if np.random.random() < p_warehouse_theft:
                stolen = True
                break

        if stolen:
            # Shipment lost in warehouse theft
            continue

        # --- F) Last-Mile Theft ---
        if np.random.random() < p_last_mile_theft:
            continue

        # --- G) Check if on-time (transit_time < 7) ---
        if transit_time < 7:
            if is_counterfeit:
                final_counterfeit_on_time += 1
            else:
                final_legit_on_time += 1

    # === 4) Compile results ===
    results = {
        "final_legit_on_time": final_legit_on_time,
        "final_counterfeit_on_time": final_counterfeit_on_time,
        "final_legit_total": legit_total,
        "final_counterfeit_total": counterfeit_total,
        "total_surviving": final_legit_on_time + final_counterfeit_on_time
    }
    return results


def simulate_local_drugs(N=10000):
    p_counterfeit_local = np.random.uniform(0.1, 0.25)
    p_warehouse_theft_local = np.random.uniform(0.0000014, 0.00001)
    p_last_mile_theft_local = np.random.uniform(0.0001, 0.001)

    # Counters
    final_legit_on_time = 0
    final_counterfeit_on_time = 0
    legit_total = 0
    counterfeit_total = 0

    for _ in range(N):
        # A) Is it counterfeit?
        is_counterfeit = (np.random.random() < p_counterfeit_local)
        if is_counterfeit:
            counterfeit_total += 1
        else:
            legit_total += 1

        # B) For local, maybe we skip import/export. Just do stops + theft
        transit_time = 0.0

        # Suppose local has 1–3 stops, etc.
        lambda_stops = 1
        num_stops = np.random.poisson(lambda_stops)
        num_stops = max(1, num_stops)

        stolen = False
        for _stop in range(num_stops):
            # add random time
            extra_time = np.random.normal(0.5, np.sqrt(0.6))
            transit_time += extra_time

            # warehouse theft
            if np.random.random() < p_warehouse_theft_local:
                stolen = True
                break

        if stolen:
            # This shipment never arrives
            continue

        # last-mile theft
        if np.random.random() < p_last_mile_theft_local:
            continue

        # arrived => check time
        if transit_time < 7:
            if is_counterfeit:
                final_counterfeit_on_time += 1
            else:
                final_legit_on_time += 1

    # Return results
    results = {
        "final_legit_on_time": final_legit_on_time,
        "final_counterfeit_on_time": final_counterfeit_on_time,
        "final_legit_total": legit_total,
        "final_counterfeit_total": counterfeit_total,
        "total_surviving": final_legit_on_time + final_counterfeit_on_time,
        "p_counterfeit_local": p_counterfeit_local,
        "p_warehouse_theft_local": p_warehouse_theft_local,
        "p_last_mile_theft_local": p_last_mile_theft_local
    }
    return results


def main_simulation(num_sims=1000, N=10000):
    """
    Runs multiple Monte Carlo simulations, each time simulating
    both imported and local supply chains. Collects and returns:
      - Average fraction of shipments that are legit vs counterfeit
      - Average fraction that arrive on time (legit vs counterfeit)
    """

    # Collect results across simulations
    imported_legit_arrivals = []
    imported_counterfeit_arrivals = []
    local_legit_arrivals = []
    local_counterfeit_arrivals = []

    # Additional stats: overall fraction legit vs counterfeit
    imported_legit_fraction = []
    imported_counterfeit_fraction = []
    local_legit_fraction = []
    local_counterfeit_fraction = []

    for _ in range(num_sims):
        imported_res = simulate_imported_drugs(N)
        local_res = simulate_local_drugs(N)

        # --- Imported ---

        # Fraction of shipments that were legit/counterfeit (before any seizure/theft)
        frac_legit_imported = imported_res["final_legit_total"] / N
        frac_counterfeit_imported = imported_res["final_counterfeit_total"] / N

        imported_legit_fraction.append(frac_legit_imported)
        imported_counterfeit_fraction.append(frac_counterfeit_imported)

        # Among those legit or counterfeit, fraction that made it on time
        if imported_res["final_legit_total"] > 0:
            frac_legit_arrived = imported_res["final_legit_on_time"] / imported_res["final_legit_total"]
        else:
            frac_legit_arrived = 0

        if imported_res["final_counterfeit_total"] > 0:
            frac_counterfeit_arrived = (imported_res["final_counterfeit_on_time"] /
                                        imported_res["final_counterfeit_total"])
        else:
            frac_counterfeit_arrived = 0

        imported_legit_arrivals.append(frac_legit_arrived)
        imported_counterfeit_arrivals.append(frac_counterfeit_arrived)

        # --- Local ---

        # Fraction of shipments that were legit/counterfeit
        frac_legit_local = local_res["final_legit_total"] / N
        frac_counterfeit_local = local_res["final_counterfeit_total"] / N

        local_legit_fraction.append(frac_legit_local)
        local_counterfeit_fraction.append(frac_counterfeit_local)

        # Among those legit/counterfeit, fraction that made it on time
        if local_res["final_legit_total"] > 0:
            frac_legit_local_arrived = local_res["final_legit_on_time"] / local_res["final_legit_total"]
        else:
            frac_legit_local_arrived = 0

        if local_res["final_counterfeit_total"] > 0:
            frac_counterfeit_local_arrived = (local_res["final_counterfeit_on_time"] /
                                              local_res["final_counterfeit_total"])
        else:
            frac_counterfeit_local_arrived = 0

        local_legit_arrivals.append(frac_legit_local_arrived)
        local_counterfeit_arrivals.append(frac_counterfeit_local_arrived)

    # Compute average (mean) across simulations
    results_summary = {
        # Overall fraction that are legit/counterfeit
        "imported_mean_legit_fraction": np.mean(imported_legit_fraction),
        "imported_mean_counterfeit_fraction": np.mean(imported_counterfeit_fraction),
        "local_mean_legit_fraction": np.mean(local_legit_fraction),
        "local_mean_counterfeit_fraction": np.mean(local_counterfeit_fraction),

        # Fraction of legit/counterfeit that survive to on-time delivery
        "imported_legit_on_time": np.mean(imported_legit_arrivals),
        "imported_counterfeit_on_time": np.mean(imported_counterfeit_arrivals),
        "local_legit_on_time": np.mean(local_legit_arrivals),
        "local_counterfeit_on_time": np.mean(local_counterfeit_arrivals),
    }

    return results_summary


if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility

    summary = main_simulation(num_sims=1000, N=10000)

    print("=== Simulation Results (Averaged over 1000 runs) ===\n")

    # Overall fraction that were legit vs counterfeit
    print(f"Imported  : Mean Legit Fraction       = {summary['imported_mean_legit_fraction']:.4f}")
    print(f"Imported  : Mean Counterfeit Fraction = {summary['imported_mean_counterfeit_fraction']:.4f}")
    print(f"Local     : Mean Legit Fraction       = {summary['local_mean_legit_fraction']:.4f}")
    print(f"Local     : Mean Counterfeit Fraction = {summary['local_mean_counterfeit_fraction']:.4f}\n")

    # Fraction of each category that successfully arrives on time
    print(f"Imported  : Legit On-Time Arrival     = {summary['imported_legit_on_time']:.4f}")
    print(f"Imported  : Counterfeit On-Time Arrival = {summary['imported_counterfeit_on_time']:.4f}")
    print(f"Local     : Legit On-Time Arrival     = {summary['local_legit_on_time']:.4f}")
    print(f"Local     : Counterfeit On-Time Arrival = {summary['local_counterfeit_on_time']:.4f}")
