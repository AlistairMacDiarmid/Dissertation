from __future__ import annotations

import csv
import os


def save_layout(
        layout,
        scenario,
        energy_cost,
        seed: int | None,
        filename: str,
        base_dir: str = ".",

):
    """
    save turbine positions to a csv file.
    columns:
    - turbine_id
    - x
    - y

    :param layout:
    :param scenario:
    :param filepath:
    :return:
    """

    #make results dir if it doesnt already exist
    results_dir = os.path.join(base_dir, "results")
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    filepath = os.path.join(results_dir, filename)

    #write data to file
    with open(filepath, "w") as f:
        writer = csv.writer(f)

        #metadata
        writer.writerow(["#seed",seed ])
        writer.writerow(["#energy_cost",energy_cost])
        writer.writerow(["#nturbines", scenario.nturbines])
        writer.writerow(["#farm_width", scenario.width])
        writer.writerow(["#farm_height", scenario.height])
        writer.writerow(["#min_spacing", scenario.minDist])

        #blank transition line
        writer.writerow([])

        # header
        writer.writerow(["turbine_id", "x", "y"])
        #fields
        for i, (x, y) in enumerate(layout):
            writer.writerow([i, float(x), float(y)])

    print(f"layout saved to {filepath}")

