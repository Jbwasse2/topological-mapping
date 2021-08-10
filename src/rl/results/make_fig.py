import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pudb
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.3)
matplotlib.rcParams["font.family"] = "Helvetica"
font = {"weight": "bold"}

matplotlib.rc("font", **font)


def get_results_codes(list_of_strings):
    for s in list_of_strings:
        if len(s) > 0:
            if s[0] == "[":
                return eval(s)


def get_dirty_results(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return get_results_codes(content)


def make_figure(results):
    entries = {
        "env": [],
        "baseline": [],
        "ours": [],
    }
    keys = list(results.keys())
    for key in keys:
        dirty, clean = results[key]
        skip_flag = 0
        if dirty == None:
            print("Missing Dirty for ", key)
            skip_flag = 1
        if clean == None:
            print("Missing Clean for ", key)
            skip_flag = 1
        if skip_flag:
            continue
        baseline = dirty[0]
        dirty_bad = dirty[1] + dirty[2] + dirty[3]
        ours = clean[0]
        clean_bad = clean[1] + clean[2] + clean[3]
        entries["env"].append(key)
        entries["baseline"].append(baseline)
        entries["ours"].append(ours)
    df = pd.DataFrame.from_dict(entries)
    df = df.set_index("env")
    pu.db
    ax = df.plot.bar(rot=0, figsize=(12,3))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.xlabel("")
    plt.savefig("./nav_results_all.jpg", bbox_inches="tight")


if __name__ == "__main__":
    results = {}
    for env in glob.glob("./dirty/*"):
        if "Mifflintown" in env or "McCloud" in env:
            continue
        result = get_dirty_results(env)
        results[os.path.basename(env)] = [result]
    for env in glob.glob("./clean/*"):
        if "Mifflintown" in env or "McCloud" in env:
            continue
        result = get_dirty_results(env)
        results[os.path.basename(env)].append(result)
    make_figure(results)

