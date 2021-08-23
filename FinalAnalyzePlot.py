import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def training_return(file_path):
    df = pd.read_csv(file_path)
    values = df["return"].to_numpy()
    sma = []
    for i in range(0, values.size):
        array = values[:i + 1]
        mean_reward = np.mean(array[-100:])
        sma.append(mean_reward)

    plt.clf()
    plt.title('training return')
    plt.xlabel('episodes')
    plt.ylabel('mean weekly return')
    plt.plot(sma)
    plt.savefig(file_path[:-19] + 'training_return.png', dpi=200, bbox_inches='tight')

def training_profit(file_path):
    df = pd.read_csv(file_path)
    values = df["profit"].to_numpy()
    sma = []
    for i in range(0, values.size):
        array = values[:i + 1]
        mean_reward = np.mean(array[-100:])
        sma.append(mean_reward)

    plt.clf()
    plt.title('training profit')
    plt.xlabel('episodes')
    plt.ylabel('mean weekly profit')
    plt.plot(sma)
    plt.savefig(file_path[:-19] + 'training_profit.png', dpi=200, bbox_inches='tight')

def training_pen(file_path):
    df = pd.read_csv(file_path)
    values = df["pen"].to_numpy()
    sma = []
    for i in range(0, values.size):
        array = values[:i + 1]
        mean_reward = np.mean(array[-100:])
        sma.append(mean_reward)

    plt.clf()
    plt.title('training pen')
    plt.xlabel('episodes')
    plt.ylabel('mean weekly pen')
    plt.plot(sma)
    plt.savefig(file_path[:-19] + 'training_pen.png', dpi=200, bbox_inches='tight')

def training_cvar(file_path):
    df = pd.read_csv(file_path)
    values = df["cvar"].to_numpy()
    sma = []
    for i in range(0, values.size):
        array = values[:i + 1]
        mean_reward = np.mean(array[-100:])
        sma.append(mean_reward)

    plt.clf()
    plt.title('training cvar')
    plt.xlabel('episodes')
    plt.ylabel('mean weekly cvar')
    plt.plot(sma)
    plt.savefig(file_path[:-19] + 'training_cvar.png', dpi=200, bbox_inches='tight')


def training_negative(file_path):
    df = pd.read_csv(file_path)
    values = (df["pen"]+df["cvar"]).to_numpy()
    sma = []
    for i in range(0, values.size):
        array = values[:i + 1]
        mean_reward = np.mean(array[-100:])
        sma.append(mean_reward)

    plt.clf()
    plt.title('training neg')
    plt.xlabel('episodes')
    plt.ylabel('mean weekly neg')
    plt.plot(sma)
    plt.savefig(file_path[:-19] + 'training_neg.png', dpi=200, bbox_inches='tight')


def cum_income(file_path):
    df = pd.read_csv(file_path)
    df["cum_base"] = df["income_base"].cumsum()
    df["cum_opt"] = df["income_opt"].cumsum()
    df.to_csv(file_path[:-20] + 'cum.csv')

    plt.clf()
    plt.title("performance on testing data")
    plt.ylabel('cumulative income')
    plt.xlabel('hours in 2019')
    plt.plot(df['cum_base'], label='base')
    plt.plot(df['cum_opt'], label='optimized')
    plt.legend(fontsize=8)

    plt.savefig(file_path[:-20] + 'test_cum_income.png', dpi=200, bbox_inches='tight')


def battery_control(file_path):
    T = 168
    df = pd.read_csv(file_path)
    max_idx = df.shape[0] - T
    start_idx = 0
    week = 1
    while start_idx <= max_idx:
        idx = start_idx
        soc = df.loc[idx:idx + T - 1, 'soc']
        control = df.loc[idx:idx + T - 1, 'control']
        delta = df.loc[idx:idx + T - 1, 'delta']
        price = df.loc[idx:idx + T - 1, 'map_ap']  # 1718: 0.08 --->1.31
        fg = df.loc[idx:idx + T - 1, 'fore_gen']
        # soc and delta e
        plt.clf()
        fig, ax1 = plt.subplots()
        index_soc = np.arange(len(soc))
        ax1.bar(index_soc, height=soc, color='grey', label='soc')
        ax1.bar(index_soc, height=control, color='cyan', label='action', alpha=0.6)
        ax1.bar(index_soc, height=delta, color='yellow', label='delta')
        ax1.set_ylabel('soc/delta/control', color='black')
        plt.legend(loc=1)
        plt.grid(axis="y")
        # price
        ax2 = ax1.twinx()
        x = np.arange(len(soc))
        y1 = price
        y2 = fg

        ax2.plot(x, y1, label='actual price', color='orange')
        ax2.plot(x, y2, label='forecasted generation', color='purple')
        ax2.set_ylabel('price/generation', color='black')

        ax1.set_ylim(-1, 1)
        ax2.set_ylim(0, 1.5)
        plt.xticks(np.arange(min(x), max(x) + 24, 24.0))
        plt.legend(loc=2)

        png_name = file_path[:-15] + "week_" + str(week) + ".png"
        plt.savefig(png_name, dpi=200, bbox_inches='tight')
        start_idx += 168
        week += 1


def sep_monthly(data):
    dataframe_collection = {}
    for month in range(1, 13):
        temp = data.loc[data['month'] == month]
        temp = temp.reset_index(drop=True)
        dataframe_collection[month] = temp

    return dataframe_collection


def compute_alpha_CVaR(df, alpha):
    base_values = df['income_base'].groupby(df["hour"] // 24).sum().to_numpy()
    base_avg = np.mean(base_values)
    base_dev = base_values - base_avg
    base_sortnegdev = np.sort(base_dev[base_dev < 0])
    nr = int(np.ceil((1 - alpha) * len(base_values)))
    CVaR_base = - base_sortnegdev[0:nr].mean().round(2)

    opt_values = df['income_opt'].groupby(df["hour"] // 24).sum().to_numpy()
    opt_avg = np.mean(opt_values)
    opt_dev = opt_values - opt_avg
    opt_sortnegdev = np.sort(opt_dev[opt_dev < 0])
    nr = int(np.ceil((1 - alpha) * len(opt_values)))
    CVaR_opt = - opt_sortnegdev[0:nr].mean().round(2)

    record = {
        "base": CVaR_base,
        "opt": CVaR_opt,
    }

    return record


def income_to_cvar(file_path, alpha):
    income_df = pd.read_csv(file_path)
    dfs = sep_monthly(income_df)
    CVaR_df = pd.DataFrame()
    for month in range(1, 13):
        df_value = compute_alpha_CVaR(dfs[month], alpha)
        CVaR_df = CVaR_df.append(df_value, ignore_index=True)
    CVaR_df["month"] = CVaR_df.index + 1
    save_path = file_path[:-20] + "cvar.csv"
    CVaR_df.to_csv(save_path)


def plot_cvar(file_path):
    CVaR_df = pd.read_csv(file_path)

    labels = CVaR_df["month"].apply(lambda x: str(x)).tolist()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    CVaR_base = CVaR_df["base"].tolist()
    CVaR_opt = CVaR_df["opt"].tolist()

    plt.clf()
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, CVaR_base, width, label='base')
    rects2 = ax.bar(x + width / 2, CVaR_opt, width, label='optimized')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Euro')
    ax.set_xlabel('Month in 2019')
    ax.set_title('90% CVaR')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    path = file_path[:-8] + "90cvar.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')


def storage_analyze(file_path):
    record = pd.read_csv(file_path)
    record["abs_delta"] = record["delta"].abs()
    total_delta = record["abs_delta"].sum()
    total_charge = record["e_ch"].sum()
    total_discharge = record["e_dis"].sum()
    print(
        "total delta:", total_delta, "\n",
        "total charge:", total_charge, "\n",
        "total discharge:", total_discharge, "\n"
    )
    txt_path = file_path[:-27] + "total storage.txt"
    txt_file = open(txt_path, "a+")
    txt_file.write("total delta:" + str(total_delta) + "\n" +
               "total charge:" + str(total_charge) + "\n" +
               "total discharge:" + str(total_discharge) + "\n"
            )
    txt_file.close()

    dfs = sep_monthly(record)
    storage_df = pd.DataFrame()
    for month in range(1, 13):
        monthly_ech = dfs[month]["e_ch"].sum()
        monthly_edis = dfs[month]["e_dis"].sum()
        value = {
            "monthly charging power": monthly_ech,
            "monthly discharging power": monthly_edis
        }
        storage_df = storage_df.append(value, ignore_index=True)
    storage_df["month"] = storage_df.index + 1
    csv_path = file_path[:-23] + "storage.csv"
    storage_df.to_csv(csv_path)

    compare_df = storage_df
    gdf = pd.read_csv("./data/pv2019genstat.csv")
    compare_df["total generation"] = gdf["total_gen"]
    compare_df["percentage"] = (compare_df["monthly charging power"] / compare_df["total generation"] * 100).round(2)
    labels = compare_df["month"].apply(lambda x: str(x)).tolist()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    height = compare_df["percentage"].tolist()

    #sns.set_theme()
    #sns.set_palette("Set2")
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(x, height, width)
    ax.set_title('percentage of discharging power in generation')
    ax.set_ylabel('%')
    ax.set_xlabel('Month in 2019')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fig.tight_layout()
    png_path = file_path[:-23] + "percentage of charging in generation.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight')

