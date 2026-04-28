
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys
import os

# Ensure we can import from local modules if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # scripts/

# Import plot settings
try:
    import plot_formatting
except ImportError:
    pass # valid if plot_formatting is in the same dir

# Import colors
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')) # papers/TOSG/

    # Fallback if import fails
OKABE_ITO_DEVICE_COLORS = {
    "DER": "#0072B2",
    "EV": "#D55E00",
    "DL": "#009E73",
}

OKABE_ITO_BENCHMARK_COLORS = {
    "Baseline": "#0072B2",
    "General Affine": "#D55E00",
    "Zonotope": "#009E73",
    "G-Polymatroid": "#009E73",
}

def plot_complexity():
    data_path = os.path.join(os.path.dirname(__file__), '../data/compVdevice.csv')
    df = pd.read_csv(data_path)

    # Use the formatting styles (this usually updates rcParams)
    # If plot_formatting has a function to setup, call it, otherwise standard update
    plt.rcParams.update({
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size":0,
        "ytick.minor.size":0,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    # Plot each benchmark
    benchmarks = df['benchmark'].unique()
    
    # We want specific order or colors
    for benchmark in ['DER', 'EV',  'DL']: # Ordered by complexity roughly
        if benchmark not in benchmarks: continue
        
        group = df[df['benchmark'] == benchmark].sort_values('T')
        
        color = OKABE_ITO_DEVICE_COLORS.get(benchmark, 'black')
        
        ax.plot(
            group['T'],
            group['time'],
            marker='o',
            linewidth=1.0, # Slightly thicker for visibility
            markersize=3,
            color=color,
            label=benchmark
        )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Time Horizon $T$')
    ax.set_ylabel('Computation Time (s)')
    
    # Grid
    ax.grid(True, which="both", ls="--", alpha=0.1, linewidth=0.5)

    # Legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fancybox=False)
    
    # Ticks - make them look nice on log scale
    # For T=6 to 60, standard log ticks (10, 100) are scarce.
    # We can force specific ticks
    ax.set_xticks([6, 12, 24, 48, 60])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    out_path = os.path.join(os.path.dirname(__file__), '../figures/compVdevice_loglog.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    print(f"Saving plot to {out_path}")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_bench():
    data_path = os.path.join(os.path.dirname(__file__), '../data/compVt.csv')
    cs = pd.read_csv(data_path)

    benchmarks = sorted(cs['benchmark'].unique())
    color_map = {}
    for b in benchmarks:
        resolved_color = OKABE_ITO_BENCHMARK_COLORS.get(b)
        if resolved_color is None:
            resolved_color = OKABE_ITO_PALETTE[len(color_map) % len(OKABE_ITO_PALETTE)]
        color_map[b] = resolved_color

    with plt.rc_context(ieee_rcparams):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
        for benchmark, group in cs.groupby('benchmark'):
            group = group.sort_values('T')
            ax.plot(
                group['T'],
                group['time'],
                marker='o',
                linewidth=.7,
                markersize=3,
                color=color_map[benchmark],
                label=benchmark,
            )
        ax.set_xlabel('Time Horizon $T$')
        ax.set_xticks(range(12, 61, 12))
        ax.set_ylabel('Computation Time (s)')
        ax.set_yscale('log')

        ax.tick_params(direction='in', top=True, right=True, which='both')
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(axis='y', which='both', linestyle='--', alpha=0.5)
        desired_order = ['Baseline', 'General Affine','Zonotope', 'G-Polymatroid']
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = []
        ordered_labels = []
        for label in desired_order:
            if label in labels:
                idx = labels.index(label)
                ordered_handles.append(handles[idx])
                ordered_labels.append(labels[idx])
        for handle, label in zip(handles, labels):
            if label not in desired_order:
                ordered_handles.append(handle)
                ordered_labels.append(label)


        legend = ax.legend(ordered_handles, ordered_labels, loc='upper left', frameon=True, framealpha=0.9)
        legend.get_frame().set_linewidth(0.6)
        plt.savefig('../figures/compVt.pdf', format='pdf', bbox_inches='tight')
        plt.show()


def plot_case():

        # Load case study data
    cs = pd.read_csv('../data/case_study.csv')
    cs = cs.sort_values(['benchmark', 'run_id', 't'])


    # Cumulative sum over t per (benchmark, run_id)
    cs['cum_value'] = cs.groupby(['benchmark', 'run_id'])['value'].cumsum()
    cs['cum_value'] -= cs.groupby(['benchmark', 'run_id'])['cum_value'].transform('first')

    cs['cum_value'] /= 1000000

    # cs['cum_value'] /= 1000
    # Aggregate quantiles across runs at each t per benchmark
    quantiles = cs.groupby(['benchmark', 't'])['cum_value'].agg(
        q10=lambda x: x.quantile(0.10),
        q25=lambda x: x.quantile(0.25),
        q50='median',
        q75=lambda x: x.quantile(0.75),
        q90=lambda x: x.quantile(0.90),
    ).reset_index()

    benchmarks = sorted(quantiles['benchmark'].unique())
    display_labels = {
        'g-polymatroid': 'G-Polymatroid',
        'general_affine': 'General Affine',
        'base_line': 'Baseline',
        'zonotope': 'Zonotope',
    }
    color_map = {}
    for b in benchmarks:
        resolved_color = None
        for candidate in (b, display_labels.get(b)):
            if candidate in OKABE_ITO_BENCHMARK_COLORS:
                resolved_color = OKABE_ITO_BENCHMARK_COLORS[candidate]
                break
        if resolved_color is None:
            resolved_color = OKABE_ITO_PALETTE[len(color_map) % len(OKABE_ITO_PALETTE)]
        color_map[b] = resolved_color

    with plt.rc_context(ieee_rcparams):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

        for b in benchmarks:
            g = quantiles[quantiles['benchmark'] == b].sort_values('t')
            c = color_map[b]
            label = display_labels.get(b, b.replace('_', ' ').title())
            # Median line
            ax.plot(g['t'], g['q50'], color=c, label=label, linewidth=1.4)
            # 25-75 IQR band
            ax.fill_between(g['t'], g['q25'], g['q75'], color=c, alpha=0.18, linewidth=0)
            # 10-90 band
            # ax.fill_between(g['t'], g['q10'], g['q90'], color=c, alpha=0.10, linewidth=0)

        ax.set_xlabel('Day')
        ax.set_ylabel(r"Cumulative Charging Cost (£$ \times 10^6$)")
        ax.tick_params(direction='in', top=True, right=True, which='both')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
        ax.set_xticks(range(0, 31, 10))
        ax.set_ylim(0,2.1)
        ax.set_yticks(np.array([0,0.5,1,1.5,2]))

        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(axis='y', which='both', linestyle='--', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        desired_order = ['Baseline', 'General Affine','Zonotope', 'G-Polymatroid']
        order = desired_order + [lbl for lbl in labels if lbl not in desired_order]
        ordered = [(h, l) for h, l in zip(handles, labels) if l in order]
        ordered.sort(key=lambda hl: order.index(hl[1]))
        leg = ax.legend(
            [h for h, _ in ordered],
            [l for _, l in ordered],
            loc='upper left',
            frameon=True,
            framealpha=0.9,
        )
        leg.get_frame().set_linewidth(0.6)
        plt.savefig('../figures/case_study.pdf', format='pdf', bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    plot_complexity()
    plot_bench()
    plot_case()
