from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def gen_solution_scaling_plot(
        dataset_name: str,
        bon_mav_accuracies: Optional[List[float]],
        self_cons_accuracies: Optional[List[float]],
        pass_at_1_accuracy: float,
        save_filepath: str,
        font: str = 'DejaVu Serif') -> None:
    """Generate a solution scaling plot for the given dataset and accuracies."""
    if bon_mav_accuracies:
        n_solutions = len(bon_mav_accuracies)
        if self_cons_accuracies:
            assert n_solutions == len(self_cons_accuracies), f"Length mismatch: {n_solutions} != {len(self_cons_accuracies)}"
    else:
        n_solutions = len(self_cons_accuracies)
    n_solutions_list = list(range(1, n_solutions + 1))

    # Set up the figure
    plt.figure(figsize=(7, 4))
    ax = plt.gca()

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font],
    })
    ax.grid(True, color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)

    # Set consistent font size
    FONTSIZE = 12

    # Define metrics styling
    metrics_style = {
        'mav': {
            'color': 'navy',
            'label': 'BoN-MAV@n',
            'linestyle': '-',
            'linewidth': 1.5,
            'marker': '*',
            'zorder': 4,
            'markersize': 11,
            'markeredgewidth': 0.5,
            'markeredgecolor': 'white',
        },
        'self_cons': {
            'color': 'orange',
            'label': 'cons@n',
            'linestyle': '--',
            'linewidth': 1.5,
            'marker': 's',
            'markersize': 4,
            'zorder': 2
        },
        'pass_at_1': {
            'color': 'black',
            'label': 'pass@1',
            'linestyle': '--',
            'linewidth': 1,
            'zorder': 1,
        }
    }

    # Plot metrics
    if bon_mav_accuracies:
        ax.plot(
            n_solutions_list,
            bon_mav_accuracies,
            **{k: v for k, v in metrics_style['mav'].items() if k != 'label'},
            label=metrics_style['mav']['label']
        )

    if self_cons_accuracies:
        ax.plot(
            n_solutions_list,
            self_cons_accuracies,
            **{k: v for k, v in metrics_style['self_cons'].items() if k != 'label'},
            label=metrics_style['self_cons']['label']
        )

    # Add pass@1 baseline
    ax.axhline(
        y=pass_at_1_accuracy,
        **{k: v for k, v in metrics_style['pass_at_1'].items() if k != 'label'},
        label=metrics_style['pass_at_1']['label'],
    )

    # Set x-ticks
    x_ticks = [i for i in range(1, n_solutions + 1)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])

    # Set y-axis limits and ticks
    y_vals = [pass_at_1_accuracy]
    if bon_mav_accuracies:
        y_vals.extend(bon_mav_accuracies)
    if self_cons_accuracies:
        y_vals.extend(self_cons_accuracies)
    y_vals = [y for y in y_vals if y is not None]
    y_min, y_max = min(y_vals), max(y_vals)
    y_range = y_max - y_min

    # Choose number of ticks based on range
    # n_ticks = 2 if y_range <= 1 else 3
    # y_ticks = np.linspace(np.floor(y_min), np.ceil(y_max), n_ticks)
    # y_ticks = np.round(y_ticks).astype(int)
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels([str(y) for y in y_ticks])

    # Add padding to y-limits
    padding = y_range * 0.2
    ax.set_ylim(y_min - padding, y_max + padding)

    # Style ticks
    # ax.tick_params(direction='in', labelsize=FONTSIZE, length=6, width=0.5, color='gray')
    ax.tick_params(labelsize=FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(font)

    # Add labels and title
    ax.set_xlabel('Number of Candidate Outputs (n)', fontsize=FONTSIZE, fontname=font)
    ax.set_ylabel(f'{dataset_name.upper()} Accuracy (%)', fontsize=FONTSIZE, fontname=font)

    # Add legend
    ax.legend(
        fontsize=FONTSIZE - 2,
        frameon=False,
        loc='upper left',
        prop={'family': ['serif', font]}
    )

    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Save plot
    plt.tight_layout(pad=1.05)
    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')

    # Also save as PDF
    pdf_filepath = save_filepath.replace('.jpg', '.pdf').replace('.png', '.pdf')
    plt.savefig(pdf_filepath, dpi=300, bbox_inches='tight')

    print(f"Saved solution scaling plot to: {save_filepath}")
    plt.close()
