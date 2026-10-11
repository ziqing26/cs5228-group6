"""Regenerate the report's descriptive EDA figures from all raw training rows."""
from pathlib import Path
import argparse
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/figures'


def main(temporal_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/train.csv')
    dates = pd.to_datetime(data['RENT_APPROVAL_DATE'])
    monthly = data.assign(month=dates.dt.to_period('M').dt.to_timestamp()).groupby('month')['MONTHLY_RENT'].agg(['mean', 'median', 'count'])
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.0), layout='constrained')
    ax.plot(monthly.index, monthly['mean'], label='Mean', color='#2563a6', linewidth=1.5)
    ax.plot(monthly.index, monthly['median'], label='Median', color='#bd5d27',
            linewidth=1.3, linestyle='--')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(monthly.index.min(), monthly.index.max())
    ax.set_ylabel('Monthly rent (SGD)')
    ax.set_xlabel('Approval month')
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(OUT / 'eda_temporal.pdf')
    fig.savefig(OUT / 'eda_temporal.png', dpi=240)
    plt.close(fig)
    if temporal_only:
        return

    def normalise(value):
        text = unicodedata.normalize('NFKC', str(value)).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'^(\d)\s*[- ]\s*room$', r'\1-room', text)

    order = ['1-room', '2-room', '3-room', '4-room', '5-room', 'executive']
    medians = data.assign(flat_type=data['FLAT_TYPE'].map(normalise)).groupby('flat_type')['MONTHLY_RENT'].median().reindex(order)
    if medians.isna().any():
        raise ValueError('Expected all six normalised flat types')
    fig, ax = plt.subplots(figsize=(5.2, 2.7), layout='constrained')
    ax.bar([x.title() for x in order], medians, color='#2563a6')
    ax.set_ylabel('Median monthly rent (SGD)')
    ax.set_xlabel('Normalised flat type')
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(OUT / 'eda_flat_type.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temporal-only', action='store_true',
                        help='Regenerate only the monthly-rent trend figure.')
    main(temporal_only=parser.parse_args().temporal_only)
