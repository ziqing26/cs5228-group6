"""Regenerate the report's descriptive EDA figures from all raw training rows."""
from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/figures'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/train.csv')
    dates = pd.to_datetime(data['RENT_APPROVAL_DATE'])
    monthly = data.assign(month=dates.dt.to_period('M').dt.to_timestamp()).groupby('month')['MONTHLY_RENT'].agg(['mean', 'median', 'count'])
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(5.2, 2.7), layout='constrained')
    counts = ax.twinx()
    counts.bar(monthly.index, monthly['count'], width=23, color='#cbd5e1', alpha=0.4)
    counts.set_ylabel('Records')
    counts.set_ylim(0, monthly['count'].max() * 3)
    ax.set_zorder(counts.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(monthly.index, monthly['mean'], label='Mean', color='#2563a6')
    ax.plot(monthly.index, monthly['median'], label='Median', color='#bd5d27')
    ax.set_ylabel('Monthly rent (SGD)')
    ax.set_xlabel('Approval month')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)
    fig.savefig(OUT / 'eda_temporal.pdf')
    plt.close(fig)

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
    main()
