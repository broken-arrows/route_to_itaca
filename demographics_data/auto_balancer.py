#!/usr/bin/env python3
"""
Auto-balancer for demographic percentage data.
This script reads a CSV file containing demographic data with percentages,
normalizes the percentages within each province and demographic group,
and saves the balanced data back to a CSV file.
Note: it balances as percentages (0-100), not per-one normalization (0-1).
"""
import pandas as pd
import argparse

def balance_percentages(df, province_col, demographic_col, percentage_col):
    def _balance(x):
        s = x.sum()
        return x * 100 / s if s != 0 else x
    df[percentage_col] = (
        df.groupby([province_col, demographic_col])[percentage_col]
        .transform(_balance)
    )
    df[percentage_col] = df[percentage_col].round(2)
    return df

def main(input_file, output_file):
    df = pd.read_csv(input_file)
    if 'sample_size' in df.columns:
        df = df.drop(columns=['sample_size'])
    province_col = 'province'
    demographic_col = 'demographic_group'
    percentage_col = 'percentage'
    df = balance_percentages(df, province_col, demographic_col, percentage_col)
    df.to_csv(output_file, index=False)
    print(f"Balanced data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance demographic percentages in a CSV file.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="demographics_data/clean/ok_vote_intention_2012.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="demographics_data/clean/ok_vote_intention_2012.csv",
        help="Path to output CSV file"
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)

