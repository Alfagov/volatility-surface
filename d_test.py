from pathlib import Path
from typing import List

import pandas as pd
import wrds


def download_option_data(secid="108105", years: List[int] = None, option_type='C'):
    db = wrds.Connection()

    for year in years:
        print(f"Connecting to WRDS to fetch {secid} for year {year}...")

        sql_query = f"""
            SELECT
                o.date,
                o.exdate,
                o.cp_flag,
                o.strike_price / 1000.0 as strike,
                (o.best_bid + o.best_offer) / 2.0 as option_price,
                o.impl_volatility,
                s.close as spot_price,
                inf.exercise_style,
                c.vix,
                v.hv_10,
                v.hv_14,
                v.hv_30,
                v.hv_60,
                v.hv_91
            FROM
                optionm.opprcd{year} as o
            JOIN
                optionm.secprd as s
                ON o.date = s.date AND o.secid = s.secid
            JOIN
                cboe.cboe as c
                ON c.date = s.date
            LEFT JOIN (
                SELECT 
                    date, 
                    secid,
                    MAX(CASE WHEN days = 10 THEN volatility END) as hv_10,
                    MAX(CASE WHEN days = 14 THEN volatility END) as hv_14,
                    MAX(CASE WHEN days = 30 THEN volatility END) as hv_30,
                    MAX(CASE WHEN days = 60 THEN volatility END) as hv_60,
                    MAX(CASE WHEN days = 91 THEN volatility END) as hv_91
                FROM 
                    optionm.hvold{year}
                WHERE 
                    secid = '{secid}'
                GROUP BY 
                    date, secid
            ) as v
                ON o.date = v.date AND o.secid = v.secid
            JOIN 
                optionm.opinfd as inf
                ON o.secid = inf.secid
            WHERE
                s.secid = '{secid}'
                AND o.cp_flag = '{option_type}'
                AND o.best_bid > 0.1 
                AND o.volume > 1
                AND o.impl_volatility IS NOT NULL
                AND inf.exercise_style = 'E'
            """

        df = db.raw_sql(sql_query)

        print("Processing data... ")

        df['date'] = pd.to_datetime(df['date'])
        df['exdate'] = pd.to_datetime(df['exdate'])

        df['T'] = (df['exdate'] - df['date']).dt.days

        df = df[(df['T'] > 1)]

        final_df = df[['date', 'spot_price', 'strike', 'T', 'vix', 'hv_10', 'hv_14', 'hv_30', 'hv_60', 'hv_91', 'option_price', 'impl_volatility', 'cp_flag', 'exercise_style']].copy()
        final_df.columns = ['date', 'S', 'K', 'T', 'vix', 'hv_10', 'hv_14', 'hv_30', 'hv_60', 'hv_91', 'Price', 'Impl_Vol', 'cp_flag', 'exercise_style']

        final_df = final_df.sort_values(['date', 'K', 'T'])

        print(f"Downloaded {len(final_df)} rows.")

        path_str = f"./data/{secid.lower()}/{year}_{option_type}_options_data.csv"
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

        final_df.to_csv(path_str, index=False)
        print(f"Saved to {path_str}")

    db.close()

if __name__ == "__main__":
    years = [2025]
    download_option_data(years=years)