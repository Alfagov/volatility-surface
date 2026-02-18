import argparse
from pathlib import Path
from typing import Iterable, List
import numpy as np
import pandas as pd
import wrds

def interpolate_rates(df: pd.DataFrame, rate_cols: List[str], tenors: List[int]) -> pd.DataFrame:
    """
    Linearly interpolates risk-free rates based on option time-to-maturity (T).

    Args:
        df: DataFrame containing options data and joined rate columns.
        rate_cols: List of column names for rates (e.g., ['dgs1mo', ...]).
        tenors: List of days corresponding to rate_cols (e.g., [30, 91, ...]).
    """
    # Create arrays for tenors and rates
    # We want to interpolate 'r' for a given 'T' (days)

    # 1. Handle NaN rates (forward fill just in case, though rates are usually daily)
    df[rate_cols] = df[rate_cols].ffill()

    # 2. Convert rates from percentage (e.g., 5.25) to decimal (0.0525)
    # FRB data is usually in percent. Check your source; typical WRDS is percent.
    rate_values = df[rate_cols].values / 100.0

    # 3. Interpolation Logic
    # We need to find the specific rate for the specific T of each row.
    # Since scipy.interpolate is slow on large DFs row-by-row, we implement a numpy lookup.

    T_days = df['T'].values
    n_rows = len(df)
    interpolated_r = np.zeros(n_rows)

    # Convert tenors to numpy array
    tenors_arr = np.array(tenors)

    # Iterate through tenors to find brackets [t_low, t_high]
    # This is much faster than row-by-row apply

    # Initialize with the shortest rate (for T < first tenor)
    interpolated_r[:] = rate_values[:, 0]

    for i in range(len(tenors) - 1):
        t_low = tenors[i]
        t_high = tenors[i + 1]

        # Mask for rows where T is between these two tenors
        mask = (T_days >= t_low) & (T_days < t_high)

        if np.any(mask):
            r_low = rate_values[mask, i]
            r_high = rate_values[mask, i + 1]
            dt = T_days[mask]

            # Linear Interpolation formula: y = y0 + (y1-y0) * (x-x0)/(x1-x0)
            fraction = (dt - t_low) / (t_high - t_low)
            interpolated_r[mask] = r_low + (r_high - r_low) * fraction

    # Handle T > max tenor (use the longest rate available)
    mask_long = T_days >= tenors[-1]
    if np.any(mask_long):
        interpolated_r[mask_long] = rate_values[mask_long, -1]

    df['rate'] = interpolated_r
    return df

def download_option_data(secid: str = "108105", years: Iterable[int] = (), option_type: str = "C") -> None:
    db = wrds.Connection()

    tenor_days = [30, 91, 182, 365, 730, 1095, 1825]
    rate_columns = ["dgs1mo", "dgs3mo", "dgs6mo", "dgs1", "dgs2", "dgs3", "dgs5"]

    for year in years:
        print(f"Connecting to WRDS to fetch {secid} for year {year}...")

        print(f"Fetching Yield Curve for {year}...")
        rates_df = db.raw_sql(f"""
                        SELECT 
                            date,
                            dgs1mo, dgs3mo, dgs6mo, dgs1, dgs2, dgs3, dgs5
                        FROM 
                            frb.rates_daily
                        WHERE 
                            date >= '{year}-01-01' AND date <= '{year}-12-31'
                    """)
        rates_df["date"] = pd.to_datetime(rates_df["date"])

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
                        d.rate / 100.0 as dividend_yield,
                        c.vix,
                        v.hv_10,
                        v.hv_14,
                        v.hv_30,
                        v.hv_60,
                        v.hv_91,
                        f.sofr as sofr
                    FROM
                        optionm.opprcd{year} as o
                    LEFT JOIN
                        optionm.secprd as s
                        ON o.date = s.date AND o.secid = s.secid
                    LEFT JOIN
                        optionm.idxdvd as d
                        ON o.secid = d.secid 
                        AND o.date = d.date 
                        AND o.exdate = d.expiration
                    JOIN
                        cboe.cboe as c
                        ON c.date = s.date
                    JOIN (
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
                    LEFT JOIN
                        frb.rates_daily as f
                        ON o.date = f.date
                    WHERE
                        s.secid = '{secid}'
                        AND o.cp_flag = '{option_type}'
                        AND o.best_bid > 0.1 
                        AND o.volume > 1
                        AND inf.exercise_style = 'E'
                    """

        try:
            df = db.raw_sql(sql_query)
        except Exception as exc:
            print(f"Query failed for: {exc}")
            print("If needed, retry with a different --table-prefix.")
            continue

        print("Processing data...")
        df["date"] = pd.to_datetime(df["date"])
        df["exdate"] = pd.to_datetime(df["exdate"])
        df["T"] = (df["exdate"] - df["date"]).dt.days
        df = df[df["T"] > 1]

        df = pd.merge(df, rates_df, on="date", how="left")

        df = interpolate_rates(df, rate_columns, tenor_days)

        final_df = df[
            [
                "date",
                "spot_price",
                "strike",
                "T",
                "vix",
                "hv_10",
                "hv_14",
                "hv_30",
                "hv_60",
                "hv_91",
                "option_price",
                "impl_volatility",
                "cp_flag",
                "exercise_style",
                "dividend_yield",
                "rate"
            ]
        ].copy()
        final_df.columns = [
            "date",
            "S",
            "K",
            "T",
            "vix",
            "hv_10",
            "hv_14",
            "hv_30",
            "hv_60",
            "hv_91",
            "Price",
            "Impl_Vol",
            "cp_flag",
            "exercise_style",
            "dividend_yield",
            "rate"
        ]
        final_df = final_df.sort_values(["date", "K", "T"])

        path = Path(f"./data/{secid.lower()}/{year}_{option_type}_options_data.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(path, index=False)
        print(f"Downloaded {len(final_df)} rows. Saved to {path}")

    db.close()


def download_vol_surface_data(
    secid: str = "108105",
    years: Iterable[int] = (),
    option_type: str = "C",
    table_prefix: str = "optionm.vsurfd",
) -> None:
    """
    Download delta-based WRDS volatility surfaces.
    Expected WRDS columns include:
    cp_flag, date, days, delta, dispersion, impl_premium, impl_strike, impl_volatility, secid
    """
    db = wrds.Connection()

    for year in years:
        table_name = f"{table_prefix}{year}"
        print(f"Connecting to WRDS to fetch volatility surface from {table_name}...")

        sql_query = f"""
            SELECT
                vs.cp_flag,
                vs.date,
                vs.days,
                vs.delta,
                vs.dispersion,
                vs.impl_premium,
                vs.impl_strike,
                vs.impl_volatility,
                vs.secid,
                s.close as spot_price,
                c.vix,
                h.hv_10,
                h.hv_14,
                h.hv_30,
                h.hv_60,
                h.hv_91
            FROM
                {table_name} as vs
            LEFT JOIN
                optionm.secprd as s
                ON vs.date = s.date AND vs.secid = s.secid
            LEFT JOIN
                cboe.cboe as c
                ON vs.date = c.date
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
            ) as h
                ON o.date = v.date AND o.secid = v.secid
            WHERE
                vs.secid = '{secid}'
                AND vs.cp_flag = '{option_type}'
                AND vs.impl_volatility IS NOT NULL
                AND vs.impl_strike > 0
                AND vs.days > 0
        """

        df = db.raw_sql(sql_query)
        if df.empty:
            print(f"No rows returned for {table_name}.")
            continue

        df["date"] = pd.to_datetime(df["date"])
        if "spot_price" in df.columns:
            df["moneyness"] = df["spot_price"] / df["impl_strike"]

        final_df = df.sort_values(["date", "days", "delta"]).reset_index(drop=True)
        path = Path(f"./data/{secid.lower()}_surface/{year}_{option_type}_vol_surface.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(path, index=False)
        print(f"Downloaded {len(final_df)} rows. Saved to {path}")

    db.close()


def _parse_years(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="WRDS downloader for options and volatility surfaces.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_options = subparsers.add_parser("options", help="Download option chain-level records.")
    p_options.add_argument("--secid", default="108105")
    p_options.add_argument("--years", required=True, help="Comma-separated list, e.g. 2023,2024")
    p_options.add_argument("--cp-flag", default="C", choices=["C", "P"])

    p_surface = subparsers.add_parser("surface", help="Download WRDS delta-based volatility surface records.")
    p_surface.add_argument("--secid", default="108105")
    p_surface.add_argument("--years", required=True, help="Comma-separated list, e.g. 2023,2024")
    p_surface.add_argument("--cp-flag", default="C", choices=["C", "P"])
    p_surface.add_argument(
        "--table-prefix",
        default="optionm.vsurfd",
        help="WRDS table prefix; final table is <prefix><year>.",
    )

    args = parser.parse_args()
    years = _parse_years(args.years)

    if args.mode == "options":
        download_option_data(secid=args.secid, years=years, option_type=args.cp_flag)
        return

    download_vol_surface_data(
        secid=args.secid,
        years=years,
        option_type=args.cp_flag,
        table_prefix=args.table_prefix,
    )


if __name__ == "__main__":
    main()
