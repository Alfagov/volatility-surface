import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import wrds


def download_option_data(secid: str = "108105", years: Iterable[int] = (), option_type: str = "C") -> None:
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
                        v.hv_91,
                        f.sofr as sofr
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
                    LEFT JOIN
                        frb.rates_daily as f
                        ON o.date = f.date
                    WHERE
                        s.secid = '{secid}'
                        AND o.cp_flag = '{option_type}'
                        AND o.best_bid > 0.1 
                        AND o.volume > 1
                        AND o.impl_volatility IS NOT NULL
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
