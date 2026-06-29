# data_pipeline.py
import pdfplumber
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_tender_pdf(pdf_path: str) -> dict:
    """
    Extracts text from a tender PDF, page by page.

    Why pdfplumber over PyPDF2?
    pdfplumber preserves table layouts and spacing better.
    Government PDFs have tables with financial thresholds —
    if those get mangled, the LLM in Component B misreads them.

    Why page-by-page dict and not just one big string?
    Compliance clauses are on specific pages. Storing by page
    lets us tell the LLM "look at pages 3-5" for eligibility,
    rather than dumping 40 pages of boilerplate at it.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"Tender PDF not found: {pdf_path}")

    result = {
        "file_name":         path.name,
        "pages":             {},
        "full_text":         "",
        "page_count":        0,
        "extraction_errors": []
    }

    try:
        with pdfplumber.open(path) as pdf:
            result["page_count"] = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        result["pages"][page_num] = text.strip()
                    else:
                        # Scanned image page — no extractable text
                        logger.warning(f"Page {page_num} has no text (scanned?): {path.name}")
                        result["extraction_errors"].append(f"Page {page_num}: no text")
                except Exception as e:
                    logger.error(f"Page {page_num} extraction failed: {e}")
                    result["extraction_errors"].append(f"Page {page_num}: {str(e)}")

            # Join all pages into one string for LLM consumption in Component B
            result["full_text"] = "\n\n".join(result["pages"].values())
            logger.info(f"Loaded {result['page_count']} pages from {path.name} "
                        f"({len(result['full_text'])} chars)")

    except Exception as e:
        logger.error(f"Cannot open PDF {path.name}: {e}")
        raise

    return result


def load_all_tenders(tender_dir: str = "data/tenders") -> dict:
    """
    Loads every PDF in the tenders folder.
    Returns a dict keyed by filename so the UI can let user pick one.

    Example output:
    {
        "tender_1.pdf": {"file_name": ..., "pages": ..., "full_text": ...},
        "tender_2.pdf": {...}
    }
    """
    folder = Path(tender_dir)

    if not folder.exists():
        raise FileNotFoundError(f"Tenders folder not found: {tender_dir}")

    pdfs = list(folder.glob("*.pdf"))

    if not pdfs:
        raise ValueError(f"No PDFs found in {tender_dir}. Add your downloaded tender PDFs there.")

    tenders = {}
    for pdf_path in pdfs:
        try:
            tenders[pdf_path.name] = load_tender_pdf(str(pdf_path))
            logger.info(f"Successfully loaded: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Skipping {pdf_path.name}: {e}")

    logger.info(f"Total tenders loaded: {len(tenders)}")
    return tenders


def load_aoc_data(csv_path: str = "data/bid_results/it_hardware_aoc.csv") -> pd.DataFrame:
    """
    Loads the historical AOC CSV that the scraper produced.
    Validates that essential columns exist before returning.

    Why validate columns here?
    If the scraper had issues and saved an incomplete CSV,
    we want a clear error NOW — not a cryptic KeyError
    deep inside the pricing model later.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"AOC data not found at {csv_path}. Run scraper.py first."
        )

    df = pd.read_csv(csv_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} raw AOC records")
    before = len(df)
    df = df.drop_duplicates(subset=["gem_bid_id"])
    logger.info(f"Removed {before - len(df)} duplicate records")
    df["turnover_required_lakhs"] = df["turnover_required_lakhs"].fillna(0)
    df["experience_required_years"] = df["experience_required_years"].fillna(0)

    numeric_cols = [
        "l1_price",
        "l2_price",
        "l3_price",
        "num_bidders",
        "price_spread_pct",
        "bid_validity_days"
    ]

    date_cols = [
        "bid_start",
        "bid_end",
        "bid_opening",
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col],
            format="%d-%m-%Y %H:%M:%S",
            errors="coerce",
        )

    df["bid_duration_hours"] = (
        df["bid_end"] - df["bid_start"]
    ).dt.total_seconds() / 3600

    df["opening_delay_minutes"] = (
        df["bid_opening"] - df["bid_end"]
    ).dt.total_seconds() / 60

    df["start_month"] = df["bid_start"].dt.month
    df["start_weekday"] = df["bid_start"].dt.weekday

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # These columns are the minimum we need for Component C
    required_columns = ["l1_price", "num_bidders"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"AOC CSV is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Drop rows with no L1 price — unusable for ML
    before = len(df)
    df = df.dropna(subset=["l1_price"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing l1_price")

    logger.info(f"Clean AOC dataset: {len(df)} usable records")
    return df


class DataPipeline:
    """
    Single entry point that loads everything.
    All other components import this class and call .run()
    instead of calling individual functions themselves.

    This is the 'orchestrator' pattern — one object owns
    all the data loading so nothing is loaded twice.
    """

    def __init__(
        self,
        tender_dir: str = "data/tenders",
        awards_csv: str = "data/bid_results/it_hardware_aoc.csv"
    ):
        self.tender_dir = tender_dir
        self.awards_csv = awards_csv
        self.tenders    = {}       # populated by run()
        self.aoc_df     = None     # populated by run()

    def run(self):
        logger.info("=== DataPipeline starting ===")
        self.tenders = load_all_tenders(self.tender_dir)
        self.aoc_df  = load_aoc_data(self.awards_csv)
        logger.info(
            f"Pipeline ready: {len(self.tenders)} tenders, "
            f"{len(self.aoc_df)} AOC records"
        )
        return self   # allows chaining: pipeline = DataPipeline().run()

    def get_tender_names(self) -> list[str]:
        """Returns list of tender filenames for UI dropdown."""
        return list(self.tenders.keys())

    def get_tender_text(self, filename: str) -> str:
        """Returns full extracted text of a specific tender."""
        if filename not in self.tenders:
            raise KeyError(f"Tender '{filename}' not loaded")
        return self.tenders[filename]["full_text"]
 
    def get_statistics(self):
        df = self.aoc_df

        return {
        "records": len(df),
        "average_l1": df["l1_price"].mean(),
        "median_l1": df["l1_price"].median(),
        "average_bidders": df["num_bidders"].mean(),
    }

if __name__ == "__main__":
    pipeline = DataPipeline().run()

    print(pipeline.get_tender_names())
    print(pipeline.aoc_df.head())