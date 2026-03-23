"""
scrape_farfetch.py  —  Farfetch jacket image scraper
=====================================================
Mirrors the dual-strategy architecture of scrape_zalando.py.

Farfetch uses Akamai bot detection on direct requests, so two strategies
are provided:

STRATEGY A — Bing Image Search (default, headless)
  Searches Bing for "farfetch women {category} jacket" and extracts
  cdn-images.farfetch-contents.com CDN image URLs from the results page
  HTML. The CDN itself has no bot protection — it is a plain file server.
  Works fully headless.

STRATEGY B — Headed browser (--headed flag)
  Opens a visible Chromium window. Farfetch rarely blocks headed browsers.
  Uses network interception to capture CDN URLs as the page loads products
  lazily via infinite scroll.  Use this if Bing yields too few URLs.

Image URL pattern
-----------------
  https://cdn-images.farfetch-contents.com/{AA}/{BB}/{CC}/{DD}/{PRODUCT_ID}_{SHOT_ID}_{SIZE}.jpg
  Size token can be 300, 480, 600, 1000 — we normalise to 1000.

Usage
-----
    pip install playwright pillow httpx
    playwright install chromium

    # Quick test — Bing strategy, one category
    python src/scrape_farfetch.py --category bomber --max-per-class 30 --dst "E:/fashion-data/01-RAW/jackets_women_farfetch"

    # Full scrape — all categories via Bing
    python src/scrape_farfetch.py --dst "E:/fashion-data/01-RAW/jackets_women_farfetch"

    # Headed mode — opens visible browser window
    python src/scrape_farfetch.py --headed --category biker --max-per-class 50 --dst "E:/fashion-data/01-RAW/jackets_women_farfetch"

    # Regenerate CSV from already-downloaded images
    python src/scrape_farfetch.py --csv-only --dst "E:/fashion-data/01-RAW/jackets_women_farfetch"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import random
import re
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Category configuration
# ─────────────────────────────────────────────────────────────────────────────

# Bing search queries — multiple per category for diversity
CATEGORY_QUERIES: dict[str, list[str]] = {
    "feather_heels": ["feather trim heels",
                    "fur mules", 
                    "fringe boots", 
                    "pom pom sandals"     
    ],
    "beaded_heels": ["sequin heels",
                    "beaded sandals", 
                    "embroidered mules", 
                    "embroidered shoes"     
    ],
    "zipper_boots": ["decorative zipper boots",
                    "decorative zipper shoes",
                    "fringe boots", 
                    "fringe shoes", 
    ],
    "biker": [
        "farfetch women biker jacket leather product",
        "farfetch womens moto jacket faux leather shop",
        "farfetch women leather biker jacket black product",
        "farfetch uk biker jacket women fashion",
    ],
    "blazer": [
        "farfetch women blazer jacket product photo",
        "farfetch womens tailored blazer shop",
        "farfetch women oversized blazer product",
        "farfetch uk blazer women fashion",
    ],
    "bomber": [
        "farfetch women bomber jacket product photo",
        "farfetch womens satin bomber jacket shop",
        "farfetch women varsity bomber jacket",
        "farfetch uk bomber jacket women product",
    ],
    "fur_jacket": [
        "farfetch women faux fur jacket product photo",
        "farfetch womens shearling jacket shop",
        "farfetch women teddy coat jacket product",
        "farfetch uk fur jacket women",
    ],
    "parka": [
        "farfetch women parka jacket product photo",
        "farfetch womens hooded parka shop",
        "farfetch women long parka coat product",
        "farfetch uk parka women fashion",
    ],
    "bolero": [
        "farfetch women bolero jacket product photo",
        "farfetch womens hooded bolero shop",
        "farfetch uk bolero women fashion",
    ],
    "trench_coat": [
        "farfetch men trench coats product photo",
        "farfetch womens trench coats raincoats shop",
        "farfetch burberry trench coats raincoats women fashion",
        "farfetch burberry trench coats raincoats men fashion",
    ],
    "cropped": [
        "farfetch women cropped jackets product photo",
        "farfetch women cropped jackets shop",
    ],  
    "fringe_jacket": [
        "farfetch women fringe jacket product photo",
        "farfetch women fringed jacket shop",
        "farfetch men fringe jacket product photo",
        "farfetch men fringed jacket shop",
    ],
    "feather_jacket": [
        "farfetch women feather-trim feather jacket product photo",
        "farfetch women feather-trim feather jacket shop",
    ],      
    "floral_jacket": [
        "farfetch women floral appliqué flower-appliqué jacket product photo",
        "farfetch women floral appliqué flower-appliqué jacket shop",
    ],   
    "belted": [
        "farfetch women belt-waist jacket product photo",
        "farfetch women belt-waist jacket shop",
        "farfetch men belt-waist jacket product photo",
        "farfetch men belt-waist jacket shop",

    ],          
    
}

# Direct Farfetch category pages — used in headed mode
# URLs derived from the farfetch.com taxonomy (verified from live site)
CATEGORY_URLS: dict[str, list[str]] = {
    "biker": [
        "https://www.farfetch.com/shopping/women/biker-jackets-1/items.aspx",
        "https://www.farfetch.com/shopping/women/leather-jackets-1/items.aspx",
    ],
    "blazer": [
        "https://www.farfetch.com/shopping/women/blazers-1/items.aspx",
        "https://www.farfetch.com/shopping/women/suit-jackets-1/items.aspx",
    ],
    "bomber": [
        "https://www.farfetch.com/shopping/women/sport-jacket-1/items.aspx",
        "https://www.farfetch.com/shopping/women/varsity-jackets-1/items.aspx",
    ],
    "fur_jacket": [
        "https://www.farfetch.com/shopping/women/fur-jackets-1/items.aspx",
        "https://www.farfetch.com/shopping/women/shearling-jackets-1/items.aspx",
    ],
    "parka": [
        "https://www.farfetch.com/shopping/women/parka-jackets-1/items.aspx",
        "https://www.farfetch.com/shopping/women/down-jackets-1/items.aspx",
    ],
    "trench_coat": [
        "https://www.farfetch.com/shopping/men/trench-coats-2/items.aspx",
        "https://www.farfetch.com/it/shopping/women/trench-raincoat-1/items.aspx"
    ],
    "cropped": [
        "https://www.farfetch.com/it/shopping/women/cropped-jackets-1/items.aspx",
    ],
    "fringe_jacket": [
        "https://www.farfetch.com/it/shopping/women/search/items.aspx",
        "https://www.farfetch.com/it/shopping/men/search/items.aspx",
    ],
    "feather_jacket": [
        "https://www.farfetch.com/it/shopping/women/search/items.aspx",
    ],
    "floral_jacket": [
        "https://www.farfetch.com/it/shopping/women/search/items.aspx",
    ],
    "belted": [
        "https://www.farfetch.com/it/shopping/women/search/items.aspx",
        "https://www.farfetch.com/it/shopping/men/search/items.aspx",
    ],
    "feather_heels": [
        "https://www.farfetch.com/it/shopping/women/search/items.aspx"
        
    ],
    "beaded_heels": [
        "https://www.farfetch.com/it/shopping/women/search/items.aspx"
        
    ],
    "zipper_boots": [
        "https://www.farfetch.com/it/shopping/women/search/items.aspx",
        "https://www.farfetch.com/it/shopping/men/search/items.aspx"
        
    ],
}

# Maps internal folder key to CSV label (must match conf/category/jackets.yaml)
LABEL_MAP = {
    "biker":      "biker",
    "blazer":     "blazer",
    "bomber":     "bomber",
    "fur_jacket": "fur jacket",
    "parka":      "parka",
    "bolero":      "bolero",
    "trench_coat": "trench coat",
    "cropped": "cropped",
    "fringe_jacket": "fringe jacket",
    "belted": "belted",
    "feather_heels": "feather heels",
    "beaded_heels": "beaded heels",
        "zipper_boots": "zipper boots",
}

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Request image at 1000 px wide — highest standard CDN size
IMG_SIZE         = 1000
MIN_SIDE         = 300          # minimum acceptable image dimension
GOTO_TIMEOUT     = 60_000       # ms
DOWNLOAD_TIMEOUT = 20           # seconds
MIN_DELAY        = 0.4          # seconds between downloads
MAX_DELAY        = 1.0
SCROLL_PAUSE     = 2.5          # seconds per scroll step in headed mode
MAX_SCROLLS      = 40

# Regex for Farfetch CDN images
# Pattern: cdn-images.farfetch-contents.com/AA/BB/CC/DD/PRODUCTID_SHOTID_SIZE.jpg
FARFETCH_CDN_RE = re.compile(
    r"https?://cdn-images\.farfetch-contents\.com"
    r"/\d+/\d+/\d+/\d+"          # path segments derived from product ID
    r"/\d+_\d+_\d+\.jpg"         # productId_shotId_size.jpg
)

IMG_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.farfetch.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

STEALTH_JS = """
() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
    Object.defineProperty(navigator, 'plugins',   { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-GB', 'en'] });
    window.chrome = { runtime: {} };
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stem(category: str, idx: int) -> str:
    return f"{category}_ff_{idx:04d}"


def _normalise_url(url: str) -> str:
    """Replace the size token in the filename with IMG_SIZE."""
    # Replace the trailing _NNN.jpg size token with _1000.jpg
    normalised = re.sub(r"_\d+\.jpg$", f"_{IMG_SIZE}.jpg", url)
    # Strip any query string
    return re.sub(r"\?.*$", "", normalised)


def _base_url(url: str) -> str:
    """Return the canonical base (product + shot, no size, no query)."""
    # Strip size token so different sizes of the same shot are deduplicated
    return re.sub(r"_\d+\.jpg$", "", re.sub(r"\?.*$", "", url))


def _image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _passes_quality(data: bytes) -> tuple[bool, str]:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if w < MIN_SIDE or h < MIN_SIDE:
            return False, f"too_small({w}x{h})"
        if w / h > 2.5 or w / h < 0.25:
            return False, "bad_aspect"
        return True, ""
    except Exception as e:
        return False, "decode_error"


async def _download(url: str) -> bytes | None:
    import httpx
    try:
        async with httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT, headers=IMG_HEADERS, follow_redirects=True
        ) as client:
            r = await client.get(url)
        return r.content if r.status_code == 200 else None
    except Exception as e:
        print(f"    DL error: {e}")
        return None


def _extract_farfetch_urls(html: str) -> list[str]:
    """Extract unique normalised Farfetch CDN URLs from raw HTML."""
    raw = FARFETCH_CDN_RE.findall(html)
    seen_bases: set[str] = set()
    result = []
    for u in raw:
        base = _base_url(u)
        if base not in seen_bases:
            seen_bases.add(base)
            result.append(_normalise_url(u))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A: Bing Image Search  (headless)
# ─────────────────────────────────────────────────────────────────────────────

async def collect_via_bing(
    context,
    queries: list[str],
    max_urls: int,
) -> list[str]:
    collected_bases: set[str] = set()
    result_urls: list[str] = []

    for query in queries:
        if len(result_urls) >= max_urls:
            break

        bing_url = (
            "https://www.bing.com/images/search"
            f"?q={query.replace(' ', '+')}&form=HDRSC2&first=1"
        )
        print(f"    Bing: {query}")

        page = await context.new_page()
        await page.add_init_script(STEALTH_JS)

        try:
            await page.goto(bing_url, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(2)

            # Scroll and expand results
            for _ in range(10):
                await page.evaluate("window.scrollBy(0, window.innerHeight * 1.5)")
                await asyncio.sleep(1.0)
                try:
                    btn = await page.query_selector("a.btn_seemore")
                    if btn:
                        await btn.click()
                        await asyncio.sleep(1.5)
                except Exception:
                    pass

            content = await page.content()
            urls = _extract_farfetch_urls(content)

            new = []
            for u in urls:
                base = _base_url(u)
                if base not in collected_bases:
                    collected_bases.add(base)
                    new.append(u)

            result_urls.extend(new)
            print(f"      +{len(new)} URLs  (total: {len(result_urls)})")

        except Exception as e:
            print(f"      Bing error: {e}")
        finally:
            try:
                await page.close()
            except Exception:
                pass

        await asyncio.sleep(random.uniform(1.5, 3.0))

    return result_urls[:max_urls]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B: Headed Farfetch direct
# ─────────────────────────────────────────────────────────────────────────────

async def collect_via_farfetch_headed(
    context,
    urls: list[str],
    max_urls: int,
) -> list[str]:
    collected_bases: set[str] = set()
    intercepted: list[str] = []

    def on_request(req):
        url = req.url
        if FARFETCH_CDN_RE.match(url):
            base = _base_url(url)
            if base not in collected_bases:
                collected_bases.add(base)
                intercepted.append(_normalise_url(url))

    for page_url in urls:
        if len(collected_bases) >= max_urls:
            break

        print(f"    Farfetch: {page_url}")
        page = await context.new_page()
        await page.add_init_script(STEALTH_JS)
        page.on("request", on_request)

        try:
            await page.goto(page_url, wait_until="commit", timeout=GOTO_TIMEOUT)
            await asyncio.sleep(4)

            # Abort if blocked
            try:
                title = await page.title()
                print(f"    Title: {title[:60]}")
                if any(k in title.lower() for k in ("denied", "captcha", "robot", "access")):
                    print("    BLOCKED — skipping this URL")
                    await page.close()
                    continue
            except Exception:
                pass

            # Scroll to trigger lazy-loaded product images
            vh = (page.viewport_size or {}).get("height", 900)
            for _ in range(MAX_SCROLLS):
                if len(collected_bases) >= max_urls:
                    break
                await page.evaluate(f"window.scrollBy(0, {int(vh * 0.85)})")
                await asyncio.sleep(SCROLL_PAUSE)
                try:
                    at_bottom = await page.evaluate(
                        "window.scrollY + window.innerHeight >= document.body.scrollHeight - 400"
                    )
                    if at_bottom:
                        break
                except Exception:
                    break

            print(f"    Intercepted {len(collected_bases)} URLs")

        except Exception as e:
            print(f"    Error: {e}")
        finally:
            try:
                await page.close()
            except Exception:
                pass

        await asyncio.sleep(random.uniform(2, 4))

    return intercepted[:max_urls]


# ─────────────────────────────────────────────────────────────────────────────
# Download + quality-filter pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def download_and_filter(
    img_urls: list[str],
    category: str,
    dst_folder: Path,
    max_images: int,
    seen_hashes: set[str],
    start_idx: int,
) -> list[dict]:
    dst_folder.mkdir(parents=True, exist_ok=True)
    collected = []
    idx = start_idx

    for img_url in img_urls:
        if len(collected) >= max_images:
            break
        await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        data = await _download(img_url)
        if data is None:
            continue

        h = _image_hash(data)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        ok, reason = _passes_quality(data)
        if not ok:
            print(f"    SKIP ({reason})")
            continue

        stem = _stem(category, idx)
        (dst_folder / f"{stem}.jpg").write_bytes(data)
        collected.append({"stem": stem, "category": category, "source": img_url})
        idx += 1
        print(f"    [{len(collected):3d}/{max_images}] {stem}.jpg")

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# Per-category orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def scrape_category(
    context,
    category: str,
    dst_folder: Path,
    max_images: int,
    headed: bool,
) -> list[dict]:
    print(f"\n{'='*55}")
    print(f"  Category: {category}  (target: {max_images})")
    print(f"{'='*55}")

    seen_hashes: set[str] = set()

    if headed:
        print("\n  [Mode: headed Farfetch direct]")
        img_urls = await collect_via_farfetch_headed(
            context,
            urls=CATEGORY_URLS[category],
            max_urls=max_images * 3,
        )
    else:
        print("\n  [Mode: Bing image search]")
        img_urls = await collect_via_bing(
            context,
            queries=CATEGORY_QUERIES[category],
            max_urls=max_images * 3,
        )

    print(f"\n  URLs to download: {len(img_urls)}")
    if not img_urls:
        print(f"  WARNING: no URLs found for '{category}'")
        return []

    print(f"\n  Downloading...")
    records = await download_and_filter(
        img_urls=img_urls,
        category=category,
        dst_folder=dst_folder,
        max_images=max_images,
        seen_hashes=seen_hashes,
        start_idx=1,
    )

    print(f"\n  [{category}] FINAL: {len(records)} images")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_scraper(
    categories: list[str],
    dst: Path,
    max_per_class: int,
    headed: bool,
) -> list[dict]:
    from playwright.async_api import async_playwright

    all_records: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not headed,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
            slow_mo=50 if headed else 0,
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-GB",
            timezone_id="Europe/London",
        )

        for cat in categories:
            records = await scrape_category(
                context=context,
                category=cat,
                dst_folder=dst / cat,
                max_images=max_per_class,
                headed=headed,
            )
            all_records.extend(records)
            await asyncio.sleep(random.uniform(2, 5))

        await browser.close()

    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# CSV utilities
# ─────────────────────────────────────────────────────────────────────────────

def generate_csv(records: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "Class", "source"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "name":   r["stem"],
                "Class":  LABEL_MAP.get(r["category"], r["category"]),
                "source": r.get("source", ""),
            })
    print(f"\nCSV → {csv_path}  ({len(records)} rows)")
    dist = Counter(LABEL_MAP.get(r["category"], r["category"]) for r in records)
    for cls, cnt in sorted(dist.items()):
        print(f"  {cls:<15} {cnt}")


def csv_from_existing(dst: Path, csv_path: Path) -> None:
    """Regenerate CSV by scanning an already-downloaded image tree."""
    records = []
    for cat_folder in sorted(dst.iterdir()):
        if not cat_folder.is_dir() or cat_folder.name not in LABEL_MAP:
            continue
        for img_file in sorted(cat_folder.glob("*.jpg")):
            records.append({
                "stem": img_file.stem, "category": cat_folder.name, "source": ""
            })
    generate_csv(records, csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Farfetch women's jacket scraper")
    p.add_argument(
        "--dst", type=Path,
        default=Path("E:/fashion-data/01-RAW/jackets_women_farfetch"),
        help="Root output folder (subfolders created per category)",
    )
    p.add_argument(
        "--csv", type=Path, default=None,
        help="Output CSV path (default: dst.parent/labels_jackets_farfetch.csv)",
    )
    p.add_argument(
        "--category", choices=list(CATEGORY_QUERIES.keys()), default=None,
        help="Scrape a single category only (default: all)",
    )
    p.add_argument(
        "--max-per-class", type=int, default=350,
        help="Maximum images to download per category (default: 350)",
    )
    p.add_argument(
        "--headed", action="store_true",
        help="Open a visible Chromium window and scrape Farfetch directly",
    )
    p.add_argument(
        "--csv-only", action="store_true",
        help="Regenerate labels CSV from already-downloaded images (no scraping)",
    )
    args = p.parse_args()

    csv_path   = args.csv or args.dst.parent / "labels_jackets_farfetch.csv"
    categories = [args.category] if args.category else list(CATEGORY_QUERIES.keys())

    if args.csv_only:
        csv_from_existing(args.dst, csv_path)
        return

    mode = "headed Farfetch" if args.headed else "Bing image search"
    print(f"Mode       : {mode}")
    print(f"Categories : {categories}")
    print(f"Output     : {args.dst}")
    print(f"Max/class  : {args.max_per_class}\n")

    records = asyncio.run(run_scraper(
        categories=categories,
        dst=args.dst,
        max_per_class=args.max_per_class,
        headed=args.headed,
    ))
    generate_csv(records, csv_path)


if __name__ == "__main__":
    main()
