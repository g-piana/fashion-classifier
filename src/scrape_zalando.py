"""
scrape_zalando.py  —  Zalando jacket image scraper  (v3)
=========================================================
Zalando uses CloudFront bot detection that terminates connections from
headless browsers. This version uses two strategies that bypass it:

STRATEGY A — Bing Image Search (default, headless)
  Search Bing for "zalando women {category} jacket" and extract
  ztat.net CDN image URLs from the results page HTML. The CDN has
  no bot protection — it is just a file server. Works headless.

STRATEGY B — Headed browser (--headed flag)
  Opens a visible Chromium window. Zalando rarely blocks headed
  browsers. Uses network interception to capture ztat.net URLs
  as Zalando loads them lazily. Use this if Bing yields too few URLs.

Usage
-----
    pip install playwright pillow httpx
    playwright install chromium

    # Quick test — Bing strategy, one category
    python src/scrape_zalando.py --category bomber --max-per-class 30 --dst "E:/fashion-data/01-RAW/jackets_women_zalando"

    # Full scrape — all categories via Bing
    python src/scrape_zalando.py --dst "E:/fashion-data/01-RAW/jackets_women_zalando"

    # Headed mode — opens visible browser window
    python src/scrape_zalando.py --headed --category biker --max-per-class 50 --dst "E:/fashion-data/01-RAW/jackets_women_zalando"

    # Regenerate CSV from downloaded images
    python src/scrape_zalando.py --csv-only --dst "E:/fashion-data/01-RAW/jackets_women_zalando"
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

CATEGORY_QUERIES: dict[str, list[str]] = {
    "biker": [
        "zalando women biker jacket leather product",
        "zalando womens moto jacket faux leather",
        "zalando women leather biker jacket black",
        "zalando uk biker jacket women shop",
    ],
    "blazer": [
        "zalando women blazer jacket product",
        "zalando womens tailored blazer",
        "zalando women oversized blazer shop",
        "zalando uk blazer women fashion",
    ],
    "bomber": [
        "zalando women bomber jacket product",
        "zalando womens satin bomber jacket",
        "zalando women varsity bomber shop",
        "zalando uk bomber jacket women",
    ],
    "fur_jacket": [
        "zalando women faux fur jacket product",
        "zalando womens shearling jacket shop",
        "zalando women teddy coat jacket",
        "zalando uk fur jacket women",
    ],
    "parka": [
        "zalando women parka jacket product",
        "zalando womens hooded parka shop",
        "zalando women long parka coat",
        "zalando uk parka women fashion",
    ],
}

CATEGORY_URLS: dict[str, list[str]] = {
    "biker": [
        "https://www.zalando.co.uk/womens-clothing-jackets-leather-jackets/",
        "https://www.zalando.co.uk/women/?q=biker+jacket",
    ],
    "blazer": [
        "https://www.zalando.co.uk/womens-clothing-jackets-blazers/",
        "https://www.zalando.co.uk/women/?q=blazer+women",
    ],
    "bomber": [
        "https://www.zalando.co.uk/womens-clothing-jackets-bomber-jackets/",
        "https://www.zalando.co.uk/women/?q=bomber+jacket+women",
    ],
    "fur_jacket": [
        "https://www.zalando.co.uk/womens-clothing-jackets-faux-fur-jackets/",
        "https://www.zalando.co.uk/women/?q=shearling+jacket+women",
    ],
    "parka": [
        "https://www.zalando.co.uk/womens-clothing-jackets-parka/",
        "https://www.zalando.co.uk/women/?q=parka+women",
    ],
}

LABEL_MAP = {
    "biker":      "biker",
    "blazer":     "blazer",
    "bomber":     "bomber",
    "fur_jacket": "fur jacket",
    "parka":      "parka",
}

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
IMG_WIDTH        = 762
MIN_SIDE         = 300
GOTO_TIMEOUT     = 60_000
DOWNLOAD_TIMEOUT = 20
MIN_DELAY        = 0.4
MAX_DELAY        = 1.0
SCROLL_PAUSE     = 2.0
MAX_SCROLLS      = 35

# Matches ztat.net product images (article/ or catalog/ paths)
ZTAT_RE = re.compile(
    r"https?://img\d*\.ztat\.net/(?:article|catalog)/[a-zA-Z0-9/_-]+\.jpg"
)

IMG_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.zalando.co.uk/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

STEALTH_JS = """
() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-GB', 'en'] });
    window.chrome = { runtime: {} };
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stem(category: str, idx: int) -> str:
    return f"{category}_zal_{idx:04d}"


def _normalise_url(url: str) -> str:
    base = re.sub(r"\?.*$", "", url)
    return f"{base}?imwidth={IMG_WIDTH}"


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


def _extract_ztat_urls(html: str) -> list[str]:
    """Extract unique normalised ztat.net URLs from raw HTML."""
    raw = ZTAT_RE.findall(html)
    seen_bases: set[str] = set()
    result = []
    for u in raw:
        base = re.sub(r"\?.*$", "", u)
        if base not in seen_bases:
            seen_bases.add(base)
            result.append(_normalise_url(u))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A: Bing Image Search
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

            # Scroll and click "See more" to load additional results
            for scroll_n in range(10):
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
            urls = _extract_ztat_urls(content)

            new = []
            for u in urls:
                base = re.sub(r"\?.*$", "", u)
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
# Strategy B: Headed Zalando direct
# ─────────────────────────────────────────────────────────────────────────────

async def collect_via_zalando_headed(
    context,
    urls: list[str],
    max_urls: int,
) -> list[str]:
    collected_bases: set[str] = set()
    intercepted: list[str] = []

    def on_request(req):
        url = req.url
        if ZTAT_RE.match(url):
            base = re.sub(r"\?.*$", "", url)
            if base not in collected_bases:
                collected_bases.add(base)
                intercepted.append(_normalise_url(url))

    for page_url in urls:
        if len(collected_bases) >= max_urls:
            break

        print(f"    Zalando: {page_url}")
        page = await context.new_page()
        await page.add_init_script(STEALTH_JS)
        page.on("request", on_request)

        try:
            # "commit" fires on first HTTP response — much faster than "load"
            await page.goto(page_url, wait_until="commit", timeout=GOTO_TIMEOUT)
            await asyncio.sleep(4)

            try:
                title = await page.title()
                print(f"    Title: {title[:60]}")
                if any(k in title.lower() for k in ("denied", "captcha", "robot")):
                    print("    BLOCKED")
                    await page.close()
                    continue
            except Exception:
                pass

            vh = (page.viewport_size or {}).get("height", 900)
            for _ in range(MAX_SCROLLS):
                if len(collected_bases) >= max_urls:
                    break
                await page.evaluate(f"window.scrollBy(0, {int(vh * 0.8)})")
                await asyncio.sleep(SCROLL_PAUSE)
                try:
                    at_bottom = await page.evaluate(
                        "window.scrollY + window.innerHeight >= document.body.scrollHeight - 300"
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
# Download + filter pipeline
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
        print("\n  [Mode: headed Zalando direct]")
        img_urls = await collect_via_zalando_headed(
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
        print(f"  WARNING: no URLs found for {category}")
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
    p = argparse.ArgumentParser(description="Zalando women's jacket scraper v3")
    p.add_argument("--dst", type=Path,
                   default=Path("E:/fashion-data/01-RAW/jackets_women_zalando"))
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--category", choices=list(CATEGORY_QUERIES.keys()), default=None)
    p.add_argument("--max-per-class", type=int, default=350)
    p.add_argument("--headed", action="store_true",
                   help="Open visible browser and scrape Zalando directly")
    p.add_argument("--csv-only", action="store_true",
                   help="Regenerate CSV from already-downloaded images")
    args = p.parse_args()

    csv_path = args.csv or args.dst.parent / "labels_jackets_zalando.csv"
    categories = [args.category] if args.category else list(CATEGORY_QUERIES.keys())

    if args.csv_only:
        csv_from_existing(args.dst, csv_path)
        return

    mode = "headed Zalando" if args.headed else "Bing image search"
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