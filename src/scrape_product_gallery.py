"""
scrape_product_gallery.py
=========================
Collects ALL shots of each shoe product (main view + details/close-ups/rear),
grouped into per-product subfolders.  Extends the existing scrape_farfetch.py /
scrape_zalando.py architecture with two site-specific strategies:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY A — Farfetch  (CDN shot-ID enumeration, no page visits)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Farfetch CDN URL anatomy:
  https://cdn-images.farfetch-contents.com/AA/BB/CC/DD/{productId}_{shotId}_{size}.jpg
                                           ──────────── ─────────── ──────── ──────────
                                           path prefix  product ID  shot #   px width

Every view of the same product shares the same prefix + productId; only the
shotId differs.  We intercept the *first* URL per product from a Bing search
or headed listing page, then probe shot IDs 1 → MAX_SHOTS until the CDN
returns 404.  No product-page visit is needed — the CDN is an open file server.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY B — Zalando  (headed product-page crawl + network interception)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zalando ztat.net URLs don't encode a predictable shot structure, so we visit
each product page, scroll through the gallery carousel to trigger lazy loads,
and capture every ztat.net request via Playwright network interception.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output layout
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  dst/
    product_0001/
      shot_01.jpg      ← main (first shot)
      shot_02.jpg      ← detail / side view
      shot_03.jpg      ← close-up / rear
      ...
    product_0002/
      ...
  gallery_manifest.csv   (product_idx, shot_idx, file, source_url)

Requirements
------------
    pip install playwright pillow httpx
    playwright install chromium

Usage
-----
  # Farfetch — Bing strategy (headless), 10 products × up to 8 shots
  python src/scrape_product_gallery.py farfetch `
    --search "women oxford shoes eyelets farfetch" `
    --max-products 10 `
    --dst "E:/fashion-data/01-RAW/eyelet_gallery_farfetch"

  # Farfetch — headed listing page strategy
  python src/scrape_product_gallery.py farfetch --headed `
    --search "men derby shoes metal eyelets farfetch" `
    --max-products 10 `
    --dst "E:/fashion-data/01-RAW/eyelet_gallery_farfetch"

  # Zalando — always headed (CloudFront blocks headless)
  python src/scrape_product_gallery.py zalando `
    --search "women lace up shoes eyelets" `
    --max-products 10 `
    --dst "E:/fashion-data/01-RAW/eyelet_gallery_zalando"

  # From a CSV of main image URLs already collected
  python src/scrape_product_gallery.py farfetch `
    --from-csv "E:/fashion-data/csv/shoes_tagged.csv" `
    --url-column source_url `
    --max-products 20 `
    --dst "E:/fashion-data/01-RAW/eyelet_gallery_farfetch"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import random
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────────────────

GOTO_TIMEOUT     = 60_000      # ms
DOWNLOAD_TIMEOUT = 15          # seconds per image
MIN_SIDE         = 300         # px — discard thumbnails
MAX_ASPECT       = 2.5
MIN_DELAY        = 0.5
MAX_DELAY        = 1.2
SCROLL_PAUSE     = 1.8
MAX_SCROLLS      = 20
STEALTH_JS = """
() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
    Object.defineProperty(navigator, 'plugins',   { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-GB', 'en'] });
    window.chrome = { runtime: {} };
}
"""

COMMON_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _passes_quality(data: bytes) -> tuple[bool, str]:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if min(w, h) < MIN_SIDE:
            return False, f"too_small({w}x{h})"
        if max(w / h, h / w) > MAX_ASPECT:
            return False, "bad_aspect"
        return True, ""
    except Exception as e:
        return False, f"decode_error:{e}"


async def _download(url: str, referer: str = "") -> bytes | None:
    import httpx
    headers = {**COMMON_HEADERS, **({"Referer": referer} if referer else {})}
    try:
        async with httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT, headers=headers, follow_redirects=True
        ) as client:
            r = await client.get(url)
        return r.content if r.status_code == 200 else None
    except Exception as e:
        return None


def _save_shot(
    data: bytes,
    product_dir: Path,
    shot_idx: int,
) -> Path:
    product_dir.mkdir(parents=True, exist_ok=True)
    dst = product_dir / f"shot_{shot_idx:02d}.jpg"
    # Convert to JPEG
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(dst, "JPEG", quality=90)
    return dst


# ═════════════════════════════════════════════════════════════════════════════
# FARFETCH  —  CDN shot-ID enumeration
# ═════════════════════════════════════════════════════════════════════════════

# Pattern: https://cdn-images.farfetch-contents.com/AA/BB/CC/DD/PRODID_SHOTID_SIZE.jpg
FF_CDN_RE = re.compile(
    r"(https?://cdn-images\.farfetch-contents\.com"
    r"/\d+/\d+/\d+/\d+/"        # path prefix (incl. trailing slash)
    r"(\d+)_)"                   # group 1: prefix+productId, group 2: productId
    r"(\d+)"                     # group 3: shotId
    r"_(\d+)\.jpg"               # group 4: size (300/480/600/1000)
)

FF_IMG_SIZE = 1000
FF_MAX_SHOTS = 8   # probe shot indices 1..N until 404


def _ff_parse(url: str) -> tuple[str, str, str, str] | None:
    """
    Returns (prefix_with_productId, productId, shotId, size) or None.
    prefix_with_productId is everything up to and including productId + "_"
    e.g. "https://cdn-images.farfetch-contents.com/17/40/04/29/17400429_"
    """
    m = FF_CDN_RE.search(url)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _ff_shot_url(prefix: str, shot_id: int, size: int = FF_IMG_SIZE) -> str:
    return f"{prefix}{shot_id}_{size}.jpg"


def _ff_group_key(url: str) -> str | None:
    """Return the dedup key: prefix+productId (ignores shotId and size)."""
    parsed = _ff_parse(url)
    return parsed[0] if parsed else None


def _ff_extract_from_html(html: str) -> list[str]:
    """Pull unique, normalised Farfetch CDN URLs from raw HTML."""
    seen_keys: set[str] = set()
    result: list[str] = []
    for m in FF_CDN_RE.finditer(html):
        key = m.group(1)
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(f"{key}{m.group(3)}_{FF_IMG_SIZE}.jpg")
    return result


# Bing image search result metadata — each card carries a JSON blob in m='...'
# containing both the image URL (murl) and the source page URL (purl).
_BING_M_RE = re.compile(r'\bm=\'(\{[^\']+\})')

def _ff_extract_bing_product_map(html: str) -> dict[str, str]:
    """
    Parse Bing image-search HTML and return {cdn_img_url: farfetch_product_page_url}.
    Only entries where the CDN URL matches FF_CDN_RE are included.
    """
    import json
    result: dict[str, str] = {}
    for m in _BING_M_RE.finditer(html):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue
        murl = data.get("murl", "")
        purl = data.get("purl", "")
        if murl and purl and FF_CDN_RE.search(murl) and "farfetch.com" in purl:
            result[murl] = purl
    return result


async def _ff_collect_via_bing(
    context,
    query: str,
    max_products: int,
) -> list[tuple[str, str]]:
    """
    Search Bing images for the query.
    Returns list of (cdn_img_url, product_page_url) pairs.
    product_page_url is '' when Bing didn't expose it for that card.
    """
    bing_url = (
        "https://www.bing.com/images/search?q="
        + query.replace(" ", "+")
        + "&form=HDRSC2&first=1"
    )
    seen_keys: set[str] = set()
    result: list[tuple[str, str]] = []

    page = await context.new_page()
    await page.add_init_script(STEALTH_JS)

    try:
        await page.goto(bing_url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT)
        await asyncio.sleep(3)

        for _ in range(6):
            if len(result) >= max_products:
                break
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(1.5)
            html = await page.content()

            # Primary: try to get CDN URL + product page URL from Bing card metadata
            product_map = _ff_extract_bing_product_map(html)
            for cdn_url, page_url in product_map.items():
                key = _ff_group_key(cdn_url)
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    result.append((_ff_normalise_cdn(cdn_url), page_url))

            # Fallback: plain CDN URL extraction (no product page URL)
            for cdn_url in _ff_extract_from_html(html):
                key = _ff_group_key(cdn_url)
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    result.append((cdn_url, ""))

    except Exception as e:
        print(f"  [Bing] Error: {e}")
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return result[:max_products]


def _ff_normalise_cdn(url: str) -> str:
    """Normalise any size token in a Farfetch CDN URL to FF_IMG_SIZE."""
    url = re.sub(r"_\d+\.jpg$", f"_{FF_IMG_SIZE}.jpg", url)
    return re.sub(r"\?.*$", "", url)


async def _ff_collect_via_headed(
    context,
    listing_url: str,
    max_products: int,
) -> list[str]:
    """Visit a Farfetch listing page headed, intercept CDN URLs."""
    seen_keys: set[str] = set()
    result: list[str] = []

    def on_request(req):
        url = req.url
        parsed = _ff_parse(url)
        if parsed:
            key = parsed[0]
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(_ff_shot_url(key, int(parsed[2])))

    page = await context.new_page()
    await page.add_init_script(STEALTH_JS)
    page.on("request", on_request)

    try:
        await page.goto(listing_url, wait_until="commit", timeout=GOTO_TIMEOUT)
        await asyncio.sleep(4)

        vh = (page.viewport_size or {}).get("height", 900)
        for _ in range(MAX_SCROLLS):
            if len(result) >= max_products:
                break
            await page.evaluate(f"window.scrollBy(0, {int(vh * 0.8)})")
            await asyncio.sleep(SCROLL_PAUSE)
    except Exception as e:
        print(f"  [Farfetch headed] Error: {e}")
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return result[:max_products]


async def _ff_visit_product_page(
    context,
    product_page_url: str,
    product_id: str,
) -> list[str]:
    """
    Visit a Farfetch product detail page with a headed/headless browser.
    Intercept every CDN request for this productId that fires as the gallery
    carousel loads, and return all unique shot URLs at full resolution.

    Strategy:
      1. Navigate to product page (Farfetch rarely blocks headed Chromium)
      2. Let the initial gallery load
      3. Click the next-image arrow up to 10 times to trigger lazy shots
      4. Click thumbnail strip to ensure all variants load
      5. Collect every intercepted CDN URL whose productId matches
    """
    intercepted: dict[str, str] = {}   # base_url → normalised_url

    def on_request(req):
        url = req.url
        parsed = _ff_parse(url)
        if parsed and parsed[1] == product_id:
            base = re.sub(r"_\d+\.jpg$", "", re.sub(r"\?.*$", "", url))
            if base not in intercepted:
                intercepted[base] = _ff_normalise_cdn(url)

    page = await context.new_page()
    await page.add_init_script(STEALTH_JS)
    page.on("request", on_request)

    try:
        await page.goto(product_page_url, wait_until="commit", timeout=GOTO_TIMEOUT)
        await asyncio.sleep(3)

        # 1. Scroll down a bit (loads hero image variants)
        await page.evaluate("window.scrollBy(0, 400)")
        await asyncio.sleep(1.5)

        # 2. Click the next-image arrow repeatedly
        next_selectors = [
            "[data-testid='pdp-carousel-next']",
            "button[aria-label*='Next']",
            "button[aria-label*='next']",
            ".pdp-media-btn--next",
            "[class*='CarouselButton'][class*='right']",
            "[class*='carousel'][class*='next']",
        ]
        clicked = False
        for sel in next_selectors:
            try:
                btn = page.locator(sel).first
                for _ in range(10):
                    if await btn.is_visible(timeout=800):
                        await btn.click()
                        await asyncio.sleep(0.7)
                        clicked = True
                    else:
                        break
                if clicked:
                    break
            except Exception:
                continue

        # 3. Click thumbnails (side strip or bottom strip)
        thumb_selectors = [
            "[data-testid='pdp-thumbnail-image']",
            "[class*='Thumbnail'] img",
            "[class*='thumbnail'] img",
            ".pdp-gallery__thumbnails img",
        ]
        for sel in thumb_selectors:
            try:
                for thumb in await page.locator(sel).all():
                    try:
                        await thumb.click()
                        await asyncio.sleep(0.5)
                    except Exception:
                        pass
            except Exception:
                continue

        await asyncio.sleep(1.0)

    except Exception as e:
        print(f"    [product page] Error: {e}")
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return list(intercepted.values())


async def _ff_enumerate_shots(
    prefix_with_product_id: str,
    max_shots: int = FF_MAX_SHOTS,
) -> list[tuple[int, str]]:
    """
    Probe shot IDs 1 → max_shots for one product.
    Returns list of (shot_index, url) for every shot that returns 200.
    Shot IDs are not always sequential from 1 — we try all IDs and 
    keep whichever return a real image.
    """
    # The first intercepted shot ID is our anchor.  Farfetch shot IDs are
    # typically 8-digit integers that increment by 1.  We probe a window of
    # [anchor - 2 .. anchor + max_shots] to be safe.
    hits: list[tuple[int, str]] = []

    # Strategy: try shot IDs in the realistic range.
    # Farfetch uses two known formats:
    #   legacy  : 22000001, 22000002, ...
    #   modern  : <8-digit product-based ID> + 1, +2, ...
    # We don't know the anchor without the URL, so the caller passes the
    # first observed shot ID via prefix.  We scan ±0..max_shots from it.

    import httpx

    async with httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT,
        headers={**COMMON_HEADERS, "Referer": "https://www.farfetch.com/"},
        follow_redirects=True,
    ) as client:
        for shot_offset in range(max_shots):
            # We'll try both directions: same shot, next shots
            # The prefix already ends with "productId_"; append shotId_size
            # We don't know the real shotId sequence, so we probe by
            # using offsets relative to what appears in the first intercepted URL.
            # This is handled in scrape_farfetch_gallery() below.
            pass

    return hits


async def scrape_farfetch_gallery(
    query: str,
    dst: Path,
    max_products: int,
    max_shots: int = FF_MAX_SHOTS,
    headed: bool = False,
    listing_url: str | None = None,
    seed_urls: list[str] | None = None,
) -> list[dict]:
    """
    Full Farfetch gallery scraper.

    Two-stage approach:
      Stage 1 — Bing (headless) or listing page (headed):
                collects one seed CDN URL + product page URL per product.
      Stage 2 — For each product:
                PRIMARY  : visit product page + intercept ALL CDN shots
                FALLBACK : CDN shot-ID enumeration if page visit yields ≤1 shot
                           (fixes: probe_start was max() instead of min(); now
                            correct as seed - max_shots)

    Returns list of manifest dicts: {product_idx, shot_idx, file, source_url}.
    """
    from playwright.async_api import async_playwright
    import httpx

    dst.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not headed,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
            slow_mo=50 if headed else 0,
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=COMMON_HEADERS["User-Agent"],
        )

        # ── Stage 1: collect (cdn_url, product_page_url) pairs ───────────
        if seed_urls:
            # seed_urls may be plain CDN strings — wrap as (url, "")
            pairs: list[tuple[str, str]] = [(u, "") for u in seed_urls[:max_products]]
        elif headed and listing_url:
            print(f"\n[Farfetch] Headed listing crawl: {listing_url}")
            raw = await _ff_collect_via_headed(context, listing_url, max_products)
            pairs = [(u, "") for u in raw]
        else:
            print(f"\n[Farfetch] Bing search: '{query}'")
            pairs = await _ff_collect_via_bing(context, query, max_products)

        n_with_page = sum(1 for _, p in pairs if p)
        print(f"  Found {len(pairs)} products  ({n_with_page} with product page URLs)")

        # ── Stage 2: collect all shots per product ────────────────────────
        seen_global_hashes: set[str] = set()

        for prod_idx, (seed_url, product_page_url) in enumerate(pairs, start=1):
            parsed = _ff_parse(seed_url)
            if not parsed:
                print(f"  [prod {prod_idx:04d}] Could not parse CDN URL: {seed_url[:80]}")
                continue

            prefix, product_id, seed_shot_str, _ = parsed
            seed_shot  = int(seed_shot_str)
            product_dir = dst / f"product_{prod_idx:04d}"

            print(f"\n  [prod {prod_idx:04d}] productId={product_id}  seed_shot={seed_shot}")

            all_shot_urls: list[str] = []

            # ── 2a: product page visit (primary) ─────────────────────────
            if product_page_url:
                print(f"    [page visit] {product_page_url[:80]}")
                page_shots = await _ff_visit_product_page(
                    context, product_page_url, product_id
                )
                print(f"    [page visit] intercepted {len(page_shots)} shot URL(s)")
                all_shot_urls = page_shots

            # ── 2b: CDN enumeration fallback ──────────────────────────────
            # Used when: no product page URL, or page visit returned ≤1 shot.
            # BUG FIX: was max(seed-2, seed-max_shots) = seed-2 always.
            # Now correctly: probe from seed-max_shots to seed+2.
            if len(all_shot_urls) <= 1:
                if product_page_url and len(all_shot_urls) <= 1:
                    print(f"    [page visit] insufficient shots — falling back to CDN probe")
                else:
                    print(f"    [CDN probe] no product page URL — using enumeration")

                probe_start = seed_shot - max_shots   # fixed: was max() not min()
                probe_end   = seed_shot + 2            # shots > seed are rare

                async with httpx.AsyncClient(
                    timeout=DOWNLOAD_TIMEOUT,
                    headers={**COMMON_HEADERS, "Referer": "https://www.farfetch.com/"},
                    follow_redirects=True,
                ) as client:
                    for shot_id in range(probe_start, probe_end + 1):
                        url = _ff_shot_url(prefix, shot_id)
                        await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                        try:
                            r = await client.get(url)
                        except Exception:
                            continue
                        if r.status_code == 200:
                            all_shot_urls.append(url)
                        elif shot_id > seed_shot and all_shot_urls:
                            break   # 404 above seed → no more

            # ── 2c: download and save ─────────────────────────────────────
            if not all_shot_urls:
                print(f"    No shot URLs found")
                continue

            shot_order = 0
            async with httpx.AsyncClient(
                timeout=DOWNLOAD_TIMEOUT,
                headers={**COMMON_HEADERS, "Referer": "https://www.farfetch.com/"},
                follow_redirects=True,
            ) as client:
                for url in all_shot_urls:
                    await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                    try:
                        r = await client.get(url)
                        if r.status_code != 200:
                            continue
                        data = r.content
                    except Exception:
                        continue

                    ok, reason = _passes_quality(data)
                    if not ok:
                        print(f"    SKIP ({reason})")
                        continue

                    h = _image_hash(data)
                    if h in seen_global_hashes:
                        print(f"    DUP")
                        continue
                    seen_global_hashes.add(h)

                    shot_order += 1
                    saved = _save_shot(data, product_dir, shot_order)
                    manifest.append({
                        "product_idx": prod_idx,
                        "shot_idx":    shot_order,
                        "file":        str(saved),
                        "source_url":  url,
                        "product_page": product_page_url,
                    })
                    print(f"    shot {shot_order:02d} → {saved.name}")

            if shot_order == 0:
                print(f"    No valid shots saved")

        await browser.close()

    return manifest


# ═════════════════════════════════════════════════════════════════════════════
# ZALANDO  —  headed product-page crawl + network interception
# ═════════════════════════════════════════════════════════════════════════════

# Matches Zalando product image CDN
ZTAT_RE = re.compile(
    r"https?://img\d*\.ztat\.net/(?:article|catalog)/[a-zA-Z0-9/_-]+\.jpg"
)
ZTAT_WIDTH = 762   # request large variant

# Zalando product page link pattern
ZALANDO_PRODUCT_RE = re.compile(
    r'href="(https?://(?:www\.)?zalando\.(?:co\.uk|de|fr|it|com)/[^"]+\.html)"'
)


def _ztat_normalise(url: str) -> str:
    base = re.sub(r"\?.*$", "", url)
    return f"{base}?imwidth={ZTAT_WIDTH}"


def _ztat_base(url: str) -> str:
    return re.sub(r"\?.*$", "", url)


async def _zal_collect_product_urls(
    context,
    listing_url: str,
    max_products: int,
) -> list[str]:
    """Visit a Zalando listing/search page and collect product page URLs."""
    page = await context.new_page()
    await page.add_init_script(STEALTH_JS)
    product_urls: list[str] = []
    seen: set[str] = set()

    try:
        await page.goto(listing_url, wait_until="commit", timeout=GOTO_TIMEOUT)
        await asyncio.sleep(4)

        vh = (page.viewport_size or {}).get("height", 900)
        for _ in range(MAX_SCROLLS):
            if len(product_urls) >= max_products:
                break
            await page.evaluate(f"window.scrollBy(0, {int(vh * 0.8)})")
            await asyncio.sleep(SCROLL_PAUSE)

            # Extract product page links from current DOM
            html = await page.content()
            for m in ZALANDO_PRODUCT_RE.finditer(html):
                url = m.group(1)
                if url not in seen:
                    seen.add(url)
                    product_urls.append(url)

    except Exception as e:
        print(f"  [Zalando listing] Error: {e}")
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return product_urls[:max_products]


async def _zal_collect_product_urls_via_bing(
    context,
    query: str,
    max_products: int,
) -> list[str]:
    """
    Use Bing web search (not image search) to find Zalando product page URLs.
    More reliable than scraping listing pages for bot-detection reasons.
    """
    bing_url = (
        "https://www.bing.com/search?q="
        + f"site:zalando.co.uk {query}".replace(" ", "+")
    )
    seen: set[str] = set()
    result: list[str] = []

    page = await context.new_page()
    await page.add_init_script(STEALTH_JS)

    try:
        await page.goto(bing_url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT)
        await asyncio.sleep(3)

        for _ in range(4):
            if len(result) >= max_products:
                break
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(1.5)
            html = await page.content()
            for m in ZALANDO_PRODUCT_RE.finditer(html):
                url = m.group(1)
                if "zalando" in url and url.endswith(".html") and url not in seen:
                    seen.add(url)
                    result.append(url)

    except Exception as e:
        print(f"  [Bing→Zalando] Error: {e}")
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return result[:max_products]


async def _zal_scrape_product_page(
    context,
    product_url: str,
) -> list[str]:
    """
    Visit one Zalando product page, scroll through the image gallery carousel,
    and return all intercepted ztat.net image URLs (deduped by base URL).
    """
    seen_bases: set[str] = set()
    image_urls: list[str] = []

    def on_request(req):
        url = req.url
        if ZTAT_RE.match(url):
            base = _ztat_base(url)
            if base not in seen_bases:
                seen_bases.add(base)
                image_urls.append(_ztat_normalise(url))

    page = await context.new_page()
    await page.add_init_script(STEALTH_JS)
    page.on("request", on_request)

    try:
        await page.goto(product_url, wait_until="commit", timeout=GOTO_TIMEOUT)
        await asyncio.sleep(3)

        # Gallery is usually a horizontal carousel — also try clicking
        # arrow / thumbnail navigation to trigger lazy loads.

        # 1. Scroll the page slowly (loads thumbnails in the sidebar/strip)
        vh = (page.viewport_size or {}).get("height", 900)
        for _ in range(6):
            await page.evaluate(f"window.scrollBy(0, {int(vh * 0.5)})")
            await asyncio.sleep(1.0)

        # 2. Click the "next image" arrow up to 8 times
        next_arrow_selectors = [
            "button[aria-label*='next']",
            "button[aria-label*='Next']",
            "[data-testid='pdp-image-carousel-next']",
            ".pdp-media-btn--next",
            "button.zds-button[direction='right']",
        ]
        for selector in next_arrow_selectors:
            for _ in range(8):
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=1000):
                        await btn.click()
                        await asyncio.sleep(0.8)
                    else:
                        break
                except Exception:
                    break

        # 3. Click thumbnail strip items
        thumb_selectors = [
            "[data-testid='pdp-thumbnail'] img",
            ".pdp-media-thumbnails img",
            "[class*='thumbnail'] img",
        ]
        for selector in thumb_selectors:
            try:
                thumbs = await page.locator(selector).all()
                for thumb in thumbs[:10]:
                    try:
                        await thumb.click()
                        await asyncio.sleep(0.6)
                    except Exception:
                        pass
            except Exception:
                pass

        await asyncio.sleep(1.5)

    except Exception as e:
        print(f"    Error visiting {product_url[:70]}: {e}")
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return image_urls


async def scrape_zalando_gallery(
    query: str,
    dst: Path,
    max_products: int,
    listing_url: str | None = None,
    seed_urls: list[str] | None = None,
) -> list[dict]:
    """
    Full Zalando gallery scraper.
    Always runs headed (CloudFront blocks headless).
    Returns manifest dicts: {product_idx, shot_idx, file, source_url}.
    """
    from playwright.async_api import async_playwright

    dst.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,   # Zalando requires headed
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
            slow_mo=50,
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=COMMON_HEADERS["User-Agent"],
        )

        # ── 1. Collect product page URLs ──────────────────────────────────
        if seed_urls:
            product_urls = seed_urls[:max_products]
        elif listing_url:
            print(f"\n[Zalando] Listing crawl: {listing_url}")
            product_urls = await _zal_collect_product_urls(context, listing_url, max_products)
        else:
            print(f"\n[Zalando] Bing→Zalando search: '{query}'")
            product_urls = await _zal_collect_product_urls_via_bing(
                context, query, max_products
            )

        print(f"  Found {len(product_urls)} product pages")

        # ── 2. Visit each product page and collect gallery images ─────────
        seen_global_hashes: set[str] = set()

        for prod_idx, product_url in enumerate(product_urls, start=1):
            print(f"\n  [prod {prod_idx:04d}] {product_url[:80]}")
            product_dir = dst / f"product_{prod_idx:04d}"

            await asyncio.sleep(random.uniform(2.0, 4.0))

            image_urls = await _zal_scrape_product_page(context, product_url)
            print(f"    Intercepted {len(image_urls)} image URLs")

            shot_idx = 1
            for img_url in image_urls:
                await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                data = await _download(img_url, referer="https://www.zalando.co.uk/")
                if data is None:
                    continue

                ok, reason = _passes_quality(data)
                if not ok:
                    print(f"    SKIP ({reason})")
                    continue

                h = _image_hash(data)
                if h in seen_global_hashes:
                    continue
                seen_global_hashes.add(h)

                saved = _save_shot(data, product_dir, shot_idx)
                manifest.append({
                    "product_idx": prod_idx,
                    "shot_idx":    shot_idx,
                    "file":        str(saved),
                    "source_url":  img_url,
                    "product_url": product_url,
                })
                print(f"    shot {shot_idx:02d} → {saved.name}")
                shot_idx += 1

        await browser.close()

    return manifest


# ═════════════════════════════════════════════════════════════════════════════
# Manifest writer
# ═════════════════════════════════════════════════════════════════════════════

def write_manifest(manifest: list[dict], dst: Path) -> Path:
    if not manifest:
        return dst
    out = dst / "gallery_manifest.csv"
    fieldnames = list(manifest[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download full product galleries (main + detail shots) from Farfetch or Zalando",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("site", choices=["farfetch", "zalando"], help="Target site")
    p.add_argument(
        "--search", default="",
        help="Search query for discovering products (used if --from-csv / --listing-url not provided)"
    )
    p.add_argument(
        "--listing-url",
        help="Direct category/listing page URL to crawl for product links (Zalando) or listing "
             "thumbnails (Farfetch headed mode)"
    )
    p.add_argument(
        "--from-csv",
        help="Path to existing CSV with a column of seed image or product URLs"
    )
    p.add_argument(
        "--url-column", default="source_url",
        help="Column name in --from-csv that holds the URLs"
    )
    p.add_argument(
        "--max-products", type=int, default=10,
        help="Number of products to collect"
    )
    p.add_argument(
        "--max-shots", type=int, default=FF_MAX_SHOTS,
        help="(Farfetch only) Maximum shots to probe per product"
    )
    p.add_argument(
        "--dst", required=True,
        help="Output directory for product subfolders"
    )
    p.add_argument(
        "--headed", action="store_true",
        help="(Farfetch only) Use headed browser listing crawl instead of Bing"
    )
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    dst = Path(args.dst)

    # Load seed URLs from CSV if provided
    seed_urls: list[str] | None = None
    if args.from_csv:
        import csv as csv_mod
        with open(args.from_csv, newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            seed_urls = [row[args.url_column] for row in reader
                         if row.get(args.url_column)]
        print(f"Loaded {len(seed_urls)} seed URLs from {args.from_csv}")

    if args.site == "farfetch":
        manifest = await scrape_farfetch_gallery(
            query        = args.search,
            dst          = dst,
            max_products = args.max_products,
            max_shots    = args.max_shots,
            headed       = args.headed,
            listing_url  = args.listing_url,
            seed_urls    = seed_urls,
        )
    else:  # zalando
        manifest = await scrape_zalando_gallery(
            query        = args.search,
            dst          = dst,
            max_products = args.max_products,
            listing_url  = args.listing_url,
            seed_urls    = seed_urls,
        )

    if manifest:
        csv_path = write_manifest(manifest, dst)
        products = len({r["product_idx"] for r in manifest})
        shots    = len(manifest)
        print(f"\n{'='*60}")
        print(f"DONE — {products} products, {shots} total shots")
        print(f"Output : {dst}")
        print(f"CSV    : {csv_path}")
        print(f"{'='*60}")
        print(
            f"\nNext — run VLM verification:\n"
            f"  python src/infer_shoes.py \\\n"
            f"    --image-dir \"{dst}\" \\\n"
            f"    --out-csv   \"E:/fashion-data/csv/eyelet_gallery_check.csv\" \\\n"
            f"    --no-toe-heel\n"
        )
    else:
        print("\n[WARNING] No images collected.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
