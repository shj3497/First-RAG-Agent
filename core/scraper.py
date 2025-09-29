# core/scraper.py
import asyncio
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from typing import Set, List
from urllib.parse import urlparse

# --- 크롤링 제외 경로 목록 ---
# 이 리스트에 포함된 문자열로 시작하는 URL은 크롤링 대상에서 제외됩니다.
EXCLUDE_URL_PATTERNS = [
    "/privacy-policy",
]

# --- 다국어 페이지 식별 경로 ---
# 아래 리스트에 포함된 경로로 시작하는 페이지는 국문 페이지가 아니라고 간주하여 제외합니다.
# 예: /us, /us/about-us 등
EXCLUDE_LANG_PREFIXES = [
    "/us",
    "/jp",
    "/vn",
    "/cn",
    "/hk",
]


async def get_all_page_urls(site_url: str) -> List[str]:
    """
    지정된 사이트의 sitemap.xml을 읽어 크롤링 대상 URL 목록을 반환합니다.

    Args:
        site_url (str): sitemap.xml 파일이 위치한 사이트의 URL.
                         (예: "https://www.megazone.com")
    """
    sitemap_url = f"{site_url}/sitemap.xml"
    print(f"Fetching sitemap from: {sitemap_url}")

    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")

        # sitemap에서 <loc> 태그에 포함된 실서버 URL들을 추출합니다.
        prod_urls = [loc.text for loc in soup.find_all("loc")]

        if not prod_urls:
            print("❌ Sitemap is empty or <loc> tags could not be found.")
            return []

        # --- URL 처리 로직 ---
        # sitemap.xml에서 추출한 실서버 URL을 그대로 사용하도록 변경합니다.
        # 더 이상 로컬 서버 주소(base_url)로 치환하지 않습니다.

        # sitemap에 있는 첫 번째 URL을 기준으로 실서버의 base_url을 알아냅니다.
        # 예: "https://www.megazone.com/some/page" -> "https://www.megazone.com"
        first_url = prod_urls[0]
        parsed_uri = urlparse(first_url)
        prod_base_url = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
        print(f"Detected production base URL from sitemap: {prod_base_url}")

        # 주석 처리: 실서버 URL의 도메인을 로컬 서버(base_url)로 교체했던 기존 로직
        # local_urls = [url.replace(prod_base_url, base_url)
        #               for url in prod_urls]

        # --- 필터링 로직 (제외 목록 적용) ---
        # 이제 prod_urls를 직접 사용하고, 상대 경로 계산도 prod_base_url을 기준으로 합니다.
        final_urls = []
        for url in prod_urls:
            relative_link = url.replace(prod_base_url, "")
            if not relative_link:
                relative_link = "/"

            # 1. 일반 제외 패턴 확인 (예: /privacy-policy)
            if any(relative_link.startswith(pattern) for pattern in EXCLUDE_URL_PATTERNS):
                continue

            # 2. 다국어 페이지 제외 패턴 확인 (예: /us, /us/about)
            is_lang_page = False
            for prefix in EXCLUDE_LANG_PREFIXES:
                # '/us' 와 정확히 일치하거나, '/us/' 로 시작하는 경우를 확인하여
                # '/users' 같은 다른 경로가 실수로 제외되는 것을 방지합니다.
                if relative_link == prefix or relative_link.startswith(prefix + '/'):
                    is_lang_page = True
                    break

            if is_lang_page:
                continue

            final_urls.append(url)

        print(
            f"Sitemap parsed. Found {len(final_urls)} non-excluded Korean pages to crawl.")
        return final_urls

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching or parsing sitemap: {e}")
        print("   - Next.js 프로젝트의 public 폴더나 빌드 결과에 sitemap.xml이 존재하는지 확인해주세요.")
        print("   - Next.js 개발 서버가 정상적으로 실행 중인지 확인해주세요.")
        return []


async def get_page_html(url: str) -> str:
    """
    Playwright를 사용하여 지정된 URL의 페이지에 접속하고,
    모든 동적 콘텐츠(JavaScript, Suspense 등) 렌더링이 완료된 후의
    최종 HTML을 반환합니다.

    Args:
        url (str): HTML을 가져올 페이지의 URL.

    Returns:
        str: 최종 렌더링된 HTML 콘텐츠.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            # 'networkidle' 상태는 네트워크 요청이 잠잠해질 때까지 기다리는 옵션으로,
            # 동적 데이터 로딩이 완료되기를 기다리는 데 효과적입니다.
            await page.goto(url, wait_until="networkidle")
            # 페이지의 최종 HTML 콘텐츠를 가져옴
            html = await page.content()
            return html
        except Exception as e:
            print(f"Could not get HTML from {url}: {e}")
            return ""
        finally:
            await browser.close()
