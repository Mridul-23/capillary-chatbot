import scrapy
from bs4 import BeautifulSoup

class DocspiderSpider(scrapy.Spider):
    name = "docspider"
    allowed_domains = ["docs.capillarytech.com"]
    start_urls = ["https://docs.capillarytech.com/docs/introduction"]

    custom_settings = {
        'DOWNLOAD_DELAY': 0.5,  
        'CONCURRENT_REQUESTS': 2,  
        'AUTOTHROTTLE_ENABLED': True, 
        'AUTOTHROTTLE_START_DELAY': 2,  
        'AUTOTHROTTLE_MAX_DELAY': 2,  
        'FEEDS': {
            'capillary_docs.json': {
                'format': 'json',
                'overwrite': False
            }
        }
    }

    def parse(self, response):
        # Extract sidebar links (adjust selectors if needed)
        items = response.css('.Sidebar1t2G1ZJq-vU1 .Sidebar-link2Dsha-r-GKh2::attr(href)').getall()
        
        for url in items:
            full_url = response.urljoin(url)
            yield response.follow(full_url, callback=self.parse_item)

    def parse_item(self, response):
        # Parse page HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags (scripts, styles, etc.)
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        # Extract clean text
        text = soup.get_text(separator="\n", strip=True)

        # Remove extra blank lines
        clean_text = "\n".join(line for line in text.splitlines() if line.strip())

        yield {
            'url': response.url,
            'text': clean_text
        }