import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import os
import time

def extract_text_content(soup):
    """Extract text from headers (h1-h6), p, summary, and label tags in natural order."""
    content = []
    
    # Find all relevant tags in the order they appear
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'summary', 'label']):
        # Get text content, strip whitespace
        text = tag.get_text(strip=True)
        # Only include if text has meaningful content
        if text and len(text) > 10:  # Minimum 10 characters to filter out tiny fragments
            # Prefix with tag type for clarity
            content.append(f"{text}")
    
    return content

def scrape_page(url):
    """Scrape a single page and extract text content."""
    try:
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        # Get the page content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        content = extract_text_content(soup)
        
        # Get page title
        title = soup.title.string if soup.title else 'No Title'
        
        return {
            'url': url,
            'title': title.strip(),
            'content': content
        }
        
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def get_urls_from_sitemap(sitemap_url):
    """Extract URLs from sitemap."""
    try:
        response = requests.get(sitemap_url)
        root = ET.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        return [loc.text for loc in root.findall('.//ns:loc', namespace)]
    except Exception as e:
        print(f"Error fetching sitemap: {str(e)}")
        return []

def save_content(data, output_dir, index):
    """Save content to a text file with proper formatting."""
    if not data or not data['content']:
        return
    
    filename = f"page_{index}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data['content']:
            f.write(item + "\n\n")

def main():
    sitemap_url = 'https://wappnet.com/sitemap.xml'
    output_dir = 'scraped_data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting web scraping process...")
    
    # Get URLs from sitemap
    urls = get_urls_from_sitemap(sitemap_url)
    print(f"Found {len(urls)} URLs in sitemap")
    
    # Process each URL
    for i, url in enumerate(urls, 1):
        print(f"\nProcessing {i}/{len(urls)}: {url}")
        
        # Scrape the page
        data = scrape_page(url)
        
        if data and data['content']:
            save_content(data, output_dir, i)
            print(f"Saved content to page_{i}.txt")
        else:
            print(f"No content extracted from {url}")
        
        # Polite delay between requests
        time.sleep(2)
    
    print("\nScraping completed!")

if __name__ == '__main__':
    main()