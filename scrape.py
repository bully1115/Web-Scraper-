from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

def scrape_website(website):
    print("Launching Chrome browser...")

    # Use ChromeDriverManager to install the correct driver
    options = Options()
    options.add_argument("--headless")  # Run in headless mode for performance
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Initialize WebDriver with ChromeDriverManager
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # Open the website
        driver.get(website)
        
        # Wait for the page to load completely (use WebDriverWait)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        print("Page loaded successfully!")

        # Get the page source
        html = driver.page_source

        return html
    finally:
        driver.quit()


def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    # Remove unwanted tags like script and style
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get the cleaned text
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content


def split_dom_content(dom_content, max_length=6000):
    # Split content into smaller chunks
    return [
        dom_content[i: i + max_length] for i in range(0, len(dom_content), max_length)
    ]
