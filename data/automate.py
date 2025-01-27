#!/usr/bin/env python3

import os
import time
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException

def setup_driver(download_dir):
    """
    Set up a Chrome WebDriver instance to automatically download PGN files
    to the specified directory without prompt.
    """
    options = webdriver.ChromeOptions()
    # Configure Chrome to download files automatically to download_dir
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True
    }
    options.add_experimental_option("prefs", prefs)

    # Optional: run headless
    # options.add_argument("--headless")

    # Initialize driver
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    return driver

def scroll_to_bottom(driver, pause_time=2):
    """
    Scroll to the bottom of the page to load all elements.
    This can be repeated if the page uses infinite scroll.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def visit_games_page_and_get_players(driver, url, demo=False, demo_limit=5):
    """
    Go to the specified Chess.com games page, scroll, and collect player profile links.
    """
    driver.get(url)

    # Wait for the page to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.daily-game-list-component"))
    )

    # Scroll to load all games
    scroll_to_bottom(driver, pause_time=2)

    # Collect all links to players from the loaded games
    # The following CSS selectors/structure may need adjustment
    game_containers = driver.find_elements(By.CSS_SELECTOR, "div.daily-game-item-container")
    
    player_links = []
    for game in game_containers:
        # Each game container might have two players: White and Black
        links = game.find_elements(By.CSS_SELECTOR, "a.user-link-component")
        for link in links:
            player_profile_url = link.get_attribute("href")
            if player_profile_url:
                player_links.append(player_profile_url)

    # Remove duplicates (in case the same user appears multiple times)
    player_links = list(dict.fromkeys(player_links))

    # If demo is enabled, limit the links
    if demo:
        player_links = player_links[:demo_limit]

    return player_links

def download_pgn_for_player(driver, player_url, wait_time=10):
    """
    Visit a player's profile, attempt to tick 'select all' or 'select games',
    and then download the PGN file. 
    """
    driver.get(player_url)

    # Example logic: you might need to navigate to their "Games" or "Archive" tab first
    # This depends on Chess.com's site structure. We'll attempt a known approach:

    try:
        # Wait for the profile or game list to load
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.profile-component, div.user-games"))
        )
    except TimeoutException:
        print(f"Timed out waiting for user profile: {player_url}")
        return

    # Navigate to "Archive" or "Games" if required (depending on the page’s design)
    # For example, we might look for a link text or specific button:
    try:
        archive_tab = driver.find_element(By.LINK_TEXT, "Games")
        archive_tab.click()
        time.sleep(2)
    except NoSuchElementException:
        print(f"No explicit 'Games' link found for {player_url}; skipping.")
        return

    # Possibly scroll again or wait for more games to load
    scroll_to_bottom(driver)

    # Try to find a “Select All” or “Select Games” checkbox.
    try:
        select_all_checkbox = driver.find_element(By.CSS_SELECTOR, "input[type='checkbox'].all-games-checkbox")
        select_all_checkbox.click()
        time.sleep(1)
    except NoSuchElementException:
        print(f"Select all checkbox not found for {player_url}. Attempting an alternative approach.")
        return

    # Look for a “Download” or “Download PGN” button.
    try:
        download_button = driver.find_element(By.CSS_SELECTOR, "button.download-pgn-button")
        download_button.click()
        time.sleep(3)  # Wait a few seconds to ensure the download is triggered
        print(f"PGN download triggered for {player_url}.")
    except NoSuchElementException:
        print(f"No download button found for {player_url}. Skipping.")
        return

def main():
    parser = argparse.ArgumentParser(description="Automate downloading Chess.com PGNs.")
    parser.add_argument("--url", type=str, default="https://www.chess.com/games?page=7",
                        help="URL of the Chess.com games page to start from.")
    parser.add_argument("--download_dir", type=str, default="downloaded_pgns",
                        help="Directory to store downloaded PGN files.")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (only process a limited number of players).")
    parser.add_argument("--demo_limit", type=int, default=3,
                        help="Number of players to process in demo mode.")
    args = parser.parse_args()

    # Create download directory if it doesn't exist
    os.makedirs(args.download_dir, exist_ok=True)

    print(f"Starting with URL: {args.url}")
    print(f"Download directory: {args.download_dir}")
    if args.demo:
        print(f"Running in DEMO mode. Will only process {args.demo_limit} player(s).")

    # Setup WebDriver
    driver = setup_driver(args.download_dir)

    try:
        # Fetch the players from the given URL
        player_links = visit_games_page_and_get_players(
            driver,
            args.url,
            demo=args.demo,
            demo_limit=args.demo_limit
        )

        print(f"Found {len(player_links)} player link(s).")

        for idx, player_url in enumerate(player_links, 1):
            print(f"[{idx}/{len(player_links)}] Processing player: {player_url}")
            download_pgn_for_player(driver, player_url)
    finally:
        print("Closing WebDriver...")
        driver.quit()


if __name__ == "__main__":
    main()
