import json
import jsonpickle
import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# represents a game of marvel rivals --
class Match:
    def __init__(self, match_id, team_one, team_two, winner):
        self.match_id = match_id
        self.team_one = team_one
        self.team_two = team_two
        self.winner = winner

    def __repr__(self):
        return repr(vars(self))

class Player:
    def __init__(self, player_id, heroes_played, rank, kills, deaths, assists, final_hits, solo_kills,
                damage, damage_taken, damage_healed, accuracy):
        self.player_id = player_id
        self.heroes_played = heroes_played
        self.rank = rank
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.final_hits = final_hits
        self.solo_kills = solo_kills
        self.damage = damage
        self.damage_taken = damage_taken
        self.damage_healed = damage_healed
        self.accuracy = accuracy

    def __repr__(self):
        return repr(vars(self))

class Crawler:
    def __init__(self, browser):
        self.profiles = {}
        self.match_ids = {}
        self.match_data = []
        self.browser = browser
        self.page = self.browser.new_page()
        self.start_time = time.time()
        self.time_step = 0

    def scrape_profile(self, profile_id, scroll_seconds: int = 10):
        if self.page is None or self.page.is_closed():
            self.page = self.browser.new_page()
        url = f"https://rivalsmeta.com/player/{profile_id}"
        self.page.goto(url, wait_until="load", timeout=0)

        # profile is private, no matches can be scraped
        if len(self.page.locator(".profile-private").all()) > 0:
            self.profiles[profile_id] = 2
            return

        # repeatedly scroll and click the "show more" button to load match history
        t0 = time.time()
        while time.time() - t0 < scroll_seconds:
            # bit of a hack
            show_more_button = self.page.locator(".show-more-btn").all()
            for b in show_more_button:
                b.click()

        # locate and click all the buttons to open (and load) match info
        dropdown_buttons = self.page.locator(".link-ind").all()
        for b in dropdown_buttons:
            b.click()
        match_details = self.page.locator(".match-details").all()
        for m in match_details:
            try:
                # find link to full match details page within dropdown display
                soup = BeautifulSoup(m.inner_html(), features="lxml")
                link = soup.find("div", "group-replay-link").find("a").get("href")
                # parse out match id number rather than storing entire url
                num = link.split("?")[0].split("/")[-1]
                # if this match isn't in our dataset, add it unvisited
                if self.match_ids.get(num) is None:
                    self.match_ids[num] = 0

            except Exception as e:
                print(str(e))
                continue
        # mark profile as successfully scraped
        self.profiles[profile_id] = 1
        # close page for good health
        self.page.close()

    def scrape_match(self, match_id):
        if self.page is None or self.page.is_closed():
            self.page = self.browser.new_page()
        url = f"https://rivalsmeta.com/matches/{match_id}"
        self.page.goto(url, wait_until="load", timeout=0)

        html = self.page.content()
        soup = BeautifulSoup(html,features="lxml")
        rows = soup.find_all("tr")
        rows = rows[1:]
        players_list = []
        for r in rows:
            try:
                # parse player info for match data
                p = parse_player_from_row(r)
                players_list.append(p)
                # if we've never seen this player, add their profile link as unvisited
                if self.profiles.get(p.player_id) is None:
                    self.profiles[p.player_id] = 0
            except Exception as e:
                print(str(e))
                with open("errors.html","a") as fp:
                    fp.write(str(r))

        self.match_data.append(Match(match_id, players_list[0:6],players_list[6:],1))
        self.match_ids[match_id] = 1

    def scrape_all_profiles(self):
        for id, visited in self.profiles.items():
            if visited == 0:
                try:
                    self.scrape_profile(id)
                except Exception as e:
                    print(str(e))
                    continue

    def scrape_all_matches(self):
        for id, visited in self.match_ids.items():
            if visited == 0:
                try:
                    self.scrape_match(id)
                except Exception as e:
                    print(str(e))
                    continue

    def scrape_n_profiles(self, n):
        scraped = 0
        for id, visited in self.profiles.items():
            if visited == 0:
                try:
                    self.scrape_profile(id)
                except Exception as e:
                    print(str(e))
                    continue
                else:
                    scraped += 1
                    if scraped == n:
                        break
        return scraped != 0

    def scrape_n_matches(self, n):
        scraped = 0
        for id, visited in self.match_ids.items():
            if visited == 0:
                try:
                    self.scrape_match(id)
                except Exception as e:
                    print(str(e))
                    continue
                else:
                    scraped += 1
                    # close page for memory health
                    if scraped % 50 == 0:
                        self.page.close()
                    if scraped == n:
                        break
        return scraped != 0

    def save_data_files(self):
        with open("profiles.json","w") as fp:
            fp.write(json.dumps(self.profiles))
        with open("match_ids.json","w") as fp:
            fp.write(json.dumps(self.match_ids))
        with open("match_data.json","w") as fp:
            fp.write(jsonpickle.encode(self.match_data))
        print("Saved data to files")

    def save_and_exit(self):
        self.save_data_files()
        self.browser.close()

    def load_data_files(self):
        with open("profiles.json","r") as fp:
            self.profiles = json.load(fp)
        with open("match_ids.json","r") as fp:
            self.match_ids = json.load(fp)
        with open("match_data.json","r") as fp:
            self.match_data = jsonpickle.decode(fp.read())
        print("Loaded data from files")

    def count_unranked_matches(self):
        count = 0
        for m in self.match_data:
            all_players = []
            all_players.extend(m.team_one)
            all_players.extend(m.team_two)
            for p in all_players:
                if p.rank == "-1":
                    count += 1
                    break
        return count

    def show_stats(self):
        private_profile_count = 0
        visited_profile_count = 0
        unvisited_profile_count = 0
        scraped_match_count = 0
        unscraped_match_count = 0
        for id, visited in self.profiles.items():
            if self.profiles[id] == 1:
                visited_profile_count += 1
            elif self.profiles[id] == 2:
                private_profile_count += 1
            else:
                unvisited_profile_count += 1
        for id, scraped in self.match_ids.items():
            if self.match_ids[id] == 1:
                scraped_match_count += 1
            else:
                unscraped_match_count += 1

        print(f"{visited_profile_count} public profiles scraped. {private_profile_count} private profiles found.")
        print(f"{unvisited_profile_count} profiles left to visit.")
        print(f"{scraped_match_count} matches scraped, {unscraped_match_count} left to scrape.")
        print(f"{self.count_unranked_matches()} unranked matches included in data.")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        crawler = Crawler(browser)
        crawler.load_data_files()
        crawler.show_stats()
        run_length = 60 * 60 * 11
        starting_matches = len(crawler.match_data)
        while (time.time() - crawler.start_time) < run_length:
            new_matches = crawler.scrape_n_matches(1000)
            if not new_matches:
                print("Exhausted known matches -- scraping profiles for more.")
                new_profiles = crawler.scrape_n_profiles(25)
                if not new_profiles:
                    print("***ENDING EARLY - NO UNVISITED PROFILES***")
                    break
            else:
                new_matches = len(crawler.match_data) - starting_matches
                run_mins = (time.time() - crawler.start_time) / 60
                print(f"Scraped {new_matches} matches in {run_mins :.1f} minutes this run.")
                print(f"{len(crawler.match_data)} matches recorded in total.")
            crawler.save_data_files()

        crawler.show_stats()
        crawler.save_and_exit()

def split_hero_id_from_url(image_link):
    i = len(image_link) - 1
    while (i > 0 and image_link[i] != "_"):
        i -= 1
    return image_link[i+1:len(image_link)-4]

def parse_player_from_row(r):
    player_id = r.find('a').get("href")
    player_id = player_id[8:]
    heroes = r.find("div", "other-heroes").find_all("div", "hero")
    hero_list = []
    for h in heroes:
        image_link = h.find("img")["src"]
        hero_id = split_hero_id_from_url(image_link)
        hero_time = h.find("div", "time").text
        hero_list.append([hero_id,hero_time])
    try:
        rank = r.find("div","score-delta").text.split()
        rank = rank[0]
    except Exception as e:
        #print(f"{str(e)} at score-delta")
        rank = "-1"
    kda = r.find("div","kda")
    kda_arr = kda.find("div","avg").text.split()
    kills = kda_arr[0]
    deaths = kda_arr[2]
    assists = kda_arr[4]
    final_hits = kda.parent.next_sibling
    solo_kills = final_hits.next_sibling.text
    final_hits = final_hits.text
    damage = r.find("div","stat-value damage").text
    damage_taken = r.find("div","stat-value dmg-taken").text
    damage_healed = r.find("div","stat-value heal")
    accuracy = damage_healed.parent.next_sibling.text.strip()
    damage_healed = damage_healed.text
    return Player(player_id, hero_list, rank, kills, deaths, assists, final_hits,
     solo_kills, damage, damage_taken, damage_healed, accuracy)

if __name__ == "__main__":
    main()
