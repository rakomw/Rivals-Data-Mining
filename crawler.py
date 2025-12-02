import json
import jsonpickle
import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

DAREDEVIL_ID = "1055001"

# represents a game of marvel rivals --
class Match:

    def __init__(self, match_id, team_one=None, team_two=None, map=None, team_one_score=None, team_two_score=None, winner=None):
        self.match_id = match_id
        self.team_one = team_one
        self.team_two = team_two
        self.map = map
        self.team_one_score = team_one_score
        self.team_two_score = team_two_score
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
        self.match_data = {}
        self.browser = browser
        self.page = self.browser.new_page()
        self.start_time = time.time()
        self.time_step = 0

    def scrape_profile(self, profile_id, scroll_seconds = 5, season = "Season 4"):
        if self.page is None or self.page.is_closed():
            self.page = self.browser.new_page()
        #url = f"https://rivalsmeta.com/player/{profile_id}"
        if len(self.page.locator(".player-search").all()) == 0:
            url = "https://rivalsmeta.com/"
            self.page.goto(url, wait_until="load", timeout=0)

        search_box = self.page.locator(".player-search").all()[0]
        search_box.locator("input").fill(f"{profile_id}")
        search_box.locator("button").click()

        # profile is private, no matches can be scraped
        if len(self.page.locator(".profile-private").all()) > 0:
            self.profiles[profile_id] = 2
            return

        # Choose which season's data to use
        season_select = self.page.locator(".select-season").locator("select")
        season_select.select_option(season)

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
        for md in match_details:
            try:
                #md.locator(".link-ind").click()
                #print("CLICK")
                # find link to full match details page within dropdown display
                soup = BeautifulSoup(md.inner_html(), features="lxml")
                link_tag = soup.find("div", "group-replay-link")
                link = link_tag.find("a").get("href")
                # parse out match id number rather than storing entire url
                num = link.split("?")[0].split("/")[-1]
                # if this match isn't in our dataset, add it unvisited
                if self.match_ids.get(num) is None:
                    self.match_ids[num] = 0
                    m = Match(num)
                    try:
                        m.map = soup.find("div", "map").find("div", "name").text
                    except Exception as e:
                        print(f"Error scraping map {str(e)}")
                        with open("errors-map.html","a") as fp:
                            fp.write(str(e))
                            fp.write(str(md.inner_html()))
                            fp.write("\n*****\n")
                        m.map = "Unknown"
                    try:
                        score_div = soup.find("div", "map").find("div", "score")
                        if score_div is not None:
                            scores = score_div.text.split(":")
                            m.team_one_score = max(int(scores[0]),int(scores[1]))
                            m.team_two_score = min(int(scores[0]),int(scores[1]))
                        else:
                            # some formats don't have a true score, we just call them 1-0
                            m.team_one_score = 1
                            m.team_two_score = 0
                    except Exception as e:
                        print(f"Error scraping score: {str(e)}")
                        with open("errors-score.html","a") as fp:
                            fp.write(str(e))
                            fp.write(str(md.inner_html()))
                            fp.write("\n*****\n")
                        m.team_one_score = 1
                        m.team_two_score = 0

                    m.winner = 1
                    self.match_data[num] = m

            except Exception as e:
                print(f"Error scraping matches from profile: {str(e)}")
                with open("errors-profile.html","a") as fp:
                    fp.write(str(e))
                    fp.write(str(md.inner_html()))
                    fp.write("\n*****\n")
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
        team_one = []
        team_two = []
        for r in rows:
            try:
                # parse player info for match data
                p = parse_player_from_row(r)
                if r.find("div", "result-border win") is not None:
                    team_one.append(p)
                elif r.find("div", "result-border loss") is not None:
                    team_two.append(p)
                # if we've never seen this player, add their profile link as unvisited
                if self.profiles.get(p.player_id) is None:
                    self.profiles[p.player_id] = 0
            except Exception as e:
                print(f"Error scraping player from table row: {str(e)}")

        self.match_data[match_id].team_one = team_one
        self.match_data[match_id].team_two = team_two
        self.match_ids[match_id] = 1

    def scrape_n_profiles(self, n):
        scraped = 0
        for id, visited in self.profiles.items():
            if visited == 0:
                try:
                    self.scrape_profile(id)
                except Exception as e:
                    print(f"Error while scraping profile {id}: {str(e)}")
                    self.page.close()
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
                    print(f"Error in scrape_n_matches: {str(e)}")
                    self.page.close()
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
        for m in self.match_data.values():
            if m.team_one is None:
                continue
            all_players = []
            all_players.extend(m.team_one)
            all_players.extend(m.team_two)
            for p in all_players:
                if p.rank == "-1":
                    count += 1
                    break
        return count

    def count_scraped_matches(self):
        count = 0
        for m in self.match_data.values():
            if m.team_one is not None:
                count += 1
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

    def filter_daredevil(self):
        return {id: match for id, match in self.match_data.items() if self.has_daredevil(match)}
    def has_daredevil(self, match):
        players = []
        if match.team_one is not None:
            players.extend(match.team_one)
        if match.team_two is not None:
            players.extend(match.team_two)
        for p in players:
            for h in p.heroes_played:
                if h[0] == DAREDEVIL_ID:
                    return True
        return False

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        crawler = Crawler(browser)
        crawler.load_data_files()
        #crawler.show_stats()
        #daredevil_games = crawler.filter_daredevil()
        #with open("daredevil_games.txt","w") as fp:
        #    for id in daredevil_games:
        #        fp.write(f"https://rivalsmeta.com/matches/{id}\n")

        #print(f"Found {len(daredevil_games)} daredevil games")
        run_length = 60 * 60 * 10
        checkpoint = 0
        starting_matches = crawler.count_scraped_matches()
        while (time.time() - crawler.start_time) < run_length:
            new_matches = crawler.scrape_n_matches(1000)
            if not new_matches:
                print("Exhausted known matches -- scraping profiles for more.")
                new_profiles = crawler.scrape_n_profiles(50)
                if not new_profiles:
                    print("***ENDING EARLY - NO UNVISITED PROFILES***")
                    break
            if (time.time() - crawler.start_time) > checkpoint:
                crawler.show_stats()
                crawler.save_data_files()
                scraped_matches = crawler.count_scraped_matches()
                new_matches = scraped_matches - starting_matches
                run_mins = (time.time() - crawler.start_time) / 60
                print(f"Scraped {new_matches} matches in {run_mins :.1f} minutes so far this run.")
                checkpoint += run_length / 10

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
