from nba_api.stats.endpoints import leaguegamelog

# Try different parameter combinations
test_cases = [
    {'season': '2023-24', 'season_type_all_star': 'Regular Season', 'player_or_team_abbreviation': 'P'},
    {'season': '2023-24', 'season_type_all_star': 'Regular', 'player_or_team_abbreviation': 'P'},
    {'season': '2022-23', 'season_type_all_star': 'Regular Season', 'player_or_team_abbreviation': 'P'},
]

for i, params in enumerate(test_cases):
    print(f"\nTest case {i+1}: {params}")
    try:
        log = leaguegamelog.LeagueGameLog(**params)
        dfs = log.get_data_frames()
        print(f"  ✓ Success! Got {len(dfs)} dataframes")
        if dfs:
            print(f"  First dataframe shape: {dfs[0].shape}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Additional test for teams
print("\nTesting teams data:")
try:
    log = leaguegamelog.LeagueGameLog(
        season='2023-24',
        season_type_all_star='Regular Season',
        player_or_team_abbreviation='T'
    )
    dfs = log.get_data_frames()
    print(f"  ✓ Teams data: Got {len(dfs)} dataframes")
    if dfs:
        print(f"  First dataframe shape: {dfs[0].shape}")
except Exception as e:
    print(f"  ✗ Teams error: {e}")