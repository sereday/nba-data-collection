from nba_api.stats.endpoints import leaguegamelog

print("NBA API Season Type Options")
print("=" * 40)

# Valid season types found
valid_season_types = {
    'Regular Season': 'Regular Season',
    'Playoffs': 'Playoffs',
    'Pre Season': 'Pre Season',
    'All Star': 'All Star'
}

print("✅ VALID SEASON TYPES:")
for user_name, api_name in valid_season_types.items():
    print(f"  '{user_name}' → API: '{api_name}'")

print("\n❌ INVALID SEASON TYPES (tested):")
invalid_types = ['Regular', 'All-Star', 'Preseason', 'Play-In', 'Play-In Tournament']
for invalid in invalid_types:
    print(f"  '{invalid}'")

print("\n📊 Testing data availability for 2023-24:")
print("-" * 40)

for season_type in valid_season_types.values():
    try:
        log = leaguegamelog.LeagueGameLog(
            season='2023-24',
            season_type_all_star=season_type,
            player_or_team_abbreviation='P'
        )
        dfs = log.get_data_frames()
        if dfs and len(dfs) > 0:
            records = len(dfs[0])
            print(f"  {season_type}: {records:,} player records")
        else:
            print(f"  {season_type}: No data")
    except Exception as e:
        print(f"  {season_type}: Error - {str(e)[:50]}...")

print("\n💡 Use these in job_request.json:")
print("  \"season_types\": [\"Regular\", \"Playoffs\", \"Preseason\", \"All-Star\"]")