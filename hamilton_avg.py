import pandas as pd

df = pd.read_csv('data/f1_2020_2025.csv')
ham = df[df['FullName'] == 'Lewis Hamilton']

for season in sorted(ham['Season'].unique()):
    avg = ham[ham['Season'] == season]['RacePosition'].mean()
    print(f"{int(season)}: avg P{avg:.1f}")
