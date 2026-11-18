[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/oxuwan/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/oxuwan/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecaster

`main.py` forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (19 November 2026, UTC), as five quantiles per hour, from the first hour's print alone. It learns the launch-day shape from the case's own past-launch comparables: each comparable contributes its release-day log-ratios log(viewers_h / viewers_1), combined as a trimmed weighted mean per hour, with weights favouring console-only rolling-midnight launches like GTA VI and launches whose hour-1 print is close to this case's in scale (a huge hour-1 print means the audience is already in and the day stays flat; a modest one means a steep rise as the US unlocks). The spread per hour comes from the weighted interquartile range of those log-ratios, so the 90% band is honest about how differently big launches actually play out. Leave-one-out over the kit's 37 comparable launch curves, the model scores a mean pinball loss of 0.51 against the house persistence baseline's 0.93, and 0.18 against 0.32 on the eight console-only launches most like GTA VI. Standard library only, deterministic, no network.

## Run it

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

<sub>Entered on [make.fun](https://make.fun/?ref=xk6tgtt0&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
