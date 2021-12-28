# nfelo

nfelo is a power ranking, prediction, and betting model for the NFL. Nfelo take's 538's Elo framework and further adapts it for the NFL, hence the name nfelo (pronounced "NFL oh").

The model's output is visualized on nfeloapp.com where you can explore:
* [Weekly Predictions](https://www.nfeloapp.com/games)
* [Current Powerrankings](https://www.nfeloapp.com/nfl-power-ratings)
* [538's QB model](https://www.nfeloapp.com/qb-rankings)
* [Analysis behind nfelo's ideas](https://www.nfeloapp.com/analysis)


## Repository Description

This repository contains all the code necessary to translate raw data into weekly predictions. This process has three main phases:

1. Pull and scrape data from nflfastR, PFF, and various Vegas Line sites
2. Compile data into a single dataset and run intermediate models (nfelo ratings and wepa)
3. Translate power ratings and contextual game information into win and line expectations


## Install and Use

nfelo is a python package. To install, simply download this repository into your site-packages folder and install the dependencies detailed in the requirements.txt file.

Because nfelo pulls from PFF, running the model requires you to access team grades that are behind a paywall (sorry!), and the PFF scraper does require you to copy your cookie into the config_private.json file. This cookie must be refreshed before each run.

Each phase of the build can be run individually, but to generate predictions, run the following script:

```python

import nfelo

## update data ##
nfelo.pull_nflfastR_pbp()
nfelo.pull_nflfastR_game()
nfelo.pull_nflfastR_roster()
nfelo.pull_nflfastR_logo()
nfelo.pull_538_games()
nfelo.pull_sbr_lines()
nfelo.pull_tfl_lines()
nfelo.pull_pff_grades()

## format ##
nfelo.format_spreads()
nfelo.game_data_merge()

## update models ##
nfelo.calculate_wepa()
nfelo.calculate_nfelo()

## ouput spreads ##
nfelo.calculate_spreads()

```

This process will output a csv in the output_data folder called 'predictions.csv'

Because this package is exclusively used as a workflow automation for building nfelo predictions each week, it's not well suited for other uses and likely has some bugs if updates are run before every game for a given week has been completed. It does produce nfelo rankings, wepa results, and a few other datapoints, which can be found in various csvs within the folder hierarchy.


## Authors

This package is built and maintained by [@greerreNFL](https://twitter.com/greerreNFL). Feel free to DM with comments and questions.


## Version History

* 0.1
    * Initial package release
    * Includes nfelo v3.0 and workflow automations to recreate weekly predictions
    
