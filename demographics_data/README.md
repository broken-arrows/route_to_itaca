
# Demographics Data: The "where did you get that fckng number?" place

Welcome, curious and probably unprepared complainer, to the `demographics_data` folder. The kitchen where the numbers used for "Route to Ítaca" were cooked. Statistical purity MAY be a distant memory, but some decisions had to be made. And hey, it's a game after all, so some sacrifices were unavoidable.

### Data Sources (or the quesionable quality of the ingredients we have to deal with)

**Centre d'Estudis d'Opinio (CEO)**
Every bit of voting intention data comes from the Catalan public agency on Public Opinion. It's the only semi-neutral source that actually is transparent with their collected data in RAW, unfiltered glory. The detail of their files is actually quite impressive and the sample size is if not the best one of the most extensive of all opinion polls of the time.

*However*, I have to do a shotout to the 2012T3 dataset: a true clusterfck of misrepresentation. Just before the November '12 Parlament election, they decided to throw data quality out the window, so I had to play data surgeon, necromancer, and therapist to get anything usable. Provincial breakdowns? More like mental breakdowns (for me). Heavy cooking was required to have a more or less beliveable voting intention maping for the starting date.

**Statistical Institute of Catalonia (IDESCAT)**
The Catalan census office has, naturaly, quite an extensive data set on population, employement, and etc. However, what data they collect and how it is territorially distributed changes wildly through time. E.g. not all data sets are divided provincially (since some use Vegueries instead), and some data is only available in n-year intervals. So, to build up the actual population of each demographic in each province, some juggling was necessary:

- General population and age distribution data is from 2015 (a good middle point between 2012 and 2019).
- Unemploymenet data is from 2012 since in-game it is only used for the starting date (it should evolve according to player action).
- Idem with the share of bussiness owners, which is actually hardcoded into the `cat_pop_weights.py` data crunching scripts since its just four numbers.
- Education population weights and employment stats. I cobble together general population data from 2015 (because why not?) and employment stats from 2012 (so we can pretend to model employment changes as the years roll by). If it works, it works.

### Methodology & Assumptions (a.k.a. cooking the data)

- The unemployed demographic only represents ages 30-65.
        - Above that it's petanca people, so retired.
        - Below 30 goes to young. Yes I am generous but in all honesty I do not think it's that bad of a representation.
        - Needless to say underage folks are sorted out to not bloat the "young" demographic.
- Rural demographic is built from census data of those living in "urban" areas according to IDESCAT.
        - This makes the cut slightly differently to the rural cut I have to do for CEO data, since there I work with people living in places with less than 10k people. This is a bit below where IDESCAT draws the line, but once again what are we going to do about that.
- Bussiness demographic is created from those that report as self-employed. This needs a bit (or a lot, depending on the province) of cooking to represent higher classes.
- Middle demographic is build from IDESCAT workers education data. Depending on the province I ended up drawing the line a bit different: university education, advanced FP degrees, or old "Superior Bachelour" titles.
- Industrial demographic is bunddles those that did not make the cut. Industrial may not be the best name, since it does mix some service sector workers with traditional industrial areas, but hey it is what it is.

- Evolving (un-)employment: when unemployment drops, the new jobs get distributed super-realistically. When it grows, I take from the same place (also 11/10 realism):
        - 30% to/from industrial workers (industrial demographic)
        - 55% to/from middle class workers (middle demographic)
        - 15% to/from new business owners (business demographic)

Is this perfect? Absolutely not. Is it good enough for a simulation? Hopefully.

### Simulation

The election simulation bit is actually the most realistic part. Voting intention is paired with demographic weights, and then real seat allocation using both the 3% electoral threshold and d'Hont law is used. Provincial seats are spread like IRL, so there's nothing else to say.

### Folder Structure

- `auto_balancer.py`: Helps a ton for cooking the data. Ensures that your percentages always properly add up to 100.
- `cat_ceo_process.py`: Wrestles raw CEO survey data into something the simulation can actually use. Manual cooking is needed after that-
- `cat_pop_weights.py`: Turns IDESCAT data into population weights. Manual cooking is also needed but to a much lesser extent.
- `cat_simulation.py`: Runs the actual election simulations
- `ceo_raw/`: All the raw CEO survey CSVs (e.g., `cat_data_2012.csv`).
- `idescat_raw/`: Raw IDESCAT data (age, education, urbanization, etc.).
- `clean/`: Where cleaned, cooked, and processed data ends up (population weights, vote intentions, etc.). Files starting with `ok_` are the result of manual cooking.
- `simulation_results/`: Results from running the simulations.
- `.temp/`: Temporary files. Ignore unless you are debugging at 3am.

### How to Use This

1. **Prepare Data:** Drop new raw files into `ceo_raw/` or `idescat_raw/` as needed. Optional: pray.
2. **Run the Scripts:** Fire up the Python scripts to process and clean the data. Outputs land in `clean/`. If something breaks, blame the CEO or IDESCAT.
3. **Simulate:** Call people to vote with `cat_simulation.py`. Results go to `simulation_results/`.

### Final Notes

- All scripts are Python, and you’ll need the usual suspects (pandas, numpy, matplotlib, etc.).
- Note that no library installation system or virtual environments are included because I was too lazy to create them. For reference I am using Python 3.12.
- For questions, complaints, or existential dread about the data, check the script comments. Or just yell into the abyss. Sometimes it yells back.