# Machine Learning for European Football Recruitment
### A KNN Approach to Player Similarity

A machine learning project exploring how match-event data can support football scouting by identifying players with similar statistical profiles. Using Wyscout data from the **2017/18 English Premier League season**, we engineer player performance features and apply a nearest-neighbor model to a historical case study involving **Kieran Trippier** and Tottenham Hotspur.

**Project status:** Mid-stage research prototype. The model measures statistical similarity; it does **not** predict transfer success or establish a player's overall quality or tactical fit.

## Project overview

Football recruitment involves more than identifying talented players: clubs also need to understand whether a prospective signing offers a suitable playing profile. This project explores an interpretable, data-driven way to narrow down scouting candidates using match-event statistics.

We use Trippier as a reference defender, identify his five nearest statistical neighbors, and compare the resulting distances with those of two players who were already at Tottenham: **Serge Aurier** and **Kyle Walker-Peters**.

## Data

The analysis uses Wyscout data from the **2017/18 Premier League**:

- **380 matches** and **643,150 recorded match events**
- Player, team, match, and event-level information
- Events such as passes, shots, and duels, including event coordinates

Playing time is estimated from lineups and substitutions, with a **900-minute minimum** for inclusion in the main nearest-neighbor comparison. Event counts are expressed per 90 minutes to account for differences in playing time.

**Data availability:** Raw Wyscout files are not included in this repository. Obtain the applicable dataset from its original source and follow the notebook's loading paths. Check the dataset's terms before redistributing it.

## Methodology

1. **Prepare the data:** Connect match events with player identities and estimate minutes played.
2. **Engineer six features:**
   - `Pass_Attempts_per90`
   - `Pass_Accuracy`
   - `Forward_Passes_per90`
   - `Smart_Passes_per90`
   - `Duel_per90`
   - `Shot_per90`
3. **Filter candidates:** Compare players classified as defenders with at least 900 estimated minutes.
4. **Standardize features:** Use scikit-learn's `StandardScaler` so features on different scales can contribute to the distance calculation.
5. **Find similar players:** Fit `NearestNeighbors` using **Euclidean distance** in the six-dimensional standardized feature space and retrieve Trippier's five closest matches (excluding Trippier himself).
6. **Explore the results:** Compare retrieved neighbors with Aurier and Walker-Peters and visualize the distances.

**Note:** `Pass_Accuracy` is a percentage, not a per-90 event count. The model uses equal feature weighting after standardization.

## Preliminary results

The five closest statistical matches to **Kieran Trippier** in the eligible defender pool were:

| Rank | Player | Euclidean distance |
| ---: | --- | ---: |
| 1 | Ashley Young | 2.207 |
| 2 | Trent Alexander-Arnold | 2.256 |
| 3 | Héctor Bellerín | 2.316 |
| 4 | Charlie Daniels | 2.561 |
| 5 | César Azpilicueta | 2.733 |

Additional comparisons:

| Tottenham player | Approximate distance from Trippier | Note |
| --- | ---: | --- |
| Serge Aurier | 4.224 | Eligible for the main comparison pool |
| Kyle Walker-Peters | 7.0 | Exploratory comparison; below the 900-minute threshold |

Smaller distances mean greater similarity **on the six selected features only**. These results are not evidence that a player would have been a successful transfer or an exact tactical replacement.

## Repository contents

```text
football-player-similarity/
├── ML_Sporting_Director.ipynb   # Data preparation, features, KNN, analysis
├── Reports/
│   └── MidStage_Report.pdf      # IEEE-style mid-stage research report
├── .gitignore
└── README.md
```

**Research report:** [Machine Learning for European Football Recruitment — Mid-Stage Report](Reports/MidStage_Report.pdf)

## Running the analysis

1. Clone the repository:
   ```bash
   git clone https://github.com/koufosa/football-player-similarity.git
   cd football-player-similarity
   ```
2. Create a Python environment and install the main dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install jupyter pandas numpy scikit-learn matplotlib
   ```
3. Download the Wyscout datasets separately and update the notebook's file paths to match their local location.
4. Launch Jupyter and run the notebook in order:
   ```bash
   jupyter notebook ML_Sporting_Director.ipynb
   ```

The notebook may require additional packages or local path adjustments depending on your environment. It is a research notebook, not a packaged application.

## Limitations and future work

- The dataset covers one Premier League season, excluding domestic cup and European competition matches.
- Playing time is estimated from substitution information and generally excludes stoppage time.
- The model groups all defenders together rather than separating full-backs and center-backs.
- Six event-based features cannot capture leadership, off-ball movement, tactical instructions, or full recruitment suitability.
- Walker-Peters is considered only as an exploratory comparison because he does not meet the main minutes threshold.

Next steps include **position-specific feature weighting**, additional engineered performance features, and more detailed evaluation of player-role and club-specific fit.

## Authors

**Alessandro Koufos** and **Yunting Zhou**  
Stevens Institute of Technology — M.S. Machine Learning research project

## Acknowledgment

This project uses Wyscout's publicly described football event-data resources. The associated research report cites the relevant academic literature.
