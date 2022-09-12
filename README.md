<img src="docs/source/images/RatingsLib.png" width="50%" />

# RatingsLib: A python library for rating methods with applications

RatingsLib is a Python library dedicated to rating/ranking systems implementation 
with applications in sports and other fields. 

## Installation

RatingsLib requires Python 3.8 or newer. More details about requirements can be found in ``requirements.txt``.
You can install ``ratingslib`` directly ::
```
 pip install git+https://github.com/ktalattinis/ratingslib
```
or by cloning the repository ::
```
 git clone https://github.com/ktalattinis/ratingslib
 cd ratingslib
 pip install .
```
## Implementation
Rating/Ranking systems:
 * WinLoss
 * Colley
 * Massey
 * Keener
 * Elo
 * Offense - Defense
 * GeM
 * AccuRATE

Ranking Aggregation methods:
 * Borda Count
 * Average Rank

Rating Aggregation methods:
 * Markov
 * Perron
 * Offense-Defense

Comparison metrics:
 * Kendall's Tau

Applications & Examples:
   * Sports (the main application of the library):
      * Soccer Teams rating
      * Soccer Teams ranking lists comparison
      * Hindsight and foresight prediction of the final outcome of soccer matches
      * Combining rating systems and machine learning methods to predict soccer matches outcome
      * Ranking NFL teams

   * Other Applications & Examples:
      * Finance:
        * Examples from investment selection and portfolios rating and ranking.
      * Domain Market:
        * An illustrative example is provided and shows the ranking of domain names.
      * Movies:
        * Application on real-world dataset from [MovieLens](https://grouplens.org/datasets/movielens/)

## Documentation
The documentation is available at: https://ktalattinis.github.io/ratingslib/





