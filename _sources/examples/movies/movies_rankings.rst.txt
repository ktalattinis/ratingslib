Movie Rankings
==============
This example illustrates the use of rating systems for movie rankings.
The dataset used [1]_ is obtained from: https://grouplens.org/datasets/movielens/ and it has the following details:
Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
More details for the dataset used can be found at: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

.. [1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

In the following code we demonstrate how to use the library to rate and rank movies with different methods.
Finally we aggregate rating lists into one.

Python code
***********

.. literalinclude:: ../../../../examples/movies/movies_rankings.py
   :language: python
   :linenos: