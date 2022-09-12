Foresight and Hindsight Predictions
===================================
An example of code is provided to illustrate how to turn ratings into predictions of the winner of soccer outcome.

Teams ratings are produced by the following methods:

   * WinLoss
   * Colley
   * Massey
   * Elo (win version)
   * Elo (points version)
   * Keener
   * OffenseDefense
   * AccuRATE
   * GeM

Two type of predictions for the final oucome are made: 

   * hindsight - predicting past matches using the ratings of teams (the first 20 matches of EPL 2018-2019)
   * foresight - predicting upcoming matches (the 3rd match week of EPL 2018-2019) using the ratings of teams for previous weeks (the first 20 matches of EPL 2018-2019)


Data (hindsight)
   :ref:`soccer_data_20first`
Data (foresight)
   :ref:`soccer_data_third`


***********
Python code
***********

.. literalinclude:: ../../../../examples/sports/soccer_predictions.py
   :language: python
   :linenos:

*******
Results
*******

Below are shown the results of hindsight and foresight predictions:

.. code-block:: console

   =====HINDSIGHT RESULTS=====


   =====MLE=====


   =====Accuracy results=====

                                 Accuracy  Correct Games  Wrong Games  Total Games                                                                                           Predictions
   Winloss[normalization=False]      0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   Colley                            0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   Massey[data_limit=10]             0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/2, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   EloWin[HA=0_K=40_ks=400]          0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   EloPoint[HA=0_K=40_ks=400]        0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   Keener[normalization=False]       0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   OffenseDefense[tol=0.0001]        0.55             11            9           20  [1/1, 1/1, 2/1, 2/1, 2/1, 1/1, 3/1, 2/1, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/1, 1/1, 2/1, 1/1, 2/2]
   Markov[b=0.85]                    0.65             13            7           20  [1/1, 1/1, 2/1, 2/2, 2/2, 1/1, 3/2, 2/1, 1/1, 3/2, 3/2, 1/1, 1/1, 1/1, 1/1, 2/1, 1/2, 2/2, 1/1, 2/2]
   AccuRATE                          0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]

   *** Predictions columns notation: (Predicted / Actual), 1 = Home Win, 2 = Away Win, 3 = Draw


   =====RANK=====


   =====Accuracy results=====

                                 Accuracy  Correct Games  Wrong Games  Total Games                                                                                           Predictions
   Winloss[normalization=False]      0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   Colley                            0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   Massey[data_limit=10]             0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/2, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   EloWin[HA=0_K=40_ks=400]          0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   EloPoint[HA=0_K=40_ks=400]        0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   Keener[normalization=False]       0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]
   OffenseDefense[tol=0.0001]        0.55             11            9           20  [1/1, 1/1, 2/1, 2/1, 2/1, 1/1, 3/1, 2/1, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/1, 1/1, 2/1, 1/1, 2/2]
   Markov[b=0.85]                    0.65             13            7           20  [1/1, 1/1, 2/1, 2/2, 2/2, 1/1, 3/2, 2/1, 1/1, 3/2, 3/2, 1/1, 1/1, 1/1, 1/1, 2/1, 1/2, 2/2, 1/1, 2/2]
   AccuRATE                          0.85             17            3           20  [1/1, 1/1, 2/2, 2/2, 2/2, 1/1, 3/2, 2/2, 1/1, 3/1, 3/1, 1/1, 1/1, 1/1, 1/1, 2/2, 1/1, 2/2, 1/1, 2/2]

   *** Predictions columns notation: (Predicted / Actual), 1 = Home Win, 2 = Away Win, 3 = Draw


   =====FORESIGHT RESULTS=====


   =====MLE=====


   =====Accuracy results=====

                              Accuracy Correct Games Wrong Games  Total Games                                         Predictions
   Winloss[normalization=False]      0.7             7           3           10  [1/1, 1/3, 1/3, 1/1, 2/2, 2/3, 1/1, 2/2, 1/1, 2/2]
   Colley                            0.5             5           5           10  [1/1, 1/3, 1/3, 1/1, 1/2, 2/3, 1/1, 2/2, 1/1, 1/2]
   Massey[data_limit=10]              NA            NA          NA            0                                                  NA
   EloWin[HA=0_K=40_ks=400]          0.5             5           5           10  [1/1, 1/3, 1/3, 1/1, 1/2, 2/3, 1/1, 2/2, 1/1, 1/2]
   EloPoint[HA=0_K=40_ks=400]        0.5             5           5           10  [1/1, 1/3, 1/3, 1/1, 1/2, 2/3, 1/1, 2/2, 1/1, 1/2]
   Keener[normalization=False]       0.5             5           5           10  [1/1, 1/3, 1/3, 1/1, 1/2, 2/3, 1/1, 2/2, 1/1, 1/2]
   OffenseDefense[tol=0.0001]        0.4             4           6           10  [1/1, 1/3, 1/3, 1/1, 1/2, 1/3, 1/1, 1/2, 1/1, 1/2]
   Markov[b=0.85]                    0.4             4           6           10  [1/1, 1/3, 1/3, 1/1, 1/2, 1/3, 1/1, 1/2, 1/1, 1/2]
   AccuRATE                          0.4             4           6           10  [1/1, 1/3, 2/3, 1/1, 1/2, 2/3, 1/1, 1/2, 1/1, 1/2]

   *** Predictions columns notation: (Predicted / Actual), 1 = Home Win, 2 = Away Win, 3 = Draw


   =====RANK=====


   =====Accuracy results=====

                                 Accuracy  Correct Games  Wrong Games  Total Games                                         Predictions
   Winloss[normalization=False]       0.6              6            4           10  [3/1, 1/3, 3/3, 1/1, 2/2, 2/3, 3/1, 2/2, 1/1, 2/2]
   Colley                             0.5              5            5           10  [2/1, 1/3, 2/3, 1/1, 2/2, 2/3, 2/1, 2/2, 1/1, 2/2]
   Massey[data_limit=10]              0.4              4            6           10  [2/1, 1/3, 2/3, 1/1, 2/2, 2/3, 1/1, 1/2, 2/1, 2/2]
   EloWin[HA=0_K=40_ks=400]           0.5              5            5           10  [3/1, 1/3, 2/3, 1/1, 2/2, 2/3, 2/1, 2/2, 1/1, 2/2]
   EloPoint[HA=0_K=40_ks=400]         0.6              6            4           10  [1/1, 1/3, 2/3, 1/1, 2/2, 2/3, 2/1, 2/2, 1/1, 2/2]
   Keener[normalization=False]        0.6              6            4           10  [1/1, 1/3, 2/3, 1/1, 2/2, 2/3, 2/1, 2/2, 1/1, 2/2]
   OffenseDefense[tol=0.0001]         0.4              4            6           10  [2/1, 1/3, 1/3, 1/1, 2/2, 2/3, 2/1, 2/2, 1/1, 1/2]
   Markov[b=0.85]                     0.4              4            6           10  [1/1, 2/3, 1/3, 2/1, 2/2, 1/3, 2/1, 2/2, 1/1, 1/2]
   AccuRATE                           0.6              6            4           10  [1/1, 1/3, 2/3, 1/1, 2/2, 2/3, 2/1, 2/2, 1/1, 2/2]

   *** Predictions columns notation: (Predicted / Actual), 1 = Home Win, 2 = Away Win, 3 = Draw


