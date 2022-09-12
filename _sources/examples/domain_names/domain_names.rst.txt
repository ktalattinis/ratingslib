Domain Names Ranking
====================
In this application we present modifications of the sports teams rating methods, in order to use them in ranking of domain names.
This kind of ranking is considered important because it is associated with the formation of the price at which domain names can be sold.
Specifically, these two figures are proportional amounts, i.e., the higher the rank of a domain name, the higher its selling price will be. 
This is an illustrative example that is included in the paper [1]_:


.. [1] Talattinis, K., Zervopoulou, C., & Stephanides, G. (2014, June). Ranking Domain Names Using Various Rating Methods. Proceedings of the Ninth International Multi-Conference on Computing in the Global Information Technology (pp. 107-114). Seville: IARIA.

In this example, there are five domain names that have been sold in early 2014,
which are jean.com, desirous.com, authorization.com, true.com and finally, peaked.com.
We will attempt to rank these domains based on search volume average they get by
Google trends during 2013.


.. note::

   The following statistics have been selected:

    * TrendsI = score of google trends relative search volume for domain name i
    * TrendsJ = score of google trends relative search volume for domain name j


.. rubric:: Domain Names Example
.. _label1:
.. csv-table::
   :file: ../../../../ratingslib/datasets/examples/domainMarketExample.csv
   :header-rows: 1

.. toctree:: 
   notebooks/domainmarket