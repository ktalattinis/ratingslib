"""Example how to get and store data"""
# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

from apprate.datasets.filenames import download_and_store_footballdata
from apprate.datasets.soccer import championships

epl = championships.PREMIERLEAGUE  # ENGLISH PREMIER LEAGUE
download_and_store_footballdata(2005, 2018, championship=epl)
