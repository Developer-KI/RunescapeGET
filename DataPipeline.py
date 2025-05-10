import pandas as pd
import numpy as np

from datetime import datetime, timezone


unix_timestamp_1 = 1427500800
unix_timestamp_1 = 1427587200
unix_timestamp_2 = 1746853118

utc_time = datetime.fromtimestamp(unix_timestamp_1, timezone.utc)
print(utc_time.strftime(format="%d/%m/%Y, %H:%M:%S"))  # "2025-05-10T04:58:38"