import pandas as pd
import numpy as np

from datetime import datetime, timezone


unix_timestamp_1 = 1746884750
unix_timestamp_1 = 1746884750
unix_timestamp_2 = int(datetime.now().timestamp())
print(unix_timestamp_2)

utc_time = datetime.fromtimestamp(unix_timestamp_1, timezone.utc)
print(utc_time.strftime(format="%d/%m/%Y, %H:%M:%S"))  # "2025-05-10T04:58:38"