from time import sleep
from datetime import datetime, time
import sys

hour = int(sys.argv[1])
minute = int(sys.argv[2])

time_to_stop = time(hour,minute)
print(f"Waiting until {time_to_stop}", flush=True)

while datetime.now().time() < time(hour,minute):
    sleep(20)