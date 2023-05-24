from datetime import datetime
import time

initial = datetime.now()
print(initial)
time.sleep(1)
final = datetime.now()
print(final)
print("elapsed time is ")
elapsed = final-initial
print(elapsed.total_seconds()*1000)