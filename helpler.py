import time
import datetime


def ts_to_time(ts):
    x = time.localtime(ts)

    return time.strftime('%Y-%m-%d %H:%M:%S', x)

def get_nextmin(ts):
    x = datetime.datetime.fromtimestamp(ts)
    x = x+datetime.timedelta(minutes=1)
    unixtime = time.mktime(x.timetuple())
    return unixtime
if __name__ == "__main__":
    print(ts_to_time(1534318140))
    print(ts_to_time(get_nextmin(1534318140)))