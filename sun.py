import numpy as np
import pytz
import ephem
import datetime


class Landmark(ephem.Observer):
    def __init__(self, lat, lon, elevation, timezone):
        self.lat = str(lat)
        self.lon = str(lon)
        self.elevation = elevation
        self.timezone = timezone

        # make utcoffset easy to access
        # (we assume no DST, so UTC offset is not date-dependent)
        dummy_dt = datetime.datetime(2010,1,1)
        utcoff = self.timezone.utcoffset(dummy_dt)
        self.utcoffset = int(round(utcoff.total_seconds() / 3600))

    def get_angle(self, timestamp):
        # make the input timestamp aware, then convert to utc
        sun = ephem.Sun()
        self.date = timestamp.replace(tzinfo=self.timezone).astimezone(pytz.utc)
        sun.compute(self)
        return sun.az, sun.alt

    def get_angles(self, date):
        midnight_local = datetime.datetime(*date, tzinfo=self.timezone)
        minute = datetime.timedelta(minutes=1)

        times = [midnight_local + minute*i for i in range(24*60)]
        angles = np.zeros((24*60, 2))
        sun = ephem.Sun()
        for i in range(24*60):
            self.date = times[i].astimezone(pytz.utc)
            sun.compute(self)
            angles[i,:] = sun.az, sun.alt
        return times, angles

    #   def _sunrise(self,date):
    #       midnight_local = datetime.datetime(*date, tzinfo=self.timezone)
    #       self.date = midnight_local.astimezone(pytz.utc)
    #       return self.next_rising(ephem.Sun())

    #   def _sunset(self,date):
    #       date = date + (12,) # make it noon
    #       noon_local = datetime.datetime(*date, tzinfo=self.timezone)
    #       self.date = noon_local.astimezone(pytz.utc)
    #       return self.next_setting(ephem.Sun())


def azel2enu(az, el):
    x = np.cos(el) * np.sin(az)
    y = np.cos(el) * np.cos(az)
    z = np.sin(el)
    return np.array([x,y,z])


def enu2azel(e,n,u):
    az = (np.arctan2(e,n) + 2*np.pi) % (2*np.pi)
    el = np.arctan2(u, np.sqrt(e**2 + n**2))
    return az, el
