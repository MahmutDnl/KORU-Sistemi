import math

def geodetic_to_ecef(lat_deg, lon_deg, h):
    """
    Enlem, boylam, yükseklik -> ECEF (X, Y, Z)
    lat_deg : derece cinsinden enlem
    lon_deg : derece cinsinden boylam
    h       : metre cinsinden yükseklik
    """

    # WGS84 sabitleri
    a = 6378137.0              # yarı büyük eksen (m)
    f = 1 / 298.257223563      # basıklık
    e2 = f * (2 - f)           # eksantriklik^2

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))

    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + h) * math.sin(lat)

    return x, y, z


def space_distance(lat1, lon1, h1, lat2, lon2, h2):
    """
    İki uzay noktası arasındaki 3B doğrusal mesafe (metre)
    """
    x1, y1, z1 = geodetic_to_ecef(lat1, lon1, h1)
    x2, y2, z2 = geodetic_to_ecef(lat2, lon2, h2)

    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return d


# Örnek kullanım
lat1, lon1, h1 = -0.0522631966050954, 41.966697015316285, 35766.09895458028  # 400 km yükseklik
lat2, lon2, h2 = -0.0506491478046723, 41.95351109886105, 35768.10180690996   # 410 km yükseklik

mesafe_m = space_distance(lat1, lon1, h1, lat2, lon2, h2)
mesafe_km = mesafe_m / 1000

print(f"Mesafe: {mesafe_m:.2f} metre")
print(f"Mesafe: {mesafe_km:.2f} km")