from const import *
from geojson import Feature, FeatureCollection, LineString, dump
from pyrosm import OSM
from shapely.geometry import LineString, Polygon
from tqdm import tqdm


def city_bound(bbox):
    min_lon, min_lat = bbox["min_lon"], bbox["min_lat"]
    max_lon, max_lat = bbox["max_lon"], bbox["max_lat"]
    return Polygon(
        [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]
    )


for city, bbox in ALL_BBOX.items():
    # 2019 net
    osm_2019 = OSM(filepath=CITY_TO_2019_PBF[city], bounding_box=city_bound(bbox))
    net_work_2019 = osm_2019.get_network(network_type="driving", nodes=True)
    nodes_2019, edges_2019 = net_work_2019  # type:ignore
    topo = []
    for index, row in tqdm(edges_2019.iterrows()):  # type:ignore
        geo = row["geometry"]
        highway = row["highway"]
        if highway not in FILTER_HIGHWAYS:
            continue
        prop = {}
        for prop_key in [
            "highway",
            "id",
            "lanes",
            "maxspeed",
            "name",
            "oneway",
        ]:
            prop[prop_key] = row[prop_key]
        topo.append(
            Feature(
                geometry=LineString([list(c) for c in geo.coords]),
                properties=prop,
            )
        )
    path_2019 = f"./data/OSM_2019_roadnet_{city}.geojson"
    topo = FeatureCollection(topo)
    with open(path_2019, encoding="utf-8", mode="w") as f:
        dump(topo, f, indent=2, ensure_ascii=False)
