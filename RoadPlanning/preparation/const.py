ALL_BBOX = {
    "shanghai": {
        "max_lat": 31.389062880586803,
        "min_lat": 31.100362963914506,
        "min_lon": 121.31295621599826,
        "max_lon": 121.67569392683028,
    },
    "paris": {
        "max_lon": 2.5142365112450342,
        "min_lon": 2.131314954259073,
        "min_lat": 48.74455236986203,
        "max_lat": 48.94873753185309,
    },
    "newyork": {
        "max_lon": -73.69676042687311,
        "min_lon": -74.05831067915697,
        "min_lat": 40.56692085062503,
        "max_lat": 40.94071553837341,
    },
    "beijing": {
        "max_lat": 40.13102651149333,
        "min_lat": 39.77077850431131,
        "min_lon": 116.15766988498272,
        "max_lon": 116.62575789722398,
    },
}
GRID_LEN = 2000
# from https://download.geofabrik.de/
CITY_TO_2019_PBF = {
    "shanghai": "./data/china-190101.osm.pbf",
    "beijing": "./data/china-190101.osm.pbf",
    "newyork": "./data/us-northeast-190101.osm.pbf",
    "paris": "./data/france-190101.osm.pbf",
}
FILTER_HIGHWAYS = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
]
