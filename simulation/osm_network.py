"""Prepare a Valhalla route as an OSMnx-compatible road graph."""

from __future__ import annotations

import math
from collections.abc import Sequence

import networkx as nx

RoutePoint = tuple[float, float]


def distance_km(point_a: RoutePoint, point_b: RoutePoint) -> float:
    """Return the great-circle distance between two latitude/longitude points."""

    latitude_a, longitude_a = map(math.radians, point_a)
    latitude_b, longitude_b = map(math.radians, point_b)
    latitude_delta = latitude_b - latitude_a
    longitude_delta = longitude_b - longitude_a
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a)
        * math.cos(latitude_b)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(haversine))


def simplify_route(
    route_points: Sequence[RoutePoint], spacing_km: float = 0.25
) -> list[RoutePoint]:
    """Keep route points at approximately ``spacing_km`` intervals."""

    if len(route_points) < 2:
        raise ValueError("A route requires at least two points.")
    if spacing_km <= 0:
        raise ValueError("spacing_km must be greater than zero.")

    simplified = [route_points[0]]
    distance_since_last_point = 0.0
    previous = route_points[0]

    for point in route_points[1:-1]:
        distance_since_last_point += distance_km(previous, point)
        if distance_since_last_point >= spacing_km:
            simplified.append(point)
            distance_since_last_point = 0.0
        previous = point

    simplified.append(route_points[-1])
    return simplified


def build_road_network(
    route_points: Sequence[RoutePoint],
    route_distance_km: float,
    duration_minutes: float,
    *,
    spacing_km: float = 0.25,
    lanes: int = 2,
) -> nx.MultiDiGraph:
    """Build a directed, OSMnx-compatible graph from Valhalla geometry."""

    if route_distance_km <= 0:
        raise ValueError("route_distance_km must be greater than zero.")
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be greater than zero.")
    if lanes < 1:
        raise ValueError("lanes must be at least one.")

    graph_points = simplify_route(route_points, spacing_km)
    average_speed_kph = route_distance_km / (duration_minutes / 60)
    graph = nx.MultiDiGraph()
    graph.graph.update(
        {
            "crs": "EPSG:4326",
            "route_nodes": list(range(len(graph_points))),
            "route_distance_km": float(route_distance_km),
            "duration_minutes": float(duration_minutes),
        }
    )

    for node_id, (latitude, longitude) in enumerate(graph_points):
        graph.add_node(node_id, x=float(longitude), y=float(latitude))

    for start_node in range(len(graph_points) - 1):
        end_node = start_node + 1
        segment_length_m = max(
            distance_km(graph_points[start_node], graph_points[end_node]) * 1000,
            1.0,
        )
        graph.add_edge(
            start_node,
            end_node,
            key=0,
            length=segment_length_m,
            speed_kph=average_speed_kph,
            travel_time=segment_length_m / (average_speed_kph / 3.6),
            lanes=lanes,
            highway="valhalla_route",
        )

    return graph


def get_route_nodes(graph: nx.MultiDiGraph) -> list[int]:
    """Return ordered node identifiers from a prepared route graph."""

    route_nodes = graph.graph.get("route_nodes")
    if not route_nodes:
        raise ValueError("The graph does not contain prepared route nodes.")
    return list(route_nodes)
