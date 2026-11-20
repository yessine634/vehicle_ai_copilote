"""Resolve user-provided Tunisian locations with validated OSM results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import math
import re
import time
from typing import Any, Callable, Mapping
import unicodedata

import requests


NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
DEFAULT_USER_AGENT = "vehicle-ai-copilot/1.0"
DEFAULT_SFAX_REFERENCE = (34.7406, 10.7603)
DEFAULT_SFAX_MAX_RADIUS_KM = 50.0


def _normalize_text(value: str) -> str:
    """Normalize case and whitespace for cache keys and text comparison."""

    return " ".join(value.casefold().split())


def _comparison_text(value: str) -> str:
    """Normalize punctuation while retaining letters, numbers, and accents."""

    return " ".join(
        token
        for token in re.split(r"[^\w]+", _normalize_text(value))
        if token
    )


@dataclass(frozen=True)
class ResolvedLocation:
    """One validated geographic point ready for a routing service."""

    raw_input: str
    canonical_name: str
    latitude: float
    longitude: float
    source: str
    confidence: float | None = None


def _alias_key(value: str) -> str:
    """Normalize common French/English spellings for trusted aliases."""

    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    normalized = _comparison_text(ascii_value)
    normalized = re.sub(r"\bkilomet(?:er|re)s?\b", "km", normalized)
    normalized = re.sub(r"\bkm\s+(\d+)\b", r"km\1", normalized)
    normalized = re.sub(r"\b(?:tunisia|tunisie)\b", "", normalized)
    return " ".join(normalized.split())


def _trusted_location(
    canonical_name: str,
    latitude: float,
    longitude: float,
) -> ResolvedLocation:
    return ResolvedLocation(
        raw_input=canonical_name,
        canonical_name=canonical_name,
        latitude=latitude,
        longitude=longitude,
        source="trusted_location",
        confidence=1.0,
    )


DEFAULT_TRUSTED_LOCATIONS: dict[str, ResolvedLocation] = {
    _alias_key(alias): _trusted_location(
        "Route El Ain Km 4, Sfax, Tunisia",
        34.7518952,
        10.7296079,
    )
    for alias in (
        "Route El Ain Km 4, Sfax",
        "Route El Aïn Km 4, Sfax",
        "Route de l'Ain Km 4, Sfax",
        "Route de l'Aïn Km 4, Sfax",
        "El Ain Km 4, Sfax",
    )
}
DEFAULT_TRUSTED_LOCATIONS.update(
    {
        _alias_key(alias): _trusted_location(
            "ENET'Com, Technopôle de Sfax, Sakiet Ezzit, Tunisia",
            34.839620,
            10.756662,
        )
        for alias in (
            "ENETCom, Sfax",
            "ENET'Com, Sfax",
            "ENETCom, Cité El Ons, Sfax",
            "ENET'Com, Cité El Ons, Sakiet Ezzit, Sfax",
            "École Nationale d'Électronique et des Télécommunications de Sfax",
        )
    }
)

# Reassign the practical ENET'Com aliases with Windows-console-safe canonical
# text. Accent-insensitive alias normalization still accepts Cité/Technopôle.
for _enetcom_alias in (
    "ENETCom, Sfax",
    "ENET'Com, Sfax",
    "ENETCom, Cite El Ons, Sfax",
    "ENET'Com, Cite El Ons, Sakiet Ezzit, Sfax",
):
    DEFAULT_TRUSTED_LOCATIONS[_alias_key(_enetcom_alias)] = _trusted_location(
        "ENET'Com, Technopole de Sfax, Sakiet Ezzit, Tunisia",
        34.839620,
        10.756662,
    )


class LocationResolutionError(RuntimeError):
    """Base error for deterministic location-resolution failures."""


class LocationNotFoundError(LocationResolutionError):
    """Raised when no validated candidate can resolve a location."""


class InvalidCoordinatesError(LocationResolutionError, ValueError):
    """Raised when direct coordinates are malformed or outside valid ranges."""


class AmbiguousLocationError(LocationResolutionError):
    """Raised when the best candidates are too similar to choose safely."""

    def __init__(
        self,
        location: str,
        candidates: list["ResolvedLocation"],
    ) -> None:
        self.location = location
        self.candidates = candidates
        options = "; ".join(
            f"{item.canonical_name} "
            f"({item.latitude:.6f}, {item.longitude:.6f})"
            for item in candidates
        )
        super().__init__(
            f"Location '{location}' is ambiguous. Best candidates: {options}"
        )


def haversine_km(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """Return the great-circle distance between two coordinate pairs."""

    latitude_a, longitude_a = map(math.radians, point_a)
    latitude_b, longitude_b = map(math.radians, point_b)
    latitude_delta = latitude_b - latitude_a
    longitude_delta = longitude_b - longitude_a
    haversine = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(latitude_a)
        * math.cos(latitude_b)
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * math.asin(math.sqrt(haversine))


def _coordinate_parts(value: str) -> tuple[float, float] | None:
    """Parse direct coordinates or identify malformed coordinate-like input."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        return None

    number_pattern = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
    matches = [bool(number_pattern.fullmatch(part)) for part in parts]
    if not any(matches):
        return None
    if not all(matches):
        raise InvalidCoordinatesError(
            f"Malformed coordinates '{value}'. Expected 'latitude, longitude'."
        )

    latitude, longitude = map(float, parts)
    if not -90.0 <= latitude <= 90.0:
        raise InvalidCoordinatesError(
            f"Latitude {latitude} is outside the valid -90 to 90 range."
        )
    if not -180.0 <= longitude <= 180.0:
        raise InvalidCoordinatesError(
            f"Longitude {longitude} is outside the valid -180 to 180 range."
        )
    return latitude, longitude


class LocationResolver:
    """Resolve coordinates directly or search validated Tunisian OSM places."""

    def __init__(
        self,
        *,
        base_url: str = NOMINATIM_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 20.0,
        sfax_reference: tuple[float, float] = DEFAULT_SFAX_REFERENCE,
        sfax_max_radius_km: float = DEFAULT_SFAX_MAX_RADIUS_KM,
        ambiguity_threshold: float = 0.03,
        minimum_request_interval_seconds: float = 1.0,
        http_client: Any = requests,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        trusted_locations: Mapping[str, ResolvedLocation] | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("A non-empty identifying User-Agent is required.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        if sfax_max_radius_km <= 0:
            raise ValueError("sfax_max_radius_km must be greater than zero.")
        if ambiguity_threshold < 0:
            raise ValueError("ambiguity_threshold cannot be negative.")
        if minimum_request_interval_seconds < 0:
            raise ValueError(
                "minimum_request_interval_seconds cannot be negative."
            )

        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self.sfax_reference = sfax_reference
        self.sfax_max_radius_km = float(sfax_max_radius_km)
        self.ambiguity_threshold = float(ambiguity_threshold)
        self.minimum_request_interval_seconds = float(
            minimum_request_interval_seconds
        )
        self.http_client = http_client
        self.clock = clock
        self.sleeper = sleeper
        self.trusted_locations = dict(DEFAULT_TRUSTED_LOCATIONS)
        if trusted_locations:
            self.trusted_locations.update(
                {
                    _alias_key(alias): resolved
                    for alias, resolved in trusted_locations.items()
                }
            )
        self._cache: dict[str, ResolvedLocation] = {}
        self._last_request_at: float | None = None

    def clear_cache(self) -> None:
        """Clear resolved locations retained during this application run."""

        self._cache.clear()

    def resolve(self, location: str) -> ResolvedLocation:
        """Resolve one user string without silently discarding its context."""

        if not isinstance(location, str) or not location.strip():
            raise LocationNotFoundError("A non-empty location is required.")

        raw_input = location.strip()
        cache_key = _normalize_text(raw_input)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return replace(cached, raw_input=raw_input)

        trusted = self.trusted_locations.get(_alias_key(raw_input))
        if trusted is not None:
            resolved = replace(trusted, raw_input=raw_input)
            self._cache[cache_key] = resolved
            return resolved

        coordinates = _coordinate_parts(raw_input)
        if coordinates is not None:
            latitude, longitude = coordinates
            resolved = ResolvedLocation(
                raw_input=raw_input,
                canonical_name=f"{latitude:.7f}, {longitude:.7f}",
                latitude=latitude,
                longitude=longitude,
                source="coordinates",
                confidence=1.0,
            )
            self._cache[cache_key] = resolved
            return resolved

        candidates = self._search(raw_input)
        resolved = self._select_candidate(raw_input, candidates)
        self._cache[cache_key] = resolved
        return resolved

    def _search(self, location: str) -> list[Mapping[str, Any]]:
        """Query Nominatim once, respecting the configured minimum interval."""

        now = self.clock()
        if self._last_request_at is not None:
            remaining = (
                self.minimum_request_interval_seconds
                - (now - self._last_request_at)
            )
            if remaining > 0:
                self.sleeper(remaining)

        try:
            response = self.http_client.get(
                f"{self.base_url}/search",
                params={
                    "q": location,
                    "format": "jsonv2",
                    "countrycodes": "tn",
                    "limit": 5,
                    "addressdetails": 1,
                    "namedetails": 1,
                },
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
            self._last_request_at = self.clock()
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise LocationResolutionError(
                f"Nominatim request failed for '{location}': {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise LocationResolutionError(
                f"Nominatim returned invalid JSON for '{location}'."
            ) from exc

        if not isinstance(payload, list):
            raise LocationResolutionError(
                f"Nominatim returned an unexpected response for '{location}'."
            )
        return [item for item in payload if isinstance(item, Mapping)]

    def _select_candidate(
        self,
        location: str,
        candidates: list[Mapping[str, Any]],
    ) -> ResolvedLocation:
        """Validate, score, and safely select the best Nominatim candidate."""

        query_text = _comparison_text(location)
        query_tokens = set(query_text.split())
        expects_sfax = "sfax" in query_tokens
        scored: list[ResolvedLocation] = []

        for candidate in candidates:
            address = candidate.get("address")
            if not isinstance(address, Mapping):
                continue
            if str(address.get("country_code", "")).casefold() != "tn":
                continue

            display_name = candidate.get("display_name")
            if not isinstance(display_name, str) or not display_name.strip():
                continue
            try:
                latitude = float(candidate["lat"])
                longitude = float(candidate["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                continue

            distance_from_sfax = haversine_km(
                (latitude, longitude), self.sfax_reference
            )
            if expects_sfax and distance_from_sfax > self.sfax_max_radius_km:
                continue

            candidate_text = _comparison_text(display_name)
            candidate_tokens = set(candidate_text.split())
            sequence_score = SequenceMatcher(
                None, query_text, candidate_text
            ).ratio()
            token_score = (
                len(query_tokens & candidate_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            text_score = (sequence_score + token_score) / 2.0
            proximity_score = (
                max(0.0, 1.0 - distance_from_sfax / self.sfax_max_radius_km)
                if expects_sfax
                else 0.5
            )
            confidence = 0.8 * text_score + 0.2 * proximity_score
            scored.append(
                ResolvedLocation(
                    raw_input=location,
                    canonical_name=display_name.strip(),
                    latitude=latitude,
                    longitude=longitude,
                    source="nominatim",
                    confidence=round(confidence, 6),
                )
            )

        if not scored:
            raise LocationNotFoundError(
                f"Location '{location}' was not found in the validated "
                "Tunisia search area."
            )

        scored.sort(
            key=lambda item: item.confidence or 0.0,
            reverse=True,
        )
        if len(scored) > 1:
            first = scored[0].confidence or 0.0
            second = scored[1].confidence or 0.0
            if first - second <= self.ambiguity_threshold:
                raise AmbiguousLocationError(location, scored[:2])

        return scored[0]


__all__ = [
    "AmbiguousLocationError",
    "DEFAULT_SFAX_MAX_RADIUS_KM",
    "DEFAULT_SFAX_REFERENCE",
    "DEFAULT_TRUSTED_LOCATIONS",
    "DEFAULT_USER_AGENT",
    "InvalidCoordinatesError",
    "LocationNotFoundError",
    "LocationResolutionError",
    "LocationResolver",
    "NOMINATIM_BASE_URL",
    "ResolvedLocation",
    "haversine_km",
]
