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



# Names are used only to check administrative identity; coordinates still come

# from Nominatim or the existing trusted-location entries.

# Administrative spelling aliases.  These names map identity, NEVER coordinates.
# The 24 governorates are covered; localities below are common examples, not
# an exhaustive catalogue of every delegation, neighborhood or street.
AREA_ALIASES = {
    "tunis": {"tunis", "تونس"},
    "ariana": {"ariana", "aryanah", "أريانة", "اريانة"},
    "ben arous": {"ben arous", "ben arus", "بن عروس"},
    "manouba": {"manouba", "mannouba", "manuba", "منوبة"},
    "nabeul": {"nabeul", "nabul", "نابل"},
    "zaghouan": {"zaghouan", "zaghwan", "زغوان"},
    "bizerte": {"bizerte", "bizerta", "بنزرت"},
    "beja": {"beja", "béja", "باجة"},
    "jendouba": {"jendouba", "jenduba", "jandouba", "جندوبة"},
    "kef": {"le kef", "el kef", "kef", "الكاف"},
    "siliana": {"siliana", "silyana", "سليانة"},
    "sousse": {"sousse", "susa", "سوسة"},
    "monastir": {"monastir", "mounastir", "المنستير"},
    "mahdia": {"mahdia", "المهدية"},
    "sfax": {"sfax", "safaqis", "صفاقس"},
    "kairouan": {"kairouan", "kayrawan", "القيروان"},
    "kasserine": {"kasserine", "القصرين"},
    "sidi bouzid": {"sidi bouzid", "sidi bou zid", "سيدي بوزيد"},
    "gabes": {"gabes", "gabès", "قابس"},
    "medenine": {"medenine", "médenine", "مدنين"},
    "tataouine": {"tataouine", "tatawin", "تطاوين"},
    "gafsa": {"gafsa", "قفصة"},
    "tozeur": {"tozeur", "tozur", "توزر"},
    "kebili": {"kebili", "kébili", "قبلي"},
    # Common cities, towns, and delegation names.  Nominatim supplies coordinates.
    "hammamet": {"hammamet", "الحمامات"},
    "sakiet ezzit": {"sakiet ezzit", "sakiet ez zit", "ساقية الزيت"},
    "sakiet eddaier": {"sakiet eddaier", "sakiet ed daier", "ساقية الدائر"},
    "la marsa": {"la marsa", "marsa", "المرسى"},
    "carthage": {"carthage", "قرطاج"},
    "la goulette": {"la goulette", "halq al wadi", "حلق الوادي"},
    "rades": {"rades", "radès", "رادس"},
    "ezzahra": {"ezzahra", "الزهراء"},
    "hammam lif": {"hammam lif", "حمام الأنف"},
    "hammam sousse": {"hammam sousse", "حمام سوسة"},
    "msaken": {"msaken", "m'saken", "مساكن"},
    "akouda": {"akouda", "أكودة", "اكودة"},
    "kalaa kebira": {"kalaa kebira", "kalâa kebira", "القلعة الكبرى"},
    "kalaa seghira": {"kalaa seghira", "kalâa seghira", "القلعة الصغرى"},
    "moknine": {"moknine", "المكنين"},
    "ksar hellal": {"ksar hellal", "قصر هلال"},
    "jemmal": {"jemmal", "جمال"},
    "el jem": {"el jem", "el djem", "الجم"},
    "chebba": {"chebba", "الشابة"},
    "kerkennah": {"kerkennah", "kerkena", "قرقنة"},
    "mahares": {"mahares", "mahrès", "المحرس"},
    "agareb": {"agareb", "عقارب"},
    "sbeitla": {"sbeitla", "سبيطلة"},
    "feriana": {"feriana", "فريانة"},
    "metlaoui": {"metlaoui", "المتلوي"},
    "redayef": {"redayef", "الرديف"},
    "douz": {"douz", "دوز"},
    "nefta": {"nefta", "نفطة"},
    "djerba": {"djerba", "jerba", "جربة"},
    "houmt souk": {"houmt souk", "حومة السوق"},
    "zarzis": {"zarzis", "جرجيس"},
    "ben gardane": {"ben gardane", "ben guerdane", "بنقردان", "بن قردان"},
    "el haouaria": {"el haouaria", "haouaria", "الهوارية"},
    "kelibia": {"kelibia", "kélibia", "قليبية"},
    "korba": {"korba", "قربة"},
    "menzel temime": {"menzel temime", "منزل تميم"},
    "menzel bourguiba": {"menzel bourguiba", "منزل بورقيبة"},
    "ras jebel": {"ras jebel", "راس الجبل", "رأس الجبل"},
    "tabarka": {"tabarka", "طبرقة"},
    "ain draham": {"ain draham", "aïn draham", "عين دراهم"},
}

# A query containing a name not in the alias table is still supported via
# Nominatim. Unknown names are not silently assigned to any governorate.

AREA_FIELDS = (

    "city", "town", "village", "municipality", "county", "state",

    "state_district", "province", "region",

)

LOCALITY_FIELDS = ("city", "town", "village", "municipality")

GEOGRAPHIC_TYPES = {

    "city", "town", "village", "hamlet", "suburb", "neighbourhood",

    "neighborhood", "quarter", "borough", "municipality", "county",

    "administrative", "residential", "city_block", "locality",

}





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





def _area_key(value: str) -> str:
    """Normalize multilingual area spellings to a canonical identity."""
    text = _alias_key(value)
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}))
    text = re.sub(
        r"^(?:ولاية|معتمدية|gouvernorat de|gouvernorat d|governorate of|"
        r"governorate|ville de|city of)\s+", "", text,
    ).strip()
    text = re.sub(r"\s+(?:governorate|gouvernorat)$", "", text).strip()
    for canonical, names in AREA_ALIASES.items():
        normalized = {_alias_key(canonical)}
        normalized.update(_alias_key(name) for name in names)
        normalized = {re.sub(r"[\u064b-\u065f\u0670\u0640]", "", name).translate(
            str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
        ) for name in normalized}
        if text in normalized:
            return canonical
    return text


CENTRE_PATTERN = re.compile(
    r"\b(?:city|ville)\s+cent(?:er|re)\b|\b(?:centre|center)[ -]ville\b|وسط\s+المدينة",
    re.I,
)


def _known_area(value: str) -> bool:
    return _area_key(value) in AREA_ALIASES


def _query_area(location: str) -> tuple[str | None, bool]:
    """Extract a known area and detect requests for a geographical city centre.

    A business such as 'City Centre Mall, Hammamet' is not a city-centre
    request; the words *city centre* there are part of a commercial name.
    """
    parts = [part.strip() for part in location.split(",")]
    parts = [part for part in parts if _area_key(part) not in {"tunisia", "tunisie", "تونس الجمهورية"}]
    if not parts:
        return None, False
    first = parts[0]
    stripped = CENTRE_PATTERN.sub("", first).strip(" ,-")
    centre = bool(CENTRE_PATTERN.search(first)) and (
        not stripped or _known_area(stripped)
        or (len(parts) > 1 and _area_key(first) in {"city centre", "city center", "centre ville", "center ville", "وسط المدينة"})
    )
    if centre:
        if stripped and _known_area(stripped):
            return _area_key(stripped), True
        # 'city center, Tunis' is meaningful; 'city center' alone is not.
        if len(parts) > 1 and _known_area(parts[-1]):
            return _area_key(parts[-1]), True
        return None, True
    if len(parts) > 1 and _known_area(parts[-1]):
        return _area_key(parts[-1]), False
    if len(parts) == 1 and _known_area(first):
        return _area_key(first), False
    return None, False


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
        try:
            resolved = self._select_candidate(raw_input, candidates)
        except LocationNotFoundError:
            area, city_centre = _query_area(raw_input)
            if not city_centre or area is None:
                raise
            # Nominatim does not always index a separate centre POI. Resolve
            # the requested city itself; do not accept a similarly named shop.
            fallback_candidates = self._search(f"{area}, Tunisia")
            resolved = self._select_candidate(area, fallback_candidates)
        self._cache[cache_key] = replace(resolved, raw_input=raw_input)
        return self._cache[cache_key]



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

        requested_area, centre_query = _query_area(location)

        first_part = location.split(",", maxsplit=1)[0].strip()

        centre_prefix = CENTRE_PATTERN.sub("", first_part).strip(" ,- ")
        city_intent = requested_area is not None and (
            (not centre_query and _area_key(first_part) == requested_area)
            or (centre_query and _area_key(centre_prefix) == requested_area)
            or (centre_query and _area_key(first_part) in {"city centre", "city center", "centre ville", "center ville", "وسط المدينة"})
        )

        if centre_query and requested_area is None:

            raise LocationNotFoundError(

                f"Location '{location}' needs a city or region to identify its centre."

            )

        scored: list[ResolvedLocation] = []



        for candidate in candidates:

            address = candidate.get("address")

            if not isinstance(address, Mapping):

                continue

            if str(address.get("country_code", "")).casefold() != "tn":

                continue



            administrative_areas = {

                _area_key(str(address[field]))

                for field in AREA_FIELDS

                if isinstance(address.get(field), str) and address[field].strip()

            }

            # Structured administration outranks a business name or a road

            # name that merely contains the requested city's word.

            if requested_area and administrative_areas:

                if requested_area not in administrative_areas:

                    continue

                locality_areas = {

                    _area_key(str(address[field]))

                    for field in LOCALITY_FIELDS

                    if isinstance(address.get(field), str) and address[field].strip()

                }

                if city_intent and locality_areas and requested_area not in locality_areas:

                    continue



            category = str(candidate.get("category") or candidate.get("class") or "").casefold()

            place_type = str(candidate.get("type") or "").casefold()

            if city_intent and not (

                category in {"place", "boundary"}

                and place_type in GEOGRAPHIC_TYPES

            ):

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



            if requested_area and not administrative_areas:

                # Old/incomplete Nominatim records may omit address details.

                # Only accept them if the displayed administrative components

                # explicitly contain the requested area.

                components = {

                    _area_key(part.strip()) for part in display_name.split(",")

                }

                if requested_area not in components:

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

            if city_intent and category == "place" and place_type in {"city", "town"}:

                confidence += 0.12

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
