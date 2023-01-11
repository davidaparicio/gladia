NER_LOOKUP = {
    "CARDINAL": {
        "long_description": "Numerals that do not fall under another type",
        "short_description": "Numerals",
        "examples": ["0", "1", "2", "20", "2017", "40", "fifty", "one-hundred"],
    },
    "DATE": {
        "long_description": "Absolute or relative dates or periods",
        "short_description": "Dates",
        "examples": [
            "today",
            "tomorrow",
            "last week",
            "27/12/2016",
            "3/14/15",
            "2017",
            "17th of March",
            "2017-18",
        ],
    },
    "EVENT": {
        "long_description": "Named hurricanes, battles, wars, sports events, etc.",
        "short_description": "Events",
        "examples": ["Hurricane Katrina", "World War II", "9/11", "Super Bowl 50"],
    },
    "FAC": {
        "long_description": "Facilities, such as buildings, airports, highways, bridges, etc.",
        "short_description": "Facilities",
        "examples": [],
    },
    "GPE": {
        "long_description": "Geopolitical entity, i.e. countries, cities, states.",
        "short_description": "Geopolitical entity",
        "examples": [
            "Germany",
            "Berlin",
            "France",
            "Paris",
            "California",
            "San Francisco",
        ],
    },
    "LANGUAGE": {
        "long_description": "Any named language",
        "short_description": "Languages",
        "examples": [
            "English",
            "French",
        ],
    },
    "LAW": {
        "long_description": "Named documents made into laws.",
        "short_description": "Laws",
        "examples": [
            "The Patriot Act",
            "The Geneva Convention",
        ],
    },
    "LOC": {
        "long_description": "Non-GPE locations, mountain ranges, bodies of water",
        "short_description": "Locations",
        "examples": [
            "the Alps",
            "the Atlantic Ocean",
        ],
    },
    "MISC": {
        "long_description": "Miscellaneous entities, e.g. events, nationalities, products or works of art",
        "short_description": "Miscellaneous",
        "examples": [
            "the Mona Lisa",
            "the Eiffel Tower",
        ],
    },
    "MONEY": {
        "long_description": "Monetary values, including unit",
        "short_description": "Monetary values",
        "examples": [
            "100 dollars",
            "100 euros",
            "100$",
        ],
    },
    "NORP": {
        "long_description": "Nationalities or religious or political groups",
        "short_description": "Nationalities, religions, political groups",
        "examples": [
            "the Catholic Church",
            "the Republican Party",
        ],
    },
    "ORDINAL": {
        "long_description": 'Ordinals element like "first", "second", etc.',
        "short_description": "Ordinals",
        "examples": [
            "first",
            "second",
        ],
    },
    "ORG": {
        "long_description": "Organizations: Companies, agencies, institutions, etc.",
        "short_description": "Organizations",
        "examples": [
            "Google",
            "Apple",
        ],
    },
    "PERCENT": {
        "long_description": 'Percentage, including "%"',
        "short_description": "Percentages",
        "examples": [
            "50%",
            "50 percents",
        ],
    },
    "PERSON": {
        "long_description": "People, including fictional",
        "short_description": "People",
        "examples": [
            "Barack Obama",
            "Hillary Clinton",
        ],
    },
    "PRODUCT": {
        "long_description": "Products: Objects, vehicles, foods, etc. (not services)",
        "short_description": "Products",
        "examples": [
            "iPhone",
            "Toyota",
        ],
    },
    "QUANTITY": {
        "long_description": "Measurements, as of weight or distance",
        "short_description": "Measurements",
        "examples": [
            "100 meters",
        ],
    },
    "TIME": {
        "long_description": "Times smaller than a day",
        "short_description": "Time in a day",
        "examples": [
            "3:30pm",
            "3:30 p.m.",
        ],
    },
    "WORK_OF_ART": {
        "long_description": "Titles of books, songs, etc.",
        "short_description": "Work of art",
        "examples": [
            "Harry Potter",
            "The Lord of the Rings",
        ],
    },
    "UNKNOWN": {
        "long_description": "Unknown entity",
        "short_description": "Unknown entity",
        "examples": [
            "Unknown entity",
        ],
    },
}

NER_SHORTCUTS = {
    "NUM": "CARDINAL",
    "CARD": "CARDINAL",
    "MIS": "MISC",
    "PER": "PERSON",
}


def explain_ner(ner_tag: str) -> dict:
    """
    Explain NER entity
    Args:
        text (str): NER entity
    Returns:
        dict: NER entity explanation
    """
    ner_tag = ner_tag.upper()

    if ner_tag in NER_SHORTCUTS:
        output = NER_LOOKUP.get(NER_SHORTCUTS[ner_tag], NER_LOOKUP["UNKNOWN"])
    else:
        output = NER_LOOKUP.get(ner_tag, NER_LOOKUP["UNKNOWN"])

    return output
