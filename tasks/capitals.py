import numpy as np

np.random.seed(42)

CAPITAL_MAP = {
    "China": "Beijing",
    "Russia": "Moscow",
    "Japan": "Tokyo",
    "Philippines": "Manila",
    "Egypt": "Cairo",
    "Iran": "Tehran",
    "Germany": "Berlin",
    "Thailand": "Bangkok",
    "England": "London",
    "France": "Paris",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Iraq": "Baghdad",
    "Poland": "Warsaw",
    "Canada": "Ottawa",
    "Chile": "Santiago",
    "Netherlands": "Amsterdam",
    "Syria": "Damascus",
    "Belgium": "Brussels",
    "Greece": "Athens",
    "Portugal": "Lisbon",
    "Sweden": "Stockholm",
    "Hungary": "Budapest",
    "Austria": "Vienna",
    "Israel": "Jerusalem",
    "Switzerland": "Bern",
}

NAMES = [
    "James",
    "Mary",
    "John",
    "Jennifer",
    "William",
    "Elizabeth",
    "Michael",
    "Sarah",
    "David",
    "Emily",
    "Robert",
    "Emma",
    "Joseph",
    "Susan",
    "Christopher",
    "Jessica",
    "Daniel",
    "Catherine",
    "Thomas",
    "Patricia",
    "Matthew",
    "Rachel",
    "Andrew",
    "Linda",
    "Richard",
    "Barbara",
    "Charles",
    "Michelle",
    "Anthony",
    "Lisa",
    "Steven",
    "Sandra",
    "Kevin",
    "Helen",
    "Brian",
    "Ashley",
    "George",
    "Anna",
    "Edward",
    "Olivia",
    "Donald",
    "Dorothy",
    "Paul",
    "Victoria",
    "Mark",
    "Rebecca",
    "Kenneth",
    "Karen",
    "Stephen",
    "Margaret",
]


def capitals_generator(n=2):
    assert n * 2 <= len(NAMES)
    assert n * 2 <= len(CAPITAL_MAP)
    while True:
        entities = np.random.choice(NAMES, n * 2, replace=False)
        attributes = np.random.choice(list(CAPITAL_MAP.keys()), n * 2, replace=False)
        yield CapitalsExample(n, entities, attributes)


class CapitalsExample:
    def __init__(
        self,
        n: int,
        entities: list[str],
        attributes: list[str],
    ):
        self.n = n
        self.entities = entities[:n]
        self.entities_p = entities[n:]
        self.attributes = attributes[:n]
        self.attributes_p = attributes[n:]
        self.answers = [CAPITAL_MAP[a] for a in self.attributes]
        self.answers_p = [CAPITAL_MAP[a] for a in self.attributes_p]

        for i, entity in enumerate(self.entities):
            setattr(self, f"E_{i}", entity)
            setattr(self, f"query_E_{i}", self._query(entity))

        for i, entity_p in enumerate(self.entities_p):
            setattr(self, f"E_{i}p", entity_p)
            setattr(self, f"query_E_{i}p", self._query(entity_p))

        for i, (attribute, answer) in enumerate(zip(self.attributes, self.answers)):
            setattr(self, f"A_{i}", attribute)
            setattr(self, f"answer_{i}", answer)

        for i, (attribute_p, answer_p) in enumerate(
            zip(self.attributes_p, self.answers_p)
        ):
            setattr(self, f"A_{i}p", attribute_p)
            setattr(self, f"answer_{i}p", answer_p)

    def _context(self, is_p) -> str:
        context = """\
Answer the question based on the context below. Keep the answer short.
Context:"""
        es_ = self.entities_p if is_p else self.entities
        as_ = self.attributes_p if is_p else self.attributes
        # iterate every 2:
        for i in range(self.n):
            E = es_[i]
            A = as_[i]
            context += f""" {E} lives in the capital city of {A}."""
        return context

    @property
    def context(self) -> str:
        return self._context(is_p=False)

    @property
    def context_p(self) -> str:
        return self._context(is_p=True)

    @staticmethod
    def _query(qn_subject) -> str:
        # note leading \n
        return f"""
Question: Which city does {qn_subject} live in?
Answer: {qn_subject} lives in the city of"""
