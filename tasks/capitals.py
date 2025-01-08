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


def capitals_generator():
    while True:
        E_0, E_1, E_0p, E_1p = np.random.choice(NAMES, 4, replace=False)
        A_0, A_1, A_0p, A_1p = np.random.choice(
            list(CAPITAL_MAP.keys()), 4, replace=False
        )
        yield CapitalsExample(E_0, E_1, A_0, A_1, E_0p, E_1p, A_0p, A_1p)


class CapitalsExample:
    def __init__(
        self,
        E_0: str,
        E_1: str,
        A_0: str,
        A_1: str,
        E_0p: str,
        E_1p: str,
        A_0p: str,
        A_1p: str,
    ):
        self.E_0 = E_0
        self.E_1 = E_1
        self.E_0p = E_0p
        self.E_1p = E_1p
        self.A_0 = A_0
        self.A_1 = A_1
        self.A_0p = A_0p
        self.A_1p = A_1p
        self.answer_0 = CAPITAL_MAP[A_0]
        self.answer_1 = CAPITAL_MAP[A_1]
        self.answer_0p = CAPITAL_MAP[A_0p]
        self.answer_1p = CAPITAL_MAP[A_1p]

    @staticmethod
    def _context(E_0: str, E_1: str, A_0: str, A_1: str) -> str:
        context = f"""\
Answer the question based on the context below. Keep the answer short.
Context: {E_0} lives in the capital city of {A_0}. {E_1} lives in the capital city of {A_1}."""
        return context

    @property
    def context(self) -> str:
        return self._context(self.E_0, self.E_1, self.A_0, self.A_1)

    @property
    def context_p(self) -> str:
        return self._context(self.E_0p, self.E_1p, self.A_0p, self.A_1p)

    @staticmethod
    def _query(qn_subject) -> str:
        # note leading \n
        return f"""
Question: Which city does {qn_subject} live in?
Answer: {qn_subject} lives in the city of"""

    @property
    def query_E_0(self) -> str:
        return self._query(self.E_0)

    @property
    def query_E_1(self) -> str:
        return self._query(self.E_1)

    @property
    def query_E_0p(self) -> str:
        return self._query(self.E_0p)

    @property
    def query_E_1p(self) -> str:
        return self._query(self.E_1p)
