import re
from typing import TextIO, List, Iterable, Tuple


def rules_to_documents(rules_input: TextIO) -> List[str]:
    """
    -> Discard until line with "Credits"
    -> Prepend all headers (to add context) to every rule line and consider this as a document
       Ex: 100.1. Game Concepts: General: These Magic rules apply to any Magic game with two or more players, including two-player games and multiplayer games.
    -> At the "Glossary" line stop and create a document per glossary item. first line is the name and subsequent lines (until empty line) is the desc
       Ex. Absorb: A keyword ability that prevents damage. See rule 702.64, “Absorb.”
    - > Stop at credits
    Stop at Glossary
    :return:
    """
    h1_re = re.compile(r"^\d\.\s([\w ]+)")
    h2_re = re.compile(r"^\d\d+\.\s([\w ]+)")
    documents = []

    def next_line_iterator() -> Iterable[Tuple[str, bool]]:
        previous_empty = True
        for line in rules_input:
            clean_line = line.strip()
            if clean_line != "":
                yield clean_line, previous_empty
                previous_empty = False
            else:
                previous_empty = True

    nl_iter = iter(next_line_iterator())

    current_line, _ = next(nl_iter)
    while current_line != "Credits":
        current_line, _ = next(nl_iter)

    context_stack = []
    while current_line != "Glossary":
        current_line, previous_empty = next(nl_iter)

        if not previous_empty:
            documents[-1] += f" {current_line}"

        h1_m = h1_re.search(current_line)
        if h1_m:
            if len(context_stack) > 0:
                context_stack.clear()
            context_stack.append(h1_m.group(1))
            continue

        h2_m = h2_re.search(current_line)
        if h2_m:
            if len(context_stack) == 2:
                context_stack.pop(-1)
            context_stack.append(h2_m.group(1))
            continue

        documents.append(
            f"{': '.join(context_stack)}: {current_line}"
        )

    current_glossary_def = []
    current_line, previous_empty = next(nl_iter)
    while current_line != "Credits":
        if previous_empty:
            if len(current_glossary_def) > 0:
                documents.append(' '.join(current_glossary_def))
            current_glossary_def.clear()
            current_glossary_def.append(f"{current_line}:")
        else:
            current_glossary_def.append(current_line)
        current_line, previous_empty = next(nl_iter)

    if len(current_glossary_def) > 0:
        documents.append(' '.join(current_glossary_def))

    return documents
