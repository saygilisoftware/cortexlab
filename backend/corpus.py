"""Original tiny educational corpus, released with CortexLab under MIT.

These are authored grammar fixtures, not observations or pretrained general knowledge.
"""
COLORS = ["cyan", "blue", "violet", "green", "amber", "white"]
OBJECTS = ["star", "planet", "signal", "crystal", "neuron", "comet"]
VERBS = ["glows", "moves", "rotates", "pulses"]


def sentences():
    texts = []
    for color in COLORS:
        for thing in OBJECTS:
            for verb in VERBS:
                texts.extend([
                    f"the {color} {thing} {verb} .",
                    f"a {thing} is {color} . the {thing} {verb} .",
                    f"observe the {thing} . its color is {color} .",
                ])
    texts.extend([
        "attention mixes value vectors .",
        "a transformer predicts the next token .",
        "each layer transforms the token state .",
        "the model observes a sequence of tokens .",
        "the signal moves through the model .",
    ] * 8)
    return texts
