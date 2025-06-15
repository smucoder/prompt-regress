

s = {"a": 1, "b": 2, "c": 3}


p = "hi {a} {b} {c}"


print(p.format(**s))