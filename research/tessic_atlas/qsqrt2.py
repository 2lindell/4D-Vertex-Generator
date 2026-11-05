"""Exact numbers a + b sqrt2 (a, b rational)."""
from __future__ import annotations

from fractions import Fraction


class K:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a, self.b = Fraction(a), Fraction(b)

    @staticmethod
    def of(x):
        return x if isinstance(x, K) else K(x)

    def __add__(s, o):
        o = K.of(o); return K(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __sub__(s, o):
        o = K.of(o); return K(s.a - o.a, s.b - o.b)

    def __rsub__(s, o):
        return K.of(o) - s

    def __neg__(s):
        return K(-s.a, -s.b)

    def __mul__(s, o):
        o = K.of(o); return K(s.a * o.a + 2 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = K.of(o)
        n = o.a * o.a - 2 * o.b * o.b
        return s * K(o.a / n, -o.b / n)

    def sign(s):
        a, b = s.a, s.b
        if a >= 0 and b >= 0:
            return int(a > 0 or b > 0)
        if a <= 0 and b <= 0:
            return -1
        d = a * a - 2 * b * b                 # (a, b of opposite signs: the larger in size wins)
        return (1 if a > 0 else -1) * (1 if d > 0 else -1 if d < 0 else 0)

    def __eq__(s, o):
        o = K.of(o); return s.a == o.a and s.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __float__(s):
        return float(s.a) + float(s.b) * 2 ** 0.5

    def __repr__(s):
        if s.b == 0:
            return str(s.a)
        if s.a == 0:
            return f"{s.b}*sqrt2"
        return f"({s.a}{'+' if s.b > 0 else '-'}{abs(s.b)}*sqrt2)"


def from_sympy(v):
    import sympy as sp
    v = sp.nsimplify(sp.expand(v))
    a, b = sp.Rational(0), sp.Rational(0)
    for t in sp.Add.make_args(v):
        c, r = t.as_coeff_Mul()
        if r == 1:
            a += c
        elif r == sp.sqrt(2):
            b += c
        else:
            raise ValueError(f"not in Q(sqrt2): {v}")
    return K(Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q)))
