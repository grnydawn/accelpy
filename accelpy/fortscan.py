"""Fortran scanner
"""

import re

_pat_word = re.compile(r"^[\w]+")

# keyword: end keyword
_spec_keywords = {
"type": re.compile(r"^end\s*type"), "class": None,
"interface": re.compile(r"^end\s*interface"),
"integer": None, "real": None, "character": None, "logical": None, "double": None, "complex": None,
"public": None, "private": None, "external": None,
"parameter": None, "implicit": None, "format": None, "entry": None,
"entry": None, "enum": None, "format": None,
"access": None, "allocatable": None, "asynchronous": None, "bind": None, "common": None,
"data": None, "dimension": None, "equivalence": None, "external": None, "intent": None,
"intrinsic": None, "namelist": None, "optional": None, "pointer": None, "protected": None,
"save": None, "target": None, "volatile": None, "value": None,
"use": None, "implicit": None, "include": None, 
"procedure": None,
}

_exec_keywords = [
    "do", "if", "program", "module", "select", "call", "write", "print", "read",
    "contains", "associate", "case", "forall", "subroutine", "function",
    "exit", "flush", "forall", "goto", "inquire", "nullify", "open",
    "return", "rewind", "stop", "wait", "where", "allocate", "backspace", "close",
    "continue", "cycle", "deallocate", "endfile", "else", "endif", "enddo", "end do",
    "end if", "else if", "elseif",
]


_skip_keywords = [
]

class Line(object):

    BLANK, COMMENT, SKIP, SPEC, CPP, CONT, EXEC = range(7)

    def __init__(self, line, lines, index, srclineno):
        self.srclineno = srclineno
        self.line = line
        self.lines = lines
        self.index = index
        self.linetype = None
        self.nextline = None

    def in_specpart(self):

        if self.linetype is None or self.nextline is None:
            self._parse()

        return self.linetype == self.SPEC

    def in_execpart(self):

        if self.linetype is None or self.nextline is None:
            self._parse()

        return self.linetype == self.EXEC

    def next(self):

        if self.linetype is None or self.nextline is None:
            self._parse()

        if self.linetype is None or self.nextline is None:
            return self.index + 1

        else:
            return self.nextline

    def _firstword(self):
        match = _pat_word.match(self.line)
        return match.group() if match else self.line

    def _is_spec(self, word):
        return word in _spec_keywords

    def _is_exec(self, word):
        return word in _exec_keywords

    def _is_skip(self, word):
        return word in _skip_keywords

    def _spec_next(self, keyword):
        if _spec_keywords[keyword]:
            for idx in range(self.index+1, len(self.lines)):
                match = _spec_keywords[keyword].match(self.lines[idx].line)
                if match:
                    return idx + 1
        return self.index + 1

    def _skip_next(self, keyword):
        return self.index + 1

    def _cpp_next(self):

        line = self.line[1:].lstrip()

        if line.startswith("if") or line.startswith("elif"):
            for idx in range(self.index+1, len(self.lines)):
                if self.lines[idx].line.startswith("#endif"):
                    return idx + 1
            return self.index + 1
        else:
            return self.index + 1


    def _parse(self):

        if self.line[0] == "!":
            self.linetype, self.nextline = self.COMMENT, self.index+1
            return

        if self.line[0] == "#":
            self.nextline = self._cpp_next()
            self.linetype = self.EXEC
            return

        firstword = self._firstword()

        if self._is_spec(firstword):
            self.nextline = self._spec_next(firstword)
            self.linetype = self.SPEC
            return

        if self._is_exec(firstword):
            self.nextline = self.index + 1
            self.linetype = self.EXEC
            return

        if self._is_skip(firstword):
            self.nextline = self.index+1
            self.linetype = self.SKIP
            return

        if firstword == "end":
            self.nextline = self.index + 1
            self.linetype = self.EXEC
            return

        if len(self.line)> len(firstword):
            l2 = self.line[len(firstword):].lstrip()
            if l2 and l2[0] == "=":
                self.nextline = self.index + 1
                self.linetype = self.EXEC
                return

        # line is not consumed by the parser

        self.nextline = self.index + 1
        self.linetype = self.EXEC
        return


def get_firstexec(linelist):

    lines = []
    line = None

    for srclineno, rawline in enumerate(linelist):
        line1 = rawline.strip().lower()

        if not line1:
            continue

        if line1[-1] == "&":
            if line is None:
                line = line1
            else:
                line += line1[:-1]
            continue

        elif line is None:
            line = line1

        else:
            line += line1
            
        lines.append(Line(line, lines, len(lines), srclineno))
        line = None

    maxlines = len(lines)
    lineno = 0
    lastspec = 0

    while lineno < maxlines:

        line = lines[lineno]

        if line.in_specpart():
            lastspec = line.srclineno

        elif line.in_execpart():
            return lines[line.index-1].srclineno

        lineno = line.next()

    return lastspec
