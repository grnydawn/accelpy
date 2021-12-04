"""Fortran scanner
"""

import re

_pat_word = re.compile(r"^[\w]+")

# keyword: end keyword
_spec_keywords = {
"type": re.compile(r"^end\s*type"),
"integer": None,
}

_exec_keywords = [
]

class Line(object):

    BLANK, COMMENT, SPEC, CPP, CONT, EXEC = range(6)

    def __init__(self, lineno, rawline, lines):
        self.lineno = lineno
        self.line = rawline.strip().lower()
        self.lines = lines
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
            return self.lineno + 1

        else:
            return self.nextline

    def _firstword(self):
        match = _pat_word.match(self.line)
        return match.group() if match else self.line

    def _nextline(self):
        idx = self.lineno
        while idx < len(self.lines):
            if self.lines[idx].line:
                import pdb; pdb.set_trace()
            else:
                import pdb; pdb.set_trace()
                
    def _comment(self):
        nextline = self._nextline()
        for index in range(self.lineno, nextline):
            self.lines[index].linetype = self.COMMENT
            self.lines[index].nextline = nextline
        return self.COMMENT, nextline

    def _is_spec(self, word):
        return word in _spec_keywords

    def _is_exec(self, word):
        return word in _exec_keywords

    def _parse_spec(self, keyword):
        linetype = self.SPEC
        endline = self.lines[self.lineno+1]._parse()
        if endline:
            import pdb; pdb.set_trace()
        else:
            nextline = self.lineno + 1

        return linetype, nextline

    def _parse(self):

        if not self.line:
            self.linetype, self.nextline = self.BLANK, self.lineno+1
            return

        if self.line[0] == "!":
            self.linetype, self.nextline = self.COMMENT, self.lineno+1
            return

        if self.line[0] == "#":
            self.linetype, self.nextline = self._parse_preproc()
            return

        firstword = self._firstword()

        if self._is_spec(firstword):
            self.linetype, self.nextline = self._parse_spec(firstword)
            return

        if self._is_exec(firstword):
            # previous steps ensure that this is not part of spec or preproc
            self.linetype, self.nextline = self.EXEC, len(self.lines)
            return

        # line is not consumed by the parser
        return self


def get_firstexec(linelist):

    lines = []

    for lineno, rawline in enumerate(linelist):
        lines.append(Line(lineno, rawline, lines))

    maxlines = len(lines)
    lineno = 0
    lastspec = 0
    firstexec = 0

    while lineno < maxlines:

        line = lines[lineno]

        if line.in_specpart():
            lastspec = lineno

        elif line.in_execpart():
            firstexec = lineno - 1
            break

        lineno = line.next()

    if firstexec <= lastspec+1:
        return lastspec + 1

    import pdb; pdb.set_trace()
    # handle the gap
