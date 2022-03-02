import re
from pprint import pprint

def abbrevs(text):
    """returns a list of pairs (abbrev, long form) from text

    using the algorithm of Schwartz and Hearst https://pubmed.ncbi.nlm.nih.gov/12603049/

    @args
      text: a text string, in which 'long form (short form)'
         abbreviations might be hiding

    @returns
      A list of pairs of the form (abbrev, long form)
      from the text
"""

    def possible_abbreviations(text):
        """A helper function. 

        returns a list of pairs with candidate abbreviations
        as the first value and strings where long forms might
        exist as the second"""
        # a regular expression to match parenthesised text as a possible abbreviation
        abbrevs_re = re.compile("\(([A-Za-z0-9]+)\)")    
        textList = abbrevs_re.split(text)
        _longforms = textList[0::2] # the even entries
        _abbrevs = textList[1::2] # the odd entries
        return list(zip(_abbrevs,_longforms))

    def findBestLongForm(sf,lf):
        """A helper function 

        Finds the long form of an abbreviation from the candidates
        and returns it. Returns nothing if not found"""
        A = len(sf)
        lfwords = lf.split()
        maxwords = min(A+5,A*2,len(lfwords))
        lfcandidates = [lfwords[(len(lfwords)-n):len(lfwords)] for n in range(1,maxwords+1)]
        def filterfun(c):
            return bool(c[0][0].lower() == sf[0].lower())
        lfcandidates = list(filter(filterfun,lfcandidates))
        result = None
        for lf in lfcandidates:
            lfx = " ".join(lf).lower()
            sfx = list(sf.lower())[1:]
            working = True
            while ((sfx) and working):
                char = sfx.pop()
                pos = lfx.rfind(char)
                if (pos > -1):
                    lfx = lfx[:pos]
                else:
                    working = False
            if working:
                return (sf," ".join(lf))

    
    candidates = possible_abbreviations(text)
    return dict([ c for c in [findBestLongForm(*sflf) for sflf in candidates] if c])
