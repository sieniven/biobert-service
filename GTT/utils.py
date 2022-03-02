from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    """Utility class and function to strip tags from HTML in Pubmed titles and abstracts"""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    n = max(n,1)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
