import requests
import logging
import json
import objectpath
from indra.databases import uniprot_client
import gilda
from GTT.config import get_config
from .abbrevs import abbrevs

baseurl = get_config("reach_url")
if (not baseurl):
    baseurl = 'http://agathon.sista.arizona.edu:8080/odinweb/api/text'


class Reach():

    def __init__(self,querytext=None):
        self.baseurl = baseurl
        self.data = None
        self.logger = logging.getLogger(__name__)
        if (querytext):
            self.query(querytext)

    def query(self,text):
        resp = requests.post(self.baseurl,
                             params={"text": text,
                                     "output": "fries"})
        self.text = text
        self.data = objectpath.Tree(json.loads(resp.content))
        self.abbrevs = abbrevs(text)
        self.get_proteins()
        self.get_families()

    def get_proteins(self):
        result = self.data.execute("$.entities.frames[@.type in ['gene', 'protein']]")
        uids = {}
        for entry in result:
            xrefs = entry['xrefs'][0]
            namespace = xrefs['namespace']
            key = xrefs['id']
            text = entry['text']
            start_pos = entry['start-pos']['offset']
            try:
                species = xrefs['species']
            except KeyError:
                species = ""
            if ((namespace == 'uaz') or
                ((namespace == "uniprot") and ((species == 'homo sapiens')
                                               or (species == 'human')))):
                if (namespace == "uniprot"):
                    hgnc_id = uniprot_client.get_hgnc_id(key)
                    entrez_id = uniprot_client.get_entrez_id(key)
                    symbol = uniprot_client.get_gene_name(key)
                else:
                    scored_matches = gilda.ground(text)
                    if scored_matches:
                        self.logger.debug(scored_matches)
                    if (scored_matches and (scored_matches[0].term.db == 'HGNC')):
                        _data = scored_matches[0].to_json()
                        hgnc_id = _data['term']['id']
                        symbol = _data['term']['entry_name']
                        namespace = 'HGNC'
                    else:
                        hgnc_id = ""
                        symbol = ""
                    entrez_id = ""
                if key in uids:
                    uids[key]['count'] = uids[key]['count'] + 1
                    uids[key]['start_pos'] = min(start_pos,uids[key]['start_pos'])
                else:
                    uids[key] = {"count": 1,
                                 "text": entry["text"],
                                 "type": entry["type"],
                                 "start_pos": start_pos,
                                 "namespace": namespace,
                                 "hgnc_id": hgnc_id,
                                 "entrez_id": entrez_id,
                                 "symbol": symbol
                                 }

        uids = {k:v for k,v in uids.items() if not(self.is_misassigned(v['symbol'],v['type']))}
        self.proteins = uids
        return uids

    def get_families(self):
        result = self.data.execute("$.entities.frames[@.type is 'family']")
        uids = {}
        for entry in result:
            xrefs = entry['xrefs'][0]
            namespace = xrefs['namespace']
            key = xrefs['id']
            start_pos = entry['start-pos']['offset']
            if key in uids:
                uids[key]['count'] = uids[key]['count'] + 1
                uids[key]['start_pos'] = min(start_pos,uids[key]['start_pos'])
            else:
                uids[key] = {"count": 1,
                             "start_pos": entry['start-pos']['offset'],
                             "namespace": namespace,
                             "symbol" : "",
                             "text": entry["text"],
                             }
        self.families = uids
        return uids

    def is_misassigned(self,symbol,namespace_target):
        lf = self.abbrevs.get(symbol)
        namespaces = list(self.data.execute(f"$.entities.frames[@.text is '{lf}'].type"))
        if (len(namespaces) > 0):
            if (namespaces[0] != namespace_target):
                return True
            return False
