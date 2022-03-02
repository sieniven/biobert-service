import pandas as pd
import numpy as np
import logging
import os
import yaml
import configparser
import requests
import dateutil.parser


class Gene2PubMed:
    g2p_url = 'https://ftp.ncbi.nih.gov/gene/DATA/gene2pubmed.gz'
    
    def __init__(self,data_dir):
        self.logger = logging.getLogger(__name__)
        try:
            os.path.exists(data_dir)
        except OSError as e:
            self.logger.error("OS Error for data_dir {}".format(data_dir))
        except ValueError as e:
            self.logger.error("Error for data_dir {}".format(data_dir))
        self.g2p_fn = os.path.abspath(os.path.join(data_dir,"g2p.parquet"))
        self.timestamp_fn = os.path.abspath(os.path.join(data_dir,"g2p.timestamp"))
        if self.has_current():
            self.local_timestamp = open(self.timestamp_fn,"r").read()
            self.g2p = pd.read_parquet(self.g2p_fn)
        else:
            self.local_timestamp = ""
            self.remote_timestamp = ""
            self.get_latest()
            self.serialise()

        
    def has_current(self):
        return (os.path.exists(self.g2p_fn) and (os.path.exists(self.timestamp_fn)))


    def is_current(self):
        if ((self.has_current()) and 
            (self.__class__.last_modified() == self.local_timestamp)):
            return True
        else:
            return False


    def get_latest(self):
        self.remote_timestamp = self.__class__.last_modified()
        iter_tsv = pd.read_csv(self.__class__.g2p_url,sep='\t',header=0,names=['taxon','GeneID','PubMedID'],compression='gzip',iterator=True, chunksize=100000)
        self.g2p_latest  = pd.concat([chunk[chunk['taxon'] == 9606] for chunk in iter_tsv])[['GeneID','PubMedID']]

    def serialise(self):
        if (self.remote_timestamp != self.local_timestamp):
            self.g2p = self.g2p_latest
            self.g2p_latest = None
            self.local_timestamp = self.remote_timestamp
            self.g2p.to_parquet(self.g2p_fn)
            f = open(self.timestamp_fn,"w")
            f.write(self.local_timestamp)
            f.close()

    def new_entries(self):
        """using https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe"""
        if (not self.is_current()):
            self.get_latest()
            df_all = self.g2p_latest.merge(self.g2p,how='left', indicator=True)
            return df_all[df_all['_merge'] == 'left_only'][['GeneID','PubMedID']]


    def new_PMIDs(self):
        """"""
        new_entries = self.new_entries()
        try:
            if (len(new_entries) > 0 ):
                return np.unique(new_entries.PubMedID)
            else:
                return []
        except:
            pass

        
    @classmethod
    def last_modified(cls):
        response = requests.head(cls.g2p_url)
        last_modified = response.headers.get('Last-Modified')
        return last_modified

