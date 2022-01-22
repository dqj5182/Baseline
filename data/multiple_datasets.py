import numpy as np
from torch.utils.data.dataset import Dataset

from core.logger import logger

class MultipleDatasets(Dataset):
    def __init__(self, dbs, partition, make_same_len=False):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])

        if make_same_len: 
            self.partition = [1.0/self.db_num for i in range(self.db_num)]
        else: 
            if len(partition) != self.db_num: 
                logger.info("Invalid parition!")
                assert 0
            self.partition = partition
        self.partition = np.array(self.partition).cumsum()

    def __len__(self):
        return self.max_db_data_num

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(self.db_num):
            if p <= self.partition[i]:
                if index < len(self.dbs[i]) * (self.max_db_data_num // len(self.dbs[i])): 
                    # before last batch: use modular
                    index = index % len(self.dbs[i])
                else:
                    # last batch: uniform sampling
                    index = int(np.random.rand() * len(self.dbs[i]))
                    
                return self.dbs[i][index]
