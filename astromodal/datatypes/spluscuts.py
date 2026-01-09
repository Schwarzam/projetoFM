from astromodal.datatypes.base import BaseDataType
from astromodal.config import load_config

import polars as pl
from astromodal.datasets.datacubes import load_datacube_files
from astromodal.datasets.spluscuts import SplusCutoutsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
        
class SplusCuts(BaseDataType):
    
    __metadata__ = {
        "id": "splus_cuts",
        "description": "S-PLUS photometric cuts data type",
        "bands": ["F378", "F395", "F410", "F430", "F515", "F660", "F861", "R", "I", "Z", "U", "G"],
        "cutout_size": 96,
    }
    
    def __init__(self):
        pass
        
    
    def get_metadata(self):
        return self.__metadata__