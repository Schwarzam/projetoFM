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
        "columns": [
                        "splus_cut_F378",
                        "splus_cut_F395",
                        "splus_cut_F410",
                        "splus_cut_F430",
                        "splus_cut_F515",
                        "splus_cut_F660",
                        "splus_cut_F861",
                        "splus_cut_R",
                        "splus_cut_I",
                        "splus_cut_Z",
                        "splus_cut_U",
                        "splus_cut_G"
                    ]
    }
    
    def __init__(self):
        pass

    
    def get_metadata(self):
        return self.__metadata__