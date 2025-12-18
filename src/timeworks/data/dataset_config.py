tfb_root_path = '/home/yuanlinfeng/data/forecasting/'
root_path = '/home/yuanlinfeng/data/'

tfb_datasets = ['Solar', 'PEMS04', 'PEMS08', 'Wind', 'AQShunyi', 'AQWan']
traditional_datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'electricity', 'traffic']


from dataclasses import dataclass

# 数据集配置结构体定义
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    split: str

config_dict = {
    # Conventional  
    "ETTh1": DatasetInfo(split='712'), 
    "ETTh2":  DatasetInfo(split='712'),
    "ETTm1":  DatasetInfo(split='712'),
    "ETTm2":  DatasetInfo(split='712'),
    "electricity":  DatasetInfo(split='712'),
    "traffic":  DatasetInfo(split='712'),
    # TFB
    "Wind":  DatasetInfo(split='712'),
    "Solar":  DatasetInfo(split='622'),
    "PEMS04":  DatasetInfo(split='622'),
    "PEMS08":  DatasetInfo(split='622'),
    "Solar":  DatasetInfo(split='622'),
    "AQShunyi":  DatasetInfo(split='622'),
    "AQWan":  DatasetInfo(split='622'), 
}
