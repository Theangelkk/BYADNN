import numpy as np
from datetime import datetime, timedelta
import pendulum
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
from utils import remove_imp_training_set

def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)

def compute_week_of_month(dt):
    dt_pendulum = pendulum.parse("{}-{}-{}".format(dt.year, dt.month, dt.day), strict=False)
    return dt_pendulum.week_of_month

class CustomDataSet_Train(Dataset):
    def __init__(self, cod_station, key, info_cod_stations, batch_size, 
                all_x_train, all_y_train, all_idx_missing_values_train,
                model_order, max_model_order, 
                dates, remove_imp_target_training=False,
                org_data=False,
                standardization_data=False):
        
        self.cod_station = cod_station
        self.model_order = model_order
        self.batch_size = batch_size
        self.dates = dates
        self.max_model_order = max_model_order
        self.org_data = org_data
        self.standardization_data = standardization_data
        self.region = info_cod_stations[key][2]
        self.start_year = info_cod_stations[key][3]
        self.end_year = info_cod_stations[key][4]

        self.mean = 0.0
        self.std = 1.0

        self.list_years = [*range(2014,2025)]

        self.idx_missing_values = all_idx_missing_values_train[key]

        self.x_train = all_x_train[key]
        self.y_train = all_y_train[key]
        
        if remove_imp_target_training:
            self.x_train_to_use, self.y_train_to_use = remove_imp_training_set(self.x_train, self.y_train, self.idx_missing_values)
        else:
            self.x_train_to_use = self.x_train
            self.y_train_to_use = self.y_train

        self.min_cod_station = info_cod_stations[key][5]
        self.max_cod_station = info_cod_stations[key][6]

        if self.org_data:
            self.x_train_to_use = self.x_train_to_use * (self.max_cod_station - self.min_cod_station) + self.min_cod_station
            self.y_train_to_use = self.y_train_to_use * (self.max_cod_station - self.min_cod_station) + self.min_cod_station
            
            self.x_train = self.x_train * (self.max_cod_station - self.min_cod_station) + self.min_cod_station
            self.y_train = self.y_train * (self.max_cod_station - self.min_cod_station) + self.min_cod_station
            
        # Create torch Tensor
        if self.standardization_data:
            self.mean = np.mean(self.y_train_to_use)
            self.std = np.std(self.y_train_to_use)

            if self.std > 0.0:
                self.x_train_to_use = (self.x_train_to_use - self.mean) / self.std
                self.y_train_to_use = (self.y_train_to_use - self.mean) / self.std
                self.x_train = (self.x_train - self.mean) / self.std
                self.y_train = (self.y_train - self.mean) / self.std
            else:
                self.x_train_to_use = self.x_train_to_use - self.mean
                self.y_train_to_use = self.y_train_to_use - self.mean
                self.x_train = self.x_train - self.mean
                self.y_train = self.y_train - self.mean

        self.x_train_tensor = torch.from_numpy(self.x_train_to_use).float()
        self.y_train_tensor = torch.from_numpy(self.y_train_to_use).float()

        self.n_training_samples = self.x_train.shape[0]

        n_batches_current_dataset = int(self.x_train_to_use.shape[0] / self.batch_size)

        if n_batches_current_dataset == 0:
            print("ERROR: The dimension of this dataset is less than the batch size specified")
            exit(-1)

        self.n_training_samples = n_batches_current_dataset * batch_size

        self.gen_mask()
        self.gen_list_info_date()
        
    def __len__(self):
        return self.n_training_samples

    def get_model_order(self):
        return self.model_order

    def gen_mask(self):
        self.mask = np.ones((self.max_model_order, self.max_model_order))
    
        for r in range(self.max_model_order):
            if r >= self.model_order - 1:
                start_c = (self.model_order - 1)
                end_c = self.max_model_order
                for c in range(start_c, end_c):
                    self.mask[r,c] = 1.0
    
    def gen_list_info_date(self):

        self.hours = np.zeros((len(self.dates), self.max_model_order))
        self.days = np.zeros((len(self.dates), self.max_model_order))
        self.weeks_of_month = np.zeros((len(self.dates), self.max_model_order))
        self.months = np.zeros((len(self.dates), self.max_model_order))
        self.years = np.zeros((len(self.dates), self.max_model_order))

        for i in range(len(self.dates)):
            
            target_date = datetime.utcfromtimestamp(self.dates[i])

            for j in range(1, self.max_model_order+1):
                
                current_date = target_date - timedelta(hours=j)

                self.hours[i, -j] = current_date.hour
                self.days[i, -j] = current_date.day - 1
                self.weeks_of_month[i, -j] = compute_week_of_month(current_date) - 1
                self.months[i, -j] = current_date.month - 1
                self.years[i, -j] = self.list_years.index(current_date.year)
        
    def get_list_info_date(self):
        
        return [    torch.from_numpy(self.hours), 
                    torch.from_numpy(self.days), 
                    torch.from_numpy(self.weeks_of_month), 
                    torch.from_numpy(self.months), 
                    torch.from_numpy(self.years)
        ]
    
    def __getitem__(self, idx):
        
        x = self.x_train_tensor[idx]
        y = self.y_train_tensor[idx]

        list_info_date = [  torch.from_numpy(self.hours[idx, :]), 
                            torch.from_numpy(self.days[idx, :]), 
                            torch.from_numpy(self.weeks_of_month[idx, :]), 
                            torch.from_numpy(self.months[idx, :]), 
                            torch.from_numpy(self.years[idx, :]), 
        ]

        return {'cod station': self.cod_station, 'model order': self.model_order, 'x': x, 'y': y, 'date': list_info_date, 'mask': self.mask}

class CustomDataSet_Test(Dataset):
    def __init__(self, cod_station, key, info_cod_stations, x, y, model_order, max_model_order, dates, org_data=False,
                standardization_data=False, mean=0.0, std=0.0):
        
        self.cod_station = cod_station
        self.model_order = model_order
        self.max_model_order = max_model_order
        self.dates = dates
        self.org_data = org_data
        self.standardization_data = standardization_data
        self.region = info_cod_stations[key][2]
        self.start_year = info_cod_stations[key][3]
        self.end_year = info_cod_stations[key][4]
        self.list_years = [*range(2014,2025)]

        self.mean = mean
        self.std = std

        self.x = x
        self.y = y
        
        self.min_cod_station = info_cod_stations[key][5]
        self.max_cod_station = info_cod_stations[key][6]

        if self.org_data:
            self.x = self.x * (self.max_cod_station - self.min_cod_station) + self.min_cod_station
            self.y = self.y * (self.max_cod_station - self.min_cod_station) + self.min_cod_station

        if self.standardization_data:
            if self.std > 0.0:
                self.x = (self.x - self.mean) / self.std
                self.y = (self.y - self.mean) / self.std
            else:
                self.x =  self.x - self.mean
                self.y =  self.y - self.mean

        # Create torch Tensor
        self.x_tensor = torch.from_numpy(self.x).float()
        self.y_tensor = torch.from_numpy(self.y).float()

        self.n_samples = self.x.shape[0]
        self.batch_size = self.n_samples

        self.gen_mask()
        self.gen_list_info_date()
        
    def __len__(self):
        return self.n_samples

    def get_model_order(self):
        return self.model_order

    def gen_mask(self):
        self.mask = np.ones((self.max_model_order, self.max_model_order))
    
        for r in range(self.max_model_order):
            if r >= self.model_order - 1:
                start_c = (self.model_order - 1)
                end_c = self.max_model_order
                for c in range(start_c, end_c):
                    self.mask[r,c] = 1.0
    
    def gen_list_info_date(self):

        self.hours = np.zeros((len(self.dates), self.max_model_order))
        self.days = np.zeros((len(self.dates), self.max_model_order))
        self.weeks_of_month = np.zeros((len(self.dates), self.max_model_order))
        self.months = np.zeros((len(self.dates), self.max_model_order))
        self.years = np.zeros((len(self.dates), self.max_model_order))

        for i in range(len(self.dates)):
            
            target_date = datetime.utcfromtimestamp(self.dates[i])

            for j in range(1, self.max_model_order+1):
                
                current_date = target_date - timedelta(hours=j)

                self.hours[i, -j] = current_date.hour
                self.days[i, -j] = current_date.day - 1
                self.weeks_of_month[i, -j] = compute_week_of_month(current_date) - 1
                self.months[i, -j] = current_date.month - 1
                self.years[i, -j] = self.list_years.index(current_date.year)
        
    def get_list_info_date(self):
        
        return [    torch.from_numpy(self.hours), 
                    torch.from_numpy(self.days), 
                    torch.from_numpy(self.weeks_of_month), 
                    torch.from_numpy(self.months), 
                    torch.from_numpy(self.years)
        ]
    
    def __getitem__(self, idx):
        
        x = self.x_tensor[idx]
        y = self.y_tensor[idx]

        list_info_date = [  torch.from_numpy(self.hours[idx, :]), 
                            torch.from_numpy(self.days[idx, :]), 
                            torch.from_numpy(self.weeks_of_month[idx, :]), 
                            torch.from_numpy(self.months[idx, :]), 
                            torch.from_numpy(self.years[idx, :]), 
        ]

        return {'cod station': self.cod_station, 'model order': self.model_order, 'x': x, 'y': y, 'date': list_info_date, 'mask': self.mask}
    
class MyBatchSampler(Sampler):
  def __init__(self, list_indices, batch_size): 
    self.list_indices = list_indices
    self.len_indices = 0
    self.batch_size = batch_size
    self.all_batches = []

    for i in range(len(self.list_indices)):
      self.len_indices += len(self.list_indices[i])

  def __len__(self):
    return self.len_indices // self.batch_size
  
  def __iter__(self):
    tmp_list_batches = []
    
    for i in range(len(self.list_indices)):
      random.shuffle(self.list_indices[i])
      current_batches = chunk(self.list_indices[i], self.batch_size)
      tmp_list_batches += current_batches

    self.all_batches = [batch.tolist() for batch in tmp_list_batches]

    return iter(self.all_batches)