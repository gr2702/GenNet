import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np
import pandas as pd
import tables
import tensorflow.keras as K

matplotlib.use('agg')


def check_data(datapath, genotype_path, mode):
    # TODO write checks for multiple genotype files.
    # global groundtruth # why is this a global? # removed did it break something?
    """
    Checks the dataset and labels for appropriate formatting and datatypes
    
    Function Inputs:
    - datapath (String): Datapath where topology.csv, subjects.csv and other important numpy files stored
    - genotype_path (String): Path where the genotype matrices files are stored in the genotype.h5 format
    - mode (String): either "classification" or "regression", allows function to check for mismatch input data/labels

    Outputs:
    None
    """
    groundtruth = None
    genotype_matrix = False
    network_structure = False
    patient_info = False
    multiple_genotype_matrices = False
    number_of_covariats = False
    classification_problem = "undetermined"

    # Checks for the existence of the genotype matrices and sets corresponding booleans to true, and prints missing statement otherwise. 
    if os.path.exists(genotype_path + 'genotype.h5'):
        genotype_matrix = True
    elif len(glob.glob(genotype_path + '*.h5')) > 0:
        multiple_genotype_matrices = True
    else:
        print("genotype missing in", genotype_path)

    # Checks for existence of network structure via .csv or .npz files, set boolean to true, and prints missing statement otherwise.  
    if os.path.exists(datapath + 'topology.csv'):
        network_structure = True
    elif len(glob.glob(datapath + '*.npz')) > 0:
        network_structure = True
    else:
        print("topology.csv and *.npz are missing")

    # Checks for existence of subject.csv file containing subject ID phenotypes, and reads file into groundtruth Pandas Dataframe
    if os.path.exists(datapath + 'subjects.csv'):
        patient_info = True
        groundtruth = pd.read_csv(datapath + "/subjects.csv")
        
        number_of_covariats = groundtruth.filter(like="cov_").shape[1]
        print('number of covariates:', number_of_covariats)
        print('Covariate columns found:', list(groundtruth.filter(like="cov_").columns.values))
        
        if {'patient_id', 'labels', 'genotype_row', 'set'}.issubset(groundtruth.columns):
            classification_problem = ((groundtruth["labels"].values == 0) | (groundtruth["labels"].values == 1)).all() # Sets Classification_problem boolean to True if labels only contain 0s and 1s, False otherwise
        else:
            print("column names missing need 'patient_id', 'labels', 'genotype_row', 'set', got:",
                  groundtruth.columns.values)
            exit()
    else:
        print("subjects.csv is missing")

    # Print the mode inputted into function
    print("mode is", mode)

    # Ensures Groundtruth Dataframe label values matches the corresponding modes ("classification" vs "regression"), throws error otherwise
    if (mode == "classification") and classification_problem:
        pass
    elif (mode == "regression") and not classification_problem:
        pass
    else:
        print("The labels and the given mode do not correspond. \n"
              "Classification problems should have binary labels [1, 0]. \n"
              "Regression problems should not be binary. \n"
              "The labels do have the following values", groundtruth["labels"].unique())
        exit()

    # Checks that genotype file(s), network structure, and patient info are all present, otherwise prints error message if any files missing
    if multiple_genotype_matrices & network_structure & patient_info:
        TrainDataGenerator.multi_h5 = True
        return
    if genotype_matrix & network_structure & patient_info:
        return
    else:
        print("Did you forget the last (/) slash?")
        exit()


def get_inputsize(genotype_path):
    """
    Loads the first HDF5 file (.h5) as a File object and returns the input size of that File
    
    Input:
    - genotype_path (String): Path where the genotype matrices files are stored in the genotype.h5 format

    Output:
    - inputsize (Integer): 
    """
    
    single_genotype_path = glob.glob(genotype_path + '*.h5')[0]
    h5file = tables.open_file(single_genotype_path, "r")
    inputsize = h5file.root.data.shape[1]
    h5file.close()
    return inputsize


def get_labels(datapath, set_number):
    """
    Extract the phenotypic labels (integers) from subjects.csv file that matches the set_number 
    
    Inputs:
    - datapath (String): Datapath where topology.csv, subjects.csv and other important numpy files stored
    - set_number (Integer): The set to which a patient belongs (1 = training set, 2 = validation set, 3 = test, others= ignored)

    Output:
    - ybatch (Numpy array of integers): Array of phenotypic labels representing 0s and 1s for classification and values for regression
    """
    groundtruth = pd.read_csv(datapath + "/subjects.csv")
    groundtruth = groundtruth[groundtruth["set"] == set_number]
    ybatch = np.reshape(np.array(groundtruth["labels"].values), (-1, 1))
    return ybatch


def get_data(datapath, genotype_path, set_number):
    """
    Extract Genotype File object with the genotype row values that match the set_number and Array of phenotypic labels representing 0s and 1s for classification and values for regression
    
    Inputs:
    - datapath (String): Datapath where topology.csv, subjects.csv and other important numpy files stored
    - genotype_path (String): Path where the genotype matrices files are stored in the genotype.h5 format
    - set_number (Integer): The set to which a patient belongs (1 = training set, 2 = validation set, 3 = test, others= ignored)

    Outputs:
    - xbatch: Genotype File object with the genotype row values that match the set_number 
    - ybatch (Numpy array of integers): Array of phenotypic labels representing 0s and 1s for classification and values for regression
    """
    print("depreciated")
    groundtruth = pd.read_csv(datapath + "/subjects.csv")
    h5file = tables.open_file(genotype_path + "genotype.h5", "r")
    groundtruth = groundtruth[groundtruth["set"] == set_number]
    xbatchid = np.array(groundtruth["genotype_row"].values, dtype=np.int64)
    xbatch = h5file.root.data[xbatchid, :]
    ybatch = np.reshape(np.array(groundtruth["labels"].values), (-1, 1))
    h5file.close()
    return xbatch, ybatch





class TrainDataGenerator(K.utils.Sequence):
    """
    TrainDataGenerator class that inherits from tf.keras.utils.Sequence object for fitting to a sequence of data. Advantage over a typical generator object is that
    this class will guarantee that the network only trains once on each sample per epoch.
    
    Every tf.keras.utils.Sequence object must implement __getitem__ and __len__ methods. 
    """

    
    def __init__(self, datapath, genotype_path, batch_size, trainsize, inputsize, epoch_size, shuffle=True, one_hot=False):
        """
        Constructor for class TrainDataGenerator. Sets necessary attributes. Shuffle training indexes attribute depending on one_hot boolean.

        Inputs:
        - datapath (String): Datapath where topology.csv, subjects.csv and other important numpy files stored
        - genotype_path (String): Path where the genotype matrices files are stored in the genotype.h5 format
        - batch_size (Integer): batch size for training
        - trainsize (Integer): Size of the training data
        - inputsize (Integer): Size of the input data
        - epoch_size (Integer): number of epochs to train
        - shuffle (Boolean): whether or not to shuffle training dataset
        - one_hot (Boolean): whether or not to one_hot encode xbatch data

        Output:
        None 
        """
        
        self.datapath = datapath
        self.batch_size = batch_size
        self.genotype_path = genotype_path
        self.shuffledindexes = np.arange(trainsize)
        self.trainsize = trainsize
        self.multi_h5 = len(glob.glob(self.genotype_path + '*.h5')) > 1
        self.h5filenames = "_UKBB_MRI_QC_T_M"
        self.training_subjects = pd.read_csv(self.datapath + "/subjects.csv")
        self.training_subjects = self.training_subjects[self.training_subjects["set"] == 1]
        self.inputsize = inputsize
        self.epoch_size = epoch_size
        self.left_in_greater_epoch = trainsize
        self.count_after_shuffle = 0
        self.one_hot = one_hot

        if shuffle:
            np.random.shuffle(self.shuffledindexes)

    def __len__(self):
        """
        Return the number of batches (Integer) in the Sequence.
        """
        return int(np.ceil(self.epoch_size / float(self.batch_size)))
        

    def __getitem__(self, idx):
        """
        Gets batch of xbatch, ybatch data at position idx from genotype matrix
        
        Input:
        - idx (Integer): position of the batch in the Sequence 

        Outputs:
        xbatch, ybatch: batch data at position idx with array of genotype row values and phenotypic labels
        """
        if self.multi_h5:
            xbatch, ybatch = self.multi_genotype_matrix(idx)
        else:
            xbatch, ybatch = self.single_genotype_matrix(idx)

        return xbatch, ybatch


    def if_one_hot(self, xbatch): 
        """
        One-hot encode the xbatch data if the dimensions are equal to 2, otherwise return as is

        Input:
        - xbatch: Array of genotype row values

        Output:
        - xbatch: one-hot encoded input values if xbatch_dim shape == 2
        """
        xbatch_dim = len(xbatch.shape) 
        if self.one_hot:
            if xbatch_dim == 3:
                pass
            elif xbatch_dim == 2:
                xbatch = K.utils.to_categorical(np.array(xbatch, dtype=np.int8))
            else:
                print("unexpected shape!")   
        return xbatch
    
    
    def single_genotype_matrix(self, idx):
        """
        Returns the idx shifted shuffled batch data of the single genotype matrix as well as the xcov values
        
        Input:
        - idx (Integer): position of the batch in the Sequence 

        Output:
        - [xbatch, xcov] (Tuple): batch data of genotype row labels and cov columns of updated shuffled batch indexes
        - ybatch (Numpy Array): batch data of phenotypic labels of updated shuffled batch indexes 
        """
        idx2 = idx + self.count_after_shuffle  # Add index to rolling counter      
        genotype_hdf = tables.open_file(self.genotype_path + "/genotype.h5", "r")
        batchindexes = self.shuffledindexes[idx2 * self.batch_size:((idx2 + 1) * self.batch_size)] # subsets the shuffled index
        ybatch = self.training_subjects["labels"].iloc[batchindexes] # subsets the training subject labels with the subsetted batch index 
        xcov = self.training_subjects.filter(like="cov_").iloc[batchindexes] # subsets the training subject covariance values with the subsetted batch index 
        xcov = xcov.values
        xbatchid = np.array(self.training_subjects["genotype_row"].iloc[batchindexes], dtype=np.int64)
        xbatch = genotype_hdf.root.data[xbatchid, :] 
        xbatch = self.if_one_hot(xbatch)
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        genotype_hdf.close()
        return [xbatch, xcov], ybatch

    def multi_genotype_matrix(self, idx):
        """
        Returns the idx shifted shuffled batch data of all the multiple genotype matrices as well as the xcov values
        
        Input:
        - idx (Integer): position of the batch in the Sequence 

        Output:
        - [xbatch, xcov] (Tuple): batch data of genotype row labels and cov columns of updated shuffled batch indexes
        - ybatch (Numpy Array): batch data of phenotypic labels of updated shuffled batch indexes 
        """
        
        idx2 = idx + self.count_after_shuffle 
        batchindexes = self.shuffledindexes[idx2 * self.batch_size:((idx2 + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        
        xcov = self.training_subjects.filter(like="cov_").iloc[batchindexes]
        xcov = xcov.values
        
        subjects_current_batch = self.training_subjects.iloc[batchindexes]
        subjects_current_batch["batch_index"] = np.arange(len(subjects_current_batch))
        xbatch = np.zeros((len(ybatch), self.inputsize))
        for i in subjects_current_batch["chunk_id"].unique(): # looping through the training subjects corresponding to the shuffled batch indexes, for every chunk_id
            genotype_hdf = tables.open_file(self.genotype_path + "/" + str(i) + self.h5filenames + ".h5", "r")
            subjects_current_chunk = subjects_current_batch[subjects_current_batch["chunk_id"] == i]
            xbatchid = np.array(subjects_current_chunk["genotype_row"].values, dtype=np.int64)
            if len(xbatchid) > 1:
                pass
            else:
                xbatchid = int(xbatchid)
            xbatch[subjects_current_chunk["batch_index"].values, :] = genotype_hdf.root.data[xbatchid, :]
            genotype_hdf.close()
            
        xbatch = self.to_one_hot(xbatch)
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        return [xbatch, xcov], ybatch

    def on_epoch_end(self):
        """Updates indexes after each epoch, shuffles index if remaining data is less than the epoch size, otherwise updates the shuffle counter"""
        left_in_epoch = self.left_in_greater_epoch - self.epoch_size
        print(left_in_epoch, 'left_in_epoch')
        if  left_in_epoch < self.epoch_size: 
            print("Shuffeling epochs")
            np.random.shuffle(self.shuffledindexes)
            self.left_in_greater_epoch = self.trainsize
            self.count_after_shuffle = 0
        else:
            self.left_in_greater_epoch = self.left_in_greater_epoch - self.epoch_size
            self.count_after_shuffle = self.count_after_shuffle + int(np.ceil(self.epoch_size / float(self.batch_size)))
            
          


class EvalGenerator(K.utils.Sequence):

    def __init__(self, datapath, genotype_path, batch_size, setsize, inputsize, evalset="undefined", one_hot=False):
        """
        Constructor for class EvalGenerator. Sets necessary attributes. Setting evaluation set to validation or test depending on input

        Inputs:
        - datapath (String): Datapath where topology.csv, subjects.csv and other important numpy files stored
        - genotype_path (String): Path where the genotype matrices files are stored in the genotype.h5 format
        - batch_size (Integer): batch size for training
        - setsize (Integer): Size of the set data
        - inputsize (Integer): Size of the input data
        - evalset (String): Argument for specifying "validation" or "test" set to use
        - one_hot (Boolean): whether or not to one_hot encode xbatch data

        Output:
        None 
        """
        
        self.datapath = datapath
        self.batch_size = batch_size
        self.yvalsize = setsize
        self.inputsize = inputsize
        self.genotype_path = genotype_path
        self.h5file = []
        self.h5filenames = "_UKBB_MRI_QC_T_M"
        self.multi_h5 = len(glob.glob(self.genotype_path + '*.h5')) > 1
        self.eval_subjects = pd.read_csv(self.datapath + "/subjects.csv")
        self.one_hot = one_hot
        
        if evalset == "validation":
            self.eval_subjects = self.eval_subjects[self.eval_subjects["set"] == 2]
        elif evalset == "test":
            self.eval_subjects = self.eval_subjects[self.eval_subjects["set"] == 3]
        else:
            print("please add which evalset should be used in the call, validation or test. Currently undefined")

    def __len__(self):
        """
        Return the number of batches (Integer) in the Sequence.
        """
        
        val_len = int(np.ceil(self.yvalsize / float(self.batch_size)))
        return val_len

    def __getitem__(self, idx):
        """
        Gets batch of xbatch, ybatch data at position idx from genotype matrix
        
        Input:
        - idx (Integer): position of the batch in the Sequence 

        Outputs:
        xbatch, ybatch: batch data at position idx with array of genotype row values and phenotypic labels
        """
        
        if self.multi_h5:
            xbatch, ybatch = self.multi_genotype_matrix(idx)
        else:
            xbatch, ybatch = self.single_genotype_matrix(idx)

        return xbatch, ybatch

    def if_one_hot(self, xbatch):  
        """
        One-hot encode the xbatch data if the dimensions are equal to 2, otherwise return as is

        Input:
        - xbatch: Array of genotype row values

        Output:
        - xbatch: one-hot encoded input values if xbatch_dim shape == 2
        """
        
        xbatch_dim = len(xbatch.shape) 
        if self.one_hot:
            if xbatch_dim == 3:
                pass
            elif xbatch_dim == 2:
                xbatch = K.utils.to_categorical(np.array(xbatch, dtype=np.int8))
            else:
                print("unexpected shape!")   
        return xbatch
    
    def single_genotype_matrix(self, idx):
        """
        Returns the idx batch data of the single genotype matrix of the evaluation subjects as well as the xcov values
        
        Input:
        - idx (Integer): position of the batch in the Sequence 

        Output:
        - [xbatch, xcov] (Tuple): batch data of genotype row labels and cov columns of the evaluation subjects
        - ybatch (Numpy Array): batch data of phenotypic labels of the evaluation subjects
        """
        
        genotype_hdf = tables.open_file(self.genotype_path + "/genotype.h5", "r")
        ybatch = self.eval_subjects["labels"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        xcov = self.eval_subjects.filter(like="cov_").iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        xcov = xcov.values
        xbatchid = np.array(self.eval_subjects["genotype_row"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)],
                            dtype=np.int64)
        xbatch = genotype_hdf.root.data[xbatchid, :]  
        xbatch = self.if_one_hot(xbatch)
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        genotype_hdf.close()
        return [xbatch, xcov], ybatch


    def multi_genotype_matrix(self, idx):     
        """
        Returns the idx batch data of the single genotype matrix of the evaluation subjects in the multiple genotype matrices as well as the xcov values
        
        Input:
        - idx (Integer): position of the batch in the Sequence 

        Output:
        - [xbatch, xcov] (Tuple): batch data of genotype row labels and cov columns of the evaluation subjects
        - ybatch (Numpy Array): batch data of phenotypic labels of the evaluation subjects
        """
        
        subjects_current_batch = self.eval_subjects.iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        subjects_current_batch["batch_index"] = np.arange(subjects_current_batch.shape[0])
              
        xbatch = np.zeros((len(subjects_current_batch["labels"]), self.inputsize))

        for i in subjects_current_batch["chunk_id"].unique(): # Looping through the current batch of evaluation subjects by unique chunk_id values
            genotype_hdf = tables.open_file(self.genotype_path + "/" + str(i) + self.h5filenames + ".h5", "r")
            subjects_current_chunk = subjects_current_batch[subjects_current_batch["chunk_id"] == i]
            xbatchid = np.array(subjects_current_chunk["genotype_row"].values, dtype=np.int64)
            xbatch[subjects_current_chunk["batch_index"].values, :] = genotype_hdf.root.data[xbatchid, :]
            genotype_hdf.close()
            
        xbatch = self.to_one_hot(xbatch)
        ybatch = np.reshape(np.array(subjects_current_batch["labels"]), (-1, 1))
        xcov = subjects_current_batch.filter(like="cov_").values                     
        return [xbatch, xcov], ybatch
    



