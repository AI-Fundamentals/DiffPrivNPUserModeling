import h5py
import numpy as np
import torch


class HDFDataset_torch(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for reading data from an HDF5 file.

    Parameters
    ----------
    filepath : str
        The path to the HDF5 file.
    n_users : int, optional
        The number of users to select, by default 16

    Attributes
    ----------
    filepath : str
        The path to the HDF5 file.
    n_users : int
        The number of users to select from the file.
    batch_names : list
        The names of the batches in the HDF5 file.
    selected_batches : list
        The names of the selected batches.
    """

    def __init__(self, filepath, n_users=16):
        self.filepath = filepath
        self.n_users = n_users
        with h5py.File(self.filepath, 'r') as hf:
            self.batch_names = list(hf['data'].keys())
        np.random.shuffle(self.batch_names)
        self.selected_batches = self.batch_names[:self.n_users]

    def __len__(self):
        """
        Returns the number of selected batches.

        Returns
        -------
        int
            The number of selected batches.
        """
        return len(self.selected_batches)

    def __getitem__(self, idx):
        """
        Returns the data for a given batch index.

        Parameters
        ----------
        idx : int
            The index of the batch to return.

        Returns
        -------
        tuple
            The xc, yc, xt, yt data for the batch.
        """
        with h5py.File(self.filepath, 'r') as hf:
            batch_name = self.selected_batches[idx]
            batch_group = hf['data'][batch_name]
            xc = torch.from_numpy(batch_group['xc'][:])
            yc = torch.from_numpy(batch_group['yc'][:])
            xt = torch.from_numpy(batch_group['xt'][:])
            yt = torch.from_numpy(batch_group['yt'][:])
        return xc, yc, xt, yt



def hdf_to_dataloader_pad(filepath, n_users=16, batch_size=1, padding_value=-1.):
    """
    Load an HDF dataset and pad it into a PyTorch DataLoader.

    Parameters
    ----------
    filepath : str
        Path to the HDF file.
    n_users : int, optional
        Number of users to be loaded from the dataset. Default is 16.
    batch_size : int, optional
        Size of the batches. Default is 1.
    padding_value : float, optional
        Value used for padding, Default is -1.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader object with the padded dataset.

    """
    dataset = HDFDataset_torch(filepath, n_users)
    
    def custom_pad(data, padding_value=-1.):
        """
        Custom padding function for 3D tensors that vary in the size of
        dimensions 0 and 2.

        Parameters
        ----------
        data : list of torch.Tensor
            List of tensors to be padded.
        padding_value : float, optional
            Value used for padding. Default is -1.

        Returns
        -------
        torch.Tensor
            Padded tensor.

        """
        max_len = max([x.size(0) for x in data])
        max_dim2 = max([x.size(2) for x in data])
        padded = torch.full((len(data), max_len, data[0].size(1), max_dim2), padding_value)

        for i, tensor in enumerate(data):
            padded[i, :tensor.size(0), :, :tensor.size(2)] = tensor
        return padded
    
    def collate_fn(batch):
        """
        Function to collate data into a padded batch.

        Parameters
        ----------
        batch : list
            List of data to be collated.

        Returns
        -------
        tuple of torch.Tensor
            Tuple of collated, padded data.

        """
        xc, yc, xt, yt = zip(*batch)
        xc = custom_pad(xc, padding_value=padding_value)
        yc = custom_pad(yc, padding_value=padding_value)
        xt = custom_pad(xt, padding_value=padding_value)
        yt = custom_pad(yt, padding_value=padding_value)
        return xc, yc, xt, yt
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Load metadata from the HDF file
    metadata = hdf_get_metadata(filepath)
    # Augment the metadata
    metadata['n_users'] = min(n_users,metadata['n_users'])
    metadata['batch_size'] = batch_size
    metadata['n_batches'] = int(np.ceil(metadata['n_users']/batch_size))
    
    return dataloader, metadata



def print_hdf_structure(file_name):
    """
    Print the structure of an hdf5 file
    """
    with h5py.File(file_name, 'r') as f:
        f.visit(print)



def hdf_get_metadata(filepath):
    """
    Return the metadata of an HDF5 file, which must be in the 'metadata' group.

    Parameters
    ----------
    filepath : str
        The path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary containing the metadata. Each key-value pair in the
        dictionary corresponds to a name-item pair in the metadata.

    """
    metadata_dict = {}
    with h5py.File(filepath, 'r') as hf:
        # Add all items from the 'metadata' group to the dictionary
        if 'metadata' in hf:
            for name, item in hf['metadata'].attrs.items():
                # Convert numpy data types to native Python types
                if isinstance(item, (np.int64,np.uint8,np.float64)):
                    item = item.item()
                # Convert byte strings to Python strings
                elif isinstance(item, np.bytes_):
                    item = item.decode('utf-8')
                
                metadata_dict[name] = item
        else:
            raise ValueError(f"File {filepath} does not contain any metadata.")
    return metadata_dict

    
def concat_user_hdf_files(filepaths, output_filepath):
    """
    Concatenates multiple HDF5 files into a new file. The keys for the output
    file will be a contiguous list "user_1", "user_2" etc. with no repeats
    regardless of what the input keys are.

    Parameters
    ----------
    filepaths : list of str
        The paths to the HDF5 files to concatenate.
    output_filepath : str
        The path to the output HDF5 file.
    """
    
    # First check / concat the metadata
    metadata_list = [hdf_get_metadata(filepath) for filepath in filepaths]
    
    # Check all eval are compatible
    for index, metadata in enumerate(metadata_list[1:]):
        if metadata["eval"] != metadata_list[0]['eval']:
            raise ValueError(f"The hdf files all need to have the same 'eval' values. File {index+1} has a different value to file 0.")
        if metadata["gen_type"] != metadata_list[0]['gen_type']:
            raise ValueError(f"The hdf files all need to have the same 'gen_type' values. File {index+1} has a different value to file 0.")
        if metadata["n_traj"] != metadata_list[0]['n_traj']:
            raise ValueError(f"The hdf files all need to have the same 'n_traj' values. File {index+1} has a different value to file 0.")
        # different n_users are compatible
        if metadata["noise_variance"] != metadata_list[0]['noise_variance']:
            raise ValueError(f"The hdf files all need to have the same 'noise_variance' values. File {index+1} has a different value to file 0.")
        if metadata["p_bias"] != metadata_list[0]['p_bias']:
            raise ValueError(f"The hdf files all need to have the same 'p_bias' values. File {index+1} has a different value to file 0.")
        
    #Make new metadata
    total_users = sum(d["n_users"] for d in metadata_list)
    new_metadata = metadata_list[0]
    new_metadata['n_users'] = total_users
    
    user_counter = 1
    with h5py.File(output_filepath, 'w') as hf_out:
        # Create a group for metadata and add new_metadata to it
        metadata_group = hf_out.create_group('metadata')
        metadata_group.attrs.update(new_metadata)

        # Create a group for data
        data_group = hf_out.create_group('data')

        for filepath in filepaths:
            with h5py.File(filepath, 'r') as hf_in:
                for key in hf_in['data'].keys():
                    # Create a new key for each dataset to ensure uniqueness
                    new_key = f"user_{user_counter}"
                    user_counter += 1
                    # Copy each dataset from the input file to the data group in the output file
                    hf_in['data'].copy(key, data_group, new_key)

