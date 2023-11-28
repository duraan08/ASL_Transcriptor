import h5py

hdf5_file = h5py.File('C:/Universidad/TFG/Desarrollo/data_vector/landmarks_data.hdf5', 'r')
video_list = list(hdf5_file.keys())

for video in video_list:
    video_group = hdf5_file[video]
    print(f"Video : {video}")

    dt = list(video_group.keys())

    for dataset in dt:
        dat = video_group[dataset]
        print(f"Dataset: {dataset}")

        landmarks = dat[:]
        print(f"Landmarks: {landmarks}")

hdf5_file.close()