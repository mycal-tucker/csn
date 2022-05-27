import h5py
import numpy as np
import os
from data_parsing.bolts.data_store import DataStore


class DataParser:
    SUPPORT = 'support'
    num_dims = 3

    def __init__(self, data_path='../../data/bolts/trajectories/', target_data_path='../../data/bolts/targets/'):
        self.data_path = data_path
        self.target_data_path = target_data_path
        self.string_to_targ_idx = {'0': 0,
                                   '1': 1,
                                   '2': 2,
                                   '3': 3,
                                   '4': 4,
                                   '5': 5,
                                   '6': 6,
                                   '7': 7,
                                   'd': 8}
        self.debug = False

    def get_trajectories(self):
        return self.get_data(self.data_path)[0]

    def get_all_targets(self):
        targets = self.get_data(self.target_data_path)[1]
        array_version = [np.asarray(targ) for targ in targets]
        transposed_verions = [np.transpose(targ) for targ in array_version]
        only_dims_version = [targ[:DataParser.num_dims] for targ in transposed_verions]
        return only_dims_version

    def get_traj_and_targ(self):
        trajectories = self.get_trajectories()
        raw_target_locations = self.get_targets()
        # Merge the target location stuff
        new_targets = np.zeros(raw_target_locations.shape)
        for i in range(new_targets.shape[0]):
            for j in range(new_targets.shape[1]):
                relevant_min = trajectories.run_id_to_mins.get(0)[i]
                relevant_scale = 1000 * trajectories.run_id_to_scale.get(0)[i]
                relevant_raw = raw_target_locations[i, j]
                new_targets[i, j] = (relevant_raw - relevant_min) / relevant_scale
        targets = new_targets
        return trajectories, targets

    def get_targets(self):
        targets = self.get_data(self.target_data_path)[1]
        # Merge the target data into one thing representing the locations of all the targets.
        num_targets = len(targets[0])
        averaged_targets = [[] for _ in range(DataParser.num_dims)]
        for target_idx in range(num_targets):
            for dim in range(DataParser.num_dims):
                just_dim = []
                for targets_in_file in targets:
                    target_in_file = targets_in_file[target_idx]
                    just_dim.append(target_in_file[dim])
                # Average the point along that dimension
                average_dim_value = np.mean(just_dim)
                averaged_targets[dim].append(average_dim_value)
        return np.asarray(averaged_targets)

    def get_data(self, path):
        # Create the structure I'll want to populate with the data I read from this file.
        data_store = DataStore()
        # Iterate over each of the h5 files in the directory
        # Every file is for a different subject.
        subject_id = 0
        targets = []
        for filename in os.listdir(path):
            targets_for_file = []
            if not filename.endswith('.h5'):
                print("Error! Can't read filename", filename)
                continue
            if 'cls' in filename:
                # print("Skipping filename", filename)
                continue
            # At this point, we know it's an h5 file, so read it in using h5py.
            # Have to add data_path to filename to actually read it in.
            if self.debug:
                print("Reading file", filename)
            file = h5py.File(path + filename, 'r')
            # h5 files are like dictionaries.
            for key, group in file.items():
                if key == 'ignore-list':
                    continue
                if self.debug:
                    print("Continuing with key", key)
                for group_key, group_entry in group.items():
                    if self.debug:
                        print("group key", group_key)
                        print("group val", group_entry)
                    if group_key == 'offset':
                        offsets = group_entry[()]
                        continue
                    if group_key == 'positions':
                        if self.debug:
                            print("Skipping group key", group_key)
                        continue
                    data_val = None
                    for subgroup_key, subgroup_entry in group_entry.items():
                        if self.debug:
                            print("subgroup key", subgroup_key)
                            print("subgroup entry", subgroup_entry)
                        if subgroup_key == 'label':
                            label_val = subgroup_entry[()]
                            targets = []
                            # Reformat the string version into a numerical thing for computation
                            for label in label_val:
                                # Split the string by the hyphen
                                str_label = label.decode('utf-8')
                                hyphen_idx = str_label.find('- ')
                                # goal_event = str_label[hyphen_idx + 2:]
                                goal_event = str_label[-1:]
                                if self.debug:
                                    print("goal event", goal_event)
                                # Fetch the proper int representation of the target
                                if goal_event not in self.string_to_targ_idx.keys():
                                    self.string_to_targ_idx[goal_event] = len(self.string_to_targ_idx.keys())
                                    print("String", goal_event, "maps to id", len(self.string_to_targ_idx.keys()) - 1)
                                targets.append(self.string_to_targ_idx.get(goal_event))
                            continue
                        if subgroup_key == 'data':
                            data_val = subgroup_entry[()]
                            continue
                        if subgroup_key == 'mean_01':  # The data for the mean position of the target?
                            vals = subgroup_entry[()]
                            mean1 = np.mean(vals, axis=1)
                            targets_for_file.append(mean1)
                            pass

                    # Once you've exited the loop, have both labels and the spatial points.
                    if data_val is None:
                        continue
                    data_store.add_data(subject_id, data_val, np.asarray(targets))
            # Update the targets for file by the offset
            if len(targets_for_file) > 0:
                for target_for_file in targets_for_file:
                    for i, offset in enumerate(offsets):
                        target_for_file[i] = target_for_file[i] - offset  # Unclear if offset should be added or subtracted.
            targets.append(targets_for_file)
            if self.debug:
                print()
                print("NEW PERSON!!!!!!!!!!!!!!!!!!!!!")
            subject_id += 1
        return data_store, targets
