import numpy as np


# Super basic object just meant to store the data I want for training and testing. I'm not sure
# of exactly what data formats I'll want, so what methods I exposes, etc, may change in the future.
class DataStore:
    num_dims = 3
    def __init__(self):
        self.all_runs = []
        self.subject_id_to_runs = {}
        self.subject_id_to_sample_run_id = {}
        self.run_id_to_subj_id = {}
        self.run_id_to_scale = {}  # Map from the run id to the
        self.subj_to_run_id_to_scale = {}  # Same, but for each subject.
        self.run_id_to_mins = {}
        self.subj_to_run_id_to_mins = {}
        self.debug = False

    def get_subj_id_order(self, subj_id):
        for i, s_id in enumerate(sorted(self.subject_id_to_runs.keys())):
            if s_id == subj_id:
                return i
        return None

    def add_data(self, subject_id, single_run, label_data, normalize=True):
        any_nans = np.isnan(single_run).any()
        assert not any_nans, "Found a nan in the data"
        fetched_data = self.subject_id_to_runs.get(subject_id)
        if fetched_data is None:
            # No existing entry, so add empty list as entry.
            fetched_data = []
            self.subject_id_to_runs[subject_id] = fetched_data
        # Add the run to fetched data
        reformatted_run = np.zeros((DataStore.num_dims + 1, single_run.shape[1]))
        if normalize:
            range_vals = []
            min_vals = []
            for i in range(DataStore.num_dims):
                max_val = max(single_run[i, :])
                min_val = min(single_run[i, :])
                range_val = max_val - min_val
                reformatted_run[i, :] = (single_run[i, :] - min_val) / range_val
                range_vals.append(range_val / 1000)  # In meters, so normalize by 1000
                min_vals.append(min_val)
            self.run_id_to_scale[len(self.all_runs)] = range_vals
            self.run_id_to_mins[len(self.all_runs)] = min_vals
            num_runs_for_subject = len(fetched_data)
            subj_run_id_dict = self.subj_to_run_id_to_scale.get(subject_id)
            if subj_run_id_dict is None:
                subj_run_id_dict = {}
                self.subj_to_run_id_to_scale[subject_id] = subj_run_id_dict
            subj_run_id_dict[num_runs_for_subject] = range_vals
            subj_run_id_min_dict = self.subj_to_run_id_to_mins.get(subject_id)
            if subj_run_id_min_dict is None:
                subj_run_id_min_dict = {}
                self.subj_to_run_id_to_mins[subject_id] = subj_run_id_min_dict
            subj_run_id_min_dict[num_runs_for_subject] = min_vals
        reformatted_run[-1, :] = label_data  # Add in the label data at the end.
        fetched_data.append(reformatted_run)
        # if self.subject_id_to_sample_run_id.get(subject_id) is None:
        self.subject_id_to_sample_run_id[subject_id] = len(self.all_runs)
        self.run_id_to_subj_id[len(self.all_runs)] = subject_id
        self.all_runs.append(reformatted_run)

    def get_scales(self, run_id):
        return self.run_id_to_scale.get(run_id)

    def get_scales_by_subect_id(self, subj_id, run_id):
        return self.subj_to_run_id_to_scale.get(subj_id).get(run_id)

    def get_num_runs_for_subject(self, subject_id):
        return len(self.subject_id_to_runs.get(subject_id))

    def get_hold_some_out_data(self, duration, step_size=10, test_idxs=None):
        return self.__get_hold_some_out_data__(duration, self.all_runs, step_size=step_size, test_idxs=test_idxs)

    # Generate training data from every single run except 1, and use that last run for the testing data.
    # Within each run, generate different trajectories by starting at different times. For now, generate those
    # starting times by marching at a fixed step size through the trajectory.
    def __get_hold_some_out_data__(self, duration, runs, step_size=10, test_idxs=None, max_runs_to_do=np.inf):
        # If the test idx isn't set, choose one randomly.
        if test_idxs is None:
            test_idxs = [int(np.random.random() * len(runs))]
        test_past_goals = []
        test_past = []
        test_subj_id = []
        train_past_goals = []
        train_past = []
        train_subj_id = []
        step_size = step_size
        if self.debug:
            print("Holding out test idxs", test_idxs, "of", len(runs), "num runs")
        for i, full_run in enumerate(runs):
            if self.debug:
                print("Considering run number", i)
            subj_id = self.run_id_to_subj_id.get(i)
            if i in test_idxs:
                if self.debug:
                    print("Generating data for test run")
                test_data = DataStore.__generate_data_from_run__(full_run, duration, step_size=step_size, shuffle=True)
                assert test_data is not None, "Couldn't generate data for specified test index " + str(i)
                past_goals, past = test_data
                test_past_goals.extend(past_goals)
                test_past.extend(past)
                test_subj_id.extend([subj_id for _ in range(len(past_goals))])
                continue
            # Generate data from the run.
            if self.debug:
                print("Generating data for training run")
            # Don't need to shuffle within a run for training data because will shuffle after anyway.
            data_from_run = DataStore.__generate_data_from_run__(full_run, duration, step_size=step_size)
            if data_from_run is None:
                continue
            past_goals, past = data_from_run
            if subj_id % 2 == 0:
                for subrun_idx in range(past_goals.shape[0]):
                    if past_goals[subrun_idx, -1] % 2 == 0:
                        # print("Skipping")
                        continue
                    else:
                        train_past_goals.append(past_goals[subrun_idx])
                        train_past.append(past[subrun_idx])
                        train_subj_id.append(subj_id)
            else:
                train_past_goals.extend(past_goals)
                train_past.extend(past)
                train_subj_id.extend([subj_id for _ in range(len(past_goals))])
            # For debugging/sanity check, only do first few runs.
            if i > max_runs_to_do:
                break
        # Reformat into arrays.
        test_past_goals = np.asarray(test_past_goals)
        test_past = np.asarray(test_past)
        test_subj = np.asarray(test_subj_id)
        train_past_goals = np.asarray(train_past_goals)
        train_past = np.asarray(train_past)
        train_subj = np.asarray(train_subj_id)
        # Shuffle the test data.
        test_perm = np.random.permutation(test_past_goals.shape[0])
        test_past_goals = test_past_goals[test_perm, :]
        reformatted_test_past_goals = np.asarray(test_past_goals[:, -1])
        test_past = test_past[test_perm, :]
        test_subj = test_subj[test_perm]
        # Shuffle the data. This now mixes the data up across runs and within runs.
        row_permutation = np.random.permutation(train_past_goals.shape[0])
        train_past_goals = train_past_goals[row_permutation, :]
        reformatted_past_goals = np.asarray(train_past_goals[:, -1])
        train_past = train_past[row_permutation, :]
        train_subj = train_subj[row_permutation]

        # Reformat test goals
        return (reformatted_past_goals, train_past, train_subj), (reformatted_test_past_goals, test_past, test_subj)

    @staticmethod
    def __generate_data_from_run__(full_run, duration, step_size, shuffle=False, max_time=np.inf):
        past_goals = []
        past_trajectories = []
        # Fancy padding logic. Because I need a fixed-size window, I can't normally predict from the very start
        # of thr trajectory. But I want to do that. So, I create "padded" sequences at the start that duplicate
        # the initial position as many times as needed to fill out the fixed length that I want.
        # for start_idx in range(-duration, 0, step_size):
        #     num_padded_points = -1 * start_idx  # How many will I have to make up.
        #     padded_subsection = np.transpose(np.asarray([full_run[:, 0] for _ in range(num_padded_points)]))
        #     actual_subsection = full_run[:, :2 * duration - num_padded_points]
        #     # Merge the two subsections together to create a past and future.
        #     catted = np.concatenate((padded_subsection, actual_subsection), axis=1)
        #     past_subsection = catted[:DataStore.num_dims, :duration]
        #     future_subsection = catted[:DataStore.num_dims, duration:]
        #     if future_subsection.shape[1] < duration:
        #         break
        #     reshaped_past = np.reshape(past_subsection, (1, DataStore.num_dims * duration))
        #     reshaped_future = np.reshape(future_subsection, (1, DataStore.num_dims * duration))
        #     past_trajectories.append(reshaped_past)
        #     # Same idea of padding, but for goals now.
        #     past_goals.append(catted[DataStore.num_dims, :duration])

        for start_idx in range(0, full_run.shape[1] - duration, step_size):
            middle_idx = start_idx + duration
            past_subsection = full_run[:DataStore.num_dims, start_idx:middle_idx]
            # Reshape the points in space (2 or 3D) into a vector.
            reshaped_past = np.reshape(past_subsection, (1, DataStore.num_dims * duration))

            subgoals = full_run[DataStore.num_dims, start_idx:middle_idx]
            if 8 in subgoals:
                continue
            past_goals.append(subgoals)
            past_trajectories.append(reshaped_past)
        if len(past_goals) == 0:
            print("Unable to create data for full run with", full_run.shape[1], "frames")
            return
        # Lastly, reformat the lists into arrays.
        past_goals = np.asarray(past_goals)
        past_trajectories = np.asarray(past_trajectories)
        past_trajectories = past_trajectories.reshape((past_trajectories.shape[0], past_trajectories.shape[2]))
        if shuffle:
            # print("Shuffling")
            # Shuffle all the parts of the data so the order in the data isn't a function of time.
            row_permutation = np.random.permutation(past_goals.shape[0])
            # Use the same permutation to apply to all 3 data structures
            past_goals = past_goals[row_permutation, :]
            past_trajectories = past_trajectories[row_permutation, :]
        return past_goals, past_trajectories


    def get_subj_to_global_run_id(self):
        subj_to_run_id = {}
        for run_id, subj_id in self.run_id_to_subj_id.items():
            if subj_id not in subj_to_run_id.keys():
                subj_to_run_id[subj_id] = []
            subj_to_run_id.get(subj_id).append(run_id)
        return subj_to_run_id