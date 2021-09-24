import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import mne
import scipy
import pickle
from mayavi import mlab
from IPython.display import Image
from mne.decoding import get_coef

mlab.init_notebook('png')
mne.viz.set_3d_backend('mayavi')

# Functions for loading/splitting data

def getEpochData(s_id, dataFolder, sensors):
    if int(s_id) < 23:
        fname = dataFolder + 'S' + s_id + '\\' + s_id + '_2_tsss_mc_trans_' + sensors + '_nobase-epochs_afterICA_manually_AR_resampled.fif'
    else:
        fname = dataFolder + 'S' + s_id + '\\block_2_tsss_mc_trans_' + sensors + '_nobase-epochs_afterICA_manually_AR_resampled.fif'

    epochs = mne.read_epochs(fname, verbose='error')
    print(fname + ' loaded!')
    return epochs


def splitData(epochs, events=None):
    # print(epochs.event_id)
    return_list = None
    if events == None:
        print('No events are given as parameter!')

    else:
        print('Requested events: ', events)
        return_list = epochs[events]

    print('Events in return list: ', return_list.event_id)

    return return_list


def getClassifierWeghts(filename):
    print('Classifier is loaded from ', filename)
    loaded_model = []
    with open(filename, "rb") as f:
        while True:
            try:
                loaded_model.append(pickle.load(f))
            except EOFError:
                break

    return loaded_model


## Beamformer

# %%

# Compute spatial filters for beamformer

def computeSpatialFilter(s_name, s_id, evoked, noise_cov, data_cov, condName, forwardModelsFolder, spatialFiltersFolder ):
    print('Beamformer')
    print('Computing spatial filter..')

    # Load forward model
    fname_fwd = forwardModelsFolder + '\\fwd_sol_ico5_' + s_name + '.fif'
    fwd = mne.read_forward_solution(fname_fwd, verbose=False)

    # Compute spatial filters with evoked data, forward model and covariance matrices
    filters = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov=data_cov, reg=0.05, noise_cov=noise_cov,
                                       pick_ori='max-power', weight_norm='unit-noise-gain', rank='full',
                                       reduce_rank=True)

    # Save filters
    filters.save(spatialFiltersFolder + s_id + '_filters-lcmv_' + s_name + '.h5', overwrite=True)
    return filters


### Prepare data

#### Morph subject's source estimate to template subject
def morphToCommonSpace(stc, s_name, src_ave, subjects_dir, smoothAmount=None):
    print('Computing source morph..')
    # Read the source space we are morphing to
    fsave_vertices = [s['vertno'] for s in src_ave]

    morph = mne.compute_source_morph(src=stc, subject_from=s_name, subject_to='fsaverage',
                                     spacing=fsave_vertices, subjects_dir=subjects_dir, verbose=False,
                                     smooth=smoothAmount)
    tstep = stc.tstep

    print('Morphing data to fsaverage..')
    stc_fsave = morph.apply(stc)
    n_vertices_fsave = morph.morph_mat.shape[0]

    return stc_fsave, n_vertices_fsave, tstep


# generate the inverse solution for group average
def prepareInverseSolution_group(data, subjects_dir, tstep, tmin_tmp=0):
    src_ave = mne.read_source_spaces(subjects_dir + 'fsaverage\\bem\\fsaverage-ico-5-src.fif')

    fsave_vertices = [s['vertno'] for s in src_ave]

    stc_return = mne.SourceEstimate(data, fsave_vertices, tmin_tmp, tstep, subject='fsaverage')

    return stc_return


### Visualize results

# The STC (Source Time Courses) are defined on a source space formed by 7498 candidate locations
# and for a duration spanning 106 time instants.

# Warning: Slide Type
# !!PQt5 is necessary and also run jupyter nbextension enable mayavi --py on command line
# before running the jupyter notebooks and also latest (6.1.1) version of module called 'traits'.

def showResult(s_id, sourceFolder, stc, condName, subjects_dir, minimum, maximum, tmin_tmp=0,
               cmap='Oranges', sequentialCmap=True):

    initial_time = tmin_tmp
    hemi_list = ['rh', 'lh']
    mid = (minimum+maximum)/2
    for hemi in hemi_list:
        print('Hemi: ', hemi)
        kwargs = dict(initial_time=initial_time, surface='inflated', hemi=hemi, subjects_dir=subjects_dir,
                      verbose=True, size=(600, 600), spacing='all', background='w',
                      cortex=(211 / 256, 211 / 256, 211 / 256), colorbar=True)

        if sequentialCmap == True:
            brain = stc.plot(**kwargs, colormap=cmap, clim=dict(kind='value', lims=[minimum, mid, maximum]))
        else: # divergent colormap
            brain = stc.plot(**kwargs, colormap=cmap, clim=dict(kind='value', pos_lims=[minimum, mid, maximum]))

        # add the vertex at the peak to the plot
        # brain.add_foci(peak_vertex_surf, coords_as_verts=True, hemi=hemi, color='blue')
        brain.show_view('lateral');
        brain.save_image(sourceFolder + s_id + '_' + hemi + '_' + condName + '.png')
        Image(filename=sourceFolder + s_id + '_' + hemi + '_' + condName + '.png', width=600)


def computeActivationMaps(model_list, epochs, tmin):
    meg_data = epochs.get_data()
    epochs.average().plot()
    print("Meg data shape: ", meg_data.shape)
    coef = None
    # get classifier weights
    if len(model_list) > 0:
        model = model_list[0]  # if model is loaded, it is stored in a list. Therefore we need to get model from index 0

        # Get classifier weights
        coef = get_coef(model, 'coef_', inverse_transform=True)

        # Compute mean and std of weights
        coef_mean = np.mean(coef)
        coef_std = np.std(coef)

        # Standardize the weights
        coef = (coef - coef_mean) / coef_std
        print('shape of coef: ', coef.shape)

    # Multiplying classifier weights with covariance of data to compute activation maps
    activations_mat = np.zeros((meg_data.shape[0], meg_data.shape[1], meg_data.shape[2]))
    # ntrials, nchannels, ntimes

    for t in range(meg_data.shape[2]):
        epochs_tmp = epochs.copy()
        epochs_tmp.crop(tmin=epochs.times[t], tmax=epochs.times[t])
        cov_tmp = mne.compute_covariance(epochs_tmp, verbose=False)

        activations = np.dot(coef, cov_tmp.data)
        if t == 0:
            print('Shape of activations: ', activations.shape)

        for i in range(meg_data.shape[0]):
            activations_mat[i, :, t] = activations.reshape(meg_data.shape[1])

        del cov_tmp

        # Simulate epoch object with activation maps
    epoched_sim = mne.EpochsArray(activations_mat, epochs.info, tmin=tmin)

    return epoched_sim


### Apply Beamformer
def applyBeamformer(conditions, s_id_list_all, n_subjects, participant_names, tminData, tmaxData, tminNoise, tmaxNoise,
                    tminEpoch, smoothAmount, task_name, clsfFolder, peak_indices, src_ave,
                    spatialFiltersFolder, subjects_dir, dataFolder, sensors, forwardModelsFolder):

    stc_fsave_all_real, n_times, tstep, n_vertices_fsave = None, None, None, None

    for s in range(n_subjects):
        s_id = s_id_list_all[s]
        s_name = participant_names[s]
        print(' ------------- ' + s_name + ' ------------- ')
        epochs = getEpochData(s_id, dataFolder, sensors)
        print(epochs.event_id)
        print('epochs shape: ', epochs.get_data().shape)

        # check if all conditions exist in the epoch (e.g. omissions_living_nores might not exist!)
        conditions = [c for c in conditions if c in epochs.event_id]
        print('Final conditions: ', conditions)
        splits = epochs[conditions]
        print('Events in splits: ', splits.event_id)

        # Load classifier weights to compute activation maps
        clsf_model_filename = clsfFolder + 'S' + s_id + '\\' + s_id + '_clsf_' + task_name + '_' + str(
            peak_indices[0]) + '_' + str(peak_indices[1]) + '_sm.sav'
        clsf_model = getClassifierWeghts(clsf_model_filename)

        # compute activation maps and simulate epoch object for source localization
        print('Compute activations')
        epoch_sim = computeActivationMaps(clsf_model, splits, tmin=tminEpoch)

        print('Compute noise covariance')
        noise_cov = mne.compute_covariance(epoch_sim, tmin=tminNoise, tmax=tmaxNoise,
                                           method=['shrunk', 'empirical'], verbose=False)
        print('Compute data covariance')
        data_cov = mne.compute_covariance(epoch_sim, tmin=tminData, tmax=tmaxData,
                                          method=['shrunk', 'empirical'], verbose=False)

        # compute average of epochs
        evoked = epoch_sim.average().crop(tmin=tminData, tmax=tmaxData)
        print('Shape of evoked data: ', evoked._data.shape)

        # computer spatial filters by LCMV
        print('Compute filter: ')
        filters = computeSpatialFilter(s_name, s_id, evoked, noise_cov, data_cov, conditions,
                                       forwardModelsFolder, spatialFiltersFolder )

        print('Apply beamformer: ')
        stc = mne.beamformer.apply_lcmv(evoked=evoked, filters=filters, max_ori_out='signed')
        n_vertices_sample, n_times = stc.data.shape

        if s_id != '16' and s_id != '31' and s_id != '40':  # for participants with MR data
            stc_fsave, n_vertices_fsave, tstep = morphToCommonSpace(stc, s_name, src_ave, subjects_dir,
                                                                    smoothAmount=smoothAmount)
        else:  # for participants without MR data
            stc_fsave = stc

        # initialize the stc data array when computing the first participant
        if s == 0:
            print('n_times: ', n_times)
            stc_fsave_all_real = np.zeros((n_vertices_fsave, n_times, n_subjects), )

        stc_fsave_all_real[:, :, s] = np.abs(stc_fsave.data.reshape(n_vertices_fsave, n_times))

    return stc_fsave_all_real, n_times, tstep


def computeStatistic(x, y):
    print('comparing 2 groups')
    stats_array, pval_array = scipy.stats.ttest_rel(x, y, axis=1)

    return stats_array, pval_array

def getMNIcoordinates(hemi, stc, subjects_dir, labels=None):
    hemi_ind, peak_vertex_surf = None, None

    if hemi == 'rh':
        hemi_ind = 1  # rh
    elif hemi == 'lh':
        hemi_ind = 0

    # find the peak on the given hemi
    if labels == None:
        peak_vertex, peak_time = stc.get_peak(hemi=hemi)
    else:
        # restrict the area to find the peak in the preferred area
        peak_vertex, peak_time = stc.in_label(labels).get_peak()
    print('peak_vertex: ', peak_vertex)

    # get the vertex at the peak
    if hemi_ind == 1:
        peak_vertex_surf = stc.rh_vertno[peak_vertex]
    elif hemi_ind == 0:
        peak_vertex_surf = stc.lh_vertno[peak_vertex]

    # convert vertex to MNI coordinates
    coordinate = mne.vertex_to_mni(peak_vertex_surf, hemis=hemi_ind, subject='fsaverage', subjects_dir=subjects_dir)

    return coordinate

