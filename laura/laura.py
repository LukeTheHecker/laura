from scipy.spatial.distance import cdist
from scipy.linalg import inv
import numpy as np
import mne

def pos_from_forward(forward, verbose=0):
    ''' Get vertex/dipole positions from mne.Forward model

    Parameters
    ----------
    forward : instance of mne.Forward
        The forward model. 
    
    Return
    ------
    pos : numpy.ndarray
        A 2D matrix containing the MNI coordinates of the vertices/ dipoles

    Note
    ----
    forward must contain some subject id in forward["src"][0]["subject_his_id"]
    in order to work.
    '''
    # Get Subjects ID
    subject_his_id = forward["src"][0]["subject_his_id"]
    src = forward["src"]

    # Extract vertex positions from left and right source space
    pos_left = mne.vertex_to_mni(src[0]["vertno"], 0, subject_his_id, verbose=verbose)
    pos_right = mne.vertex_to_mni(src[1]["vertno"], 1, subject_his_id, verbose=verbose)

    # concatenate coordinates from both hemispheres
    pos = np.concatenate([pos_left, pos_right], axis=0)

    return pos

# def compute_laura(evoked, forward, noise_cov=None, reg=200, drop_off=2,
#     verbose=0):
#     """ Calculate Local AUto-Regressive Averages (LAURA) inverse solution


#     Parameters
#     ----------
#     evoked : instance of mne.Evoked
#         The evoked data
#     forward : instance of Forward
#         The forward solution.
#     noise_cov : instance of Covariance
#         The noise covariance.
#     reg : float
#         Value that weights noise regularization. The higher, the more
#         conservative the source localization will be. As a rule of thumb you can
#         estimate the optimal reg parameter as follows: reg = 1e3 / SNR, whereas
#         SNR corresponds to the signal-to-noise ratio in your data.
#     drop_off : int/float
#         Value that is often denoted as exponent "e_i" in the literature [2,
#         Appendix] and is set to "2" at default. It corresponds to fields that
#         decrease with the square of the inverse distance. Lowering this value
#         (e.g., drop_off=1.5) will decrease sparsity of sources, whereas
#         increasing this value (e.g., drop_off=2.5) will increase sparsity.
#     verbose : bool,
#         Controls verbosity.

#     Returns
#     -------
#     stc : instance of SourceEstimate
#         The source estimates.
    
#     References
#     ---------

#     [1] Menendez, R. G. D. P., Andino, S. G., Lantz, G., Michel, C. M., &
#     Landis, T. (2001). Noninvasive localization of electromagnetic epileptic
#     activity. I. Method descriptions and simulations. Brain topography, 14(2),
#     131-137.

#     [2] de Peralta Menendez, R. G., Murray, M. M., Michel, C. M., Martuzzi, R.,
#     & Andino, S. L. G. (2004). Electrical neuroimaging based on biophysical
#     constraints. Neuroimage, 21(2), 527-539.

    
#     """
#     assert forward['surf_ori'], "dipole orientations in mne.Forward must be fixed!"
#     evoked = evoked.copy().set_eeg_reference('average', proj=True).apply_proj()
#     forward = forward.copy()

#     # make sure to select all the right channels
#     if noise_cov is not None:
#         noise_cov = noise_cov.copy()
#         ch_names = list(set(noise_cov.ch_names).intersection(evoked.ch_names))
#         ch_names = list(set(ch_names).intersection(forward.ch_names))

#         noise_cov = noise_cov.pick_channels(ch_names)
#         evoked = evoked.pick_channels(ch_names)
#         forward = forward.pick_channels(ch_names)
#     else:
#         ch_names = list(set(forward.ch_names).intersection(evoked.ch_names))
#         evoked = evoked.pick_channels(ch_names)
#         forward = forward.pick_channels(ch_names)

#     # Extract the necessary variables from the forward model
#     pos = pos_from_forward(forward, verbose=verbose)
#     leadfield = forward['sol']['data']
#     vertices = [forward["src"][0]['vertno'], forward["src"][1]['vertno']]
  
#     # Select channels of interest
#     # sel = [evoked.ch_names.index(name) for name in ch_names]
#     x = evoked.data#[sel]
#     # ensure common average reference:
#     x -= x.mean(axis=0)

#     # pairwise distance matrix of all vertex/dipole locations
#     d = cdist(pos, pos)
#     # Get the adjacency matrix of the source spaces
#     adj = mne.spatial_src_adjacency(forward["src"], verbose=verbose).toarray()
#     # set non-neighboring dipoles to zero
#     for i in range(d.shape[0]):
#         # find dipoles that are no neighbor to dipole i
#         non_neighbors = np.where(~adj.astype(bool)[i, :])[0]
#         # append dipole itself
#         non_neighbors = np.append(non_neighbors, i)
#         # set non-neighbors to zero
#         d[i, non_neighbors] = 0
#         # neighbors = np.where(adj.astype(bool)[i, :])[0]
#         # d[i, neighbors] = -d[i, neighbors]**(drop_off)

#     A = -d**-drop_off
#     A[np.isinf(A)] = 0
#     W = np.identity(A.shape[0])
#     M = np.matmul(W, A)

#     # Source Space metric
#     W_j = np.matrix(inv(np.matmul(M.T, M)))
#     W_j_inv = np.matrix(inv(W_j))

#     # Data Space metric
#     if noise_cov is None:
#         W_d = np.matrix(np.identity(leadfield.shape[0]))
#     else:
#         # check if mne.Covariance is 1D (only diagonal values of the matrix)
#         if len(noise_cov.data.shape) == 1:
#             W_d = inv(noise_cov.data*np.identity(noise_cov.data.shape[0]))
#             # W_d = noise_cov.data*np.identity(noise_cov.data.shape[0])
#         else:
#             W_d = inv(noise_cov.data)
#             # W_d = noise_cov.data


#     # Calculate noise term using regularization parameter reg and the Data Space Metric
#     noise_term = (reg**2) * inv(W_d)

#     # noise-free case
#     # G = W_j_inv * leadfield.T * inv(leadfield * W_j_inv * leadfield.T) #* leadfield.T * W_d

#     # Calculate the inverse operator G
#     G = W_j_inv * leadfield.T * inv(leadfield * W_j_inv * leadfield.T + noise_term) #* leadfield.T * W_d

#     # calculate source
#     y_hat = np.array(np.matmul(G, x))
#     stc = mne.SourceEstimate(y_hat, vertices, tmin=evoked.tmin, 
#         tstep=1/evoked.info["sfreq"], subject=forward["src"]._subject,
#         verbose=verbose)
#     return stc

def compute_laura(evoked, forward, noise_cov=None, reg=200, drop_off=2,
    verbose=0):
    """ Calculate Local AUto-Regressive Averages (LAURA) inverse solution


    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    reg : float
        Value that weights noise regularization. The higher, the more
        conservative the source localization will be. As a rule of thumb you can
        estimate the optimal reg parameter as follows: reg = 1e3 / SNR, whereas
        SNR corresponds to the signal-to-noise ratio in your data.
    drop_off : int/float
        Value that is often denoted as exponent "e_i" in the literature [2,
        Appendix] and is set to "2" at default. It corresponds to fields that
        decrease with the square of the inverse distance. Lowering this value
        (e.g., drop_off=1.5) will decrease sparsity of sources, whereas
        increasing this value (e.g., drop_off=2.5) will increase sparsity.
    verbose : bool,
        Controls verbosity.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    
    References
    ---------

    [1] Menendez, R. G. D. P., Andino, S. G., Lantz, G., Michel, C. M., &
    Landis, T. (2001). Noninvasive localization of electromagnetic epileptic
    activity. I. Method descriptions and simulations. Brain topography, 14(2),
    131-137.

    [2] de Peralta Menendez, R. G., Murray, M. M., Michel, C. M., Martuzzi, R.,
    & Andino, S. L. G. (2004). Electrical neuroimaging based on biophysical
    constraints. Neuroimage, 21(2), 527-539.

    
    """
    G = make_laura(forward, noise_cov=noise_cov, reg=reg, 
        drop_off=drop_off, verbose=verbose)
    stc = apply_laura(evoked, G, forward, verbose=0)
    return stc

def make_laura(forward, noise_cov=None, reg=200, drop_off=2,
    verbose=0):
    """ Calculate the Local AUto-Regressive Averages (LAURA) inverse operator


    Parameters
    ----------
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    reg : float
        Value that weights noise regularization. The higher, the more
        conservative the source localization will be. As a rule of thumb you can
        estimate the optimal reg parameter as follows: reg = 1e3 / SNR, whereas
        SNR corresponds to the signal-to-noise ratio in your data.
    drop_off : int/float
        Value that is often denoted as exponent "e_i" in the literature [2,
        Appendix] and is set to "2" at default. It corresponds to fields that
        decrease with the square of the inverse distance. Lowering this value
        (e.g., drop_off=1.5) will decrease sparsity of sources, whereas
        increasing this value (e.g., drop_off=2.5) will increase sparsity.
    verbose : bool,
        Controls verbosity.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    
    References
    ---------

    [1] Menendez, R. G. D. P., Andino, S. G., Lantz, G., Michel, C. M., &
    Landis, T. (2001). Noninvasive localization of electromagnetic epileptic
    activity. I. Method descriptions and simulations. Brain topography, 14(2),
    131-137.

    [2] de Peralta Menendez, R. G., Murray, M. M., Michel, C. M., Martuzzi, R.,
    & Andino, S. L. G. (2004). Electrical neuroimaging based on biophysical
    constraints. Neuroimage, 21(2), 527-539.

    
    """
    assert forward['surf_ori'], "dipole orientations in mne.Forward must be fixed!"
    forward = forward.copy()

    # make sure to select all the right channels
    if noise_cov is not None:
        noise_cov = noise_cov.copy()
        ch_names = list(set(noise_cov.ch_names).intersection(forward.ch_names))
        noise_cov = noise_cov.pick_channels(ch_names)
        forward = forward.pick_channels(ch_names)
    
    # Extract the necessary variables from the forward model
    pos = pos_from_forward(forward, verbose=verbose)
    leadfield = forward['sol']['data']
    
    
    # pairwise distance matrix of all vertex/dipole locations
    d = cdist(pos, pos)
    # Get the adjacency matrix of the source spaces
    adj = mne.spatial_src_adjacency(forward["src"], verbose=verbose).toarray()
    # set non-neighboring dipoles to zero
    for i in range(d.shape[0]):
        # find dipoles that are no neighbor to dipole i
        non_neighbors = np.where(~adj.astype(bool)[i, :])[0]
        # append dipole itself
        non_neighbors = np.append(non_neighbors, i)
        # set non-neighbors to zero
        d[i, non_neighbors] = 0
        # neighbors = np.where(adj.astype(bool)[i, :])[0]
        # d[i, neighbors] = -d[i, neighbors]**(drop_off)

    A = -d**-drop_off
    A[np.isinf(A)] = 0
    W = np.identity(A.shape[0])
    M = np.matmul(W, A)

    # Source Space metric
    W_j = np.matrix(inv(np.matmul(M.T, M)))
    W_j_inv = np.matrix(inv(W_j))

    # Data Space metric
    if noise_cov is None:
        W_d = np.matrix(np.identity(leadfield.shape[0]))
    else:
        # check if mne.Covariance is 1D (only diagonal values of the matrix)
        if len(noise_cov.data.shape) == 1:
            W_d = inv(noise_cov.data*np.identity(noise_cov.data.shape[0]))
            # W_d = noise_cov.data*np.identity(noise_cov.data.shape[0])
        else:
            W_d = inv(noise_cov.data)
            # W_d = noise_cov.data


    # Calculate noise term using regularization parameter reg and the Data Space Metric
    noise_term = (reg**2) * inv(W_d)

    # noise-free case
    # G = W_j_inv * leadfield.T * inv(leadfield * W_j_inv * leadfield.T) #* leadfield.T * W_d

    # Calculate the inverse operator G
    G = W_j_inv * leadfield.T * inv(leadfield * W_j_inv * leadfield.T + noise_term) #* leadfield.T * W_d
    return G

def apply_laura(evoked, G, forward, verbose=0):
    '''
    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked data
    G : numpy.ndarray
        The generalized inverse operator. Can be computed with the funciton
        laura.make_laura
    forward : mne.Forward
        Instance of the mne.Forward object - forward model
    
    Return
    ------
    stc : mne.SourceEstimate
        The LAURA inverse solution as instance of mne.SourceEstimate
    '''
    # Extract necessary parameters
    evoked = evoked.copy().set_eeg_reference('average', projection=True).apply_proj()
    forward = forward.copy()
    vertices = [forward["src"][0]['vertno'], forward["src"][1]['vertno']]

    # calculate source
    x = evoked.data
    
    y_hat = np.array(np.matmul(G, x))
    stc = mne.SourceEstimate(y_hat, vertices, tmin=evoked.tmin, 
        tstep=1/evoked.info["sfreq"], subject=forward["src"]._subject,
        verbose=verbose)
    return stc